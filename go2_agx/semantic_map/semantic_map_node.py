#!/usr/bin/env python3
"""Semantic map MVP (ROS 1 Noetic, Python 3.8; runs in the ons2-go2 container): keeps every object the detectors saw,
in the map frame, with its metadata, and publishes it. See README.md next to this file.

In:   /perception/objects      std_msgs/String JSON per frame (GPU detector, go2_agx/agx/detector): name, score, box,
                               track_id, xyz (camera frame, m), size (m)
      /florence/objects_3d     std_msgs/String JSON (Florence-2 relay): its objects with a 3D point (optional)
      /map                     nav_msgs/OccupancyGrid (map_server; optional, only for the BEV image)
      TF map <- camera frame at the photo's time stamp; map <- base_link for the robot in the BEV image
      /semantic_map/command    std_msgs/String: "clear"
Out:  /semantic_map/objects    std_msgs/String JSON (latched), ~publish_rate Hz: the stored objects (see README)
      /semantic_map/markers    visualization_msgs/MarkerArray: a sphere + label per object
      /semantic_map/bev        sensor_msgs/Image, and /semantic_map/bev/compressed (JPEG): top view on the map
                               (or a 1 m grid without a map); drawn only while someone subscribes
"""
import json
import math
import os
import sys
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bev  # noqa: E402
from object_store import ObjectStore, Params  # noqa: E402


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class SemanticMapNode:
    def __init__(self):
        g = rospy.get_param
        self.map_frame = g("~map_frame", "map")
        self.robot_frame = g("~robot_frame", "base_link")
        self.fallback_frame = g("~camera_frame", "camera_link")
        self.min_score = float(g("~min_score", 0.4))
        self.max_range = float(g("~max_range_m", 8.0))   # far lidar points on small boxes are unreliable
        self.with_embeddings = bool(g("~publish_embeddings", False))
        self.bev_window = float(g("~bev_window_m", 0.0))
        dyn = [c.strip().lower() for c in str(g("~dynamic_classes", "person,dog,cat")).split(",") if c.strip()]
        self.store = ObjectStore(Params(assoc_dist_m=float(g("~assoc_dist_m", 0.6)),
                                        min_observations=int(g("~min_observations", 3)),
                                        ttl_s=float(g("~ttl_s", 0.0)), dynamic_classes=tuple(dyn),
                                        dynamic_ttl_s=float(g("~dynamic_ttl_s", 10.0))))
        self.lock = threading.Lock()
        self.stats = {"frames": 0, "placed": 0, "no_tf": 0, "skipped": 0}
        self.latest_stamp = 0.0
        self.grid = None

        self.tfb = tf2_ros.Buffer(rospy.Duration(30.0))   # photos can be a few s old (Florence): keep 30 s of TF
        tf2_ros.TransformListener(self.tfb)
        self.pub_objects = rospy.Publisher("/semantic_map/objects", String, queue_size=1, latch=True)
        self.pub_markers = rospy.Publisher("/semantic_map/markers", MarkerArray, queue_size=1)
        self.pub_bev = rospy.Publisher("/semantic_map/bev", Image, queue_size=1)
        self.pub_bev_c = rospy.Publisher("/semantic_map/bev/compressed", CompressedImage, queue_size=1)
        rospy.Subscriber(g("~objects_topic", "/perception/objects"), String, self._on_perception, queue_size=5)
        if g("~use_florence", True):
            rospy.Subscriber("/florence/objects_3d", String, self._on_florence, queue_size=2)
        rospy.Subscriber(g("~map_topic", "/map"), OccupancyGrid, self._on_map, queue_size=1)
        rospy.Subscriber("/semantic_map/command", String, self._on_command, queue_size=5)
        rospy.Timer(rospy.Duration(1.0 / float(g("~publish_rate", 1.0))), self._publish)
        rospy.loginfo("[semantic_map] ready: %s -> %s, min score %.2f, merge within %.2f m, publish after %d sightings"
                      % (g("~objects_topic", "/perception/objects"), self.map_frame, self.min_score,
                         self.store.p.assoc_dist_m, self.store.p.min_observations))

    # ------------------------------------------------------------------ inputs
    def _to_map(self, frame, stamp, xyz):
        """Camera point -> map point with the transform at the photo's time. None when TF has no answer."""
        try:
            t = self.tfb.lookup_transform(self.map_frame, frame, rospy.Time.from_sec(stamp), rospy.Duration(0.2)).transform
        except Exception:
            return None
        R, tr = quat_to_R(t.rotation), np.array([t.translation.x, t.translation.y, t.translation.z])
        return R.dot(np.asarray(xyz, np.float64)) + tr

    def _ingest(self, frame, stamp, items, source):
        """items: [{"label", "score", "xyz", optional "size", "track_id", "box", "embedding"}] in the camera frame."""
        with self.lock:
            self.stats["frames"] += 1
            self.latest_stamp = max(self.latest_stamp, stamp)
            for it in items:
                xyz = it.get("xyz")
                if xyz is None or it["score"] < self.min_score or math.hypot(xyz[0], xyz[2]) > self.max_range:
                    self.stats["skipped"] += 1
                    continue
                p = self._to_map(frame, stamp, xyz)
                if p is None:
                    self.stats["no_tf"] += 1
                    continue
                self.store.ingest({"label": it["label"], "score": it["score"], "position": p, "source": source,
                                   "size": it.get("size"), "track_id": it.get("track_id"), "box": it.get("box"),
                                   "distance": round(math.hypot(xyz[0], xyz[2]), 2), "embedding": it.get("embedding")},
                                  stamp)
                self.stats["placed"] += 1

    def _on_perception(self, m):
        d = json.loads(m.data)
        items = [{"label": o["name"], "score": float(o["score"]), "xyz": o.get("xyz"), "size": o.get("size"),
                  "track_id": o.get("track_id"), "box": o.get("box"), "embedding": o.get("embedding")}
                 for o in d.get("objects", [])]
        self._ingest(d.get("frame_id") or self.fallback_frame, float(d["stamp"]), items, d.get("detector", "detector"))

    def _on_florence(self, m):
        d = json.loads(m.data)
        if d.get("cleared") or not d.get("objects"):
            return   # Florence "clean" only removes its F objects from the YOLO bridge, not from the map
        items = [{"label": o.get("label") or d.get("phrase") or o.get("name", "object"), "score": 0.9,
                  "xyz": [o["x"], o["y"], o["z"]], "box": o.get("box")} for o in d["objects"]]
        self._ingest(d.get("frame") or self.fallback_frame, float(d.get("stamp") or 0.0), items, "florence")

    def _on_map(self, m):
        info = m.info
        self.grid = {"data": np.asarray(m.data, np.int8).reshape(info.height, info.width),
                     "resolution": info.resolution, "origin": (info.origin.position.x, info.origin.position.y)}

    def _on_command(self, m):
        if m.data.strip().lower() == "clear":
            with self.lock:
                self.store.clear()
            rospy.loginfo("[semantic_map] cleared")
            self._publish(None)

    # ------------------------------------------------------------------ outputs
    def _robot(self):
        try:
            t = self.tfb.lookup_transform(self.map_frame, self.robot_frame, rospy.Time(0), rospy.Duration(0.1)).transform
        except Exception:
            return None
        q = t.rotation
        return t.translation.x, t.translation.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def _publish(self, _event):
        with self.lock:
            self.store.prune(self.latest_stamp)
            doc = self.store.to_dict(self.latest_stamp, self.map_frame, self.with_embeddings)
            doc["stats"] = dict(self.stats)
        self.pub_objects.publish(String(data=json.dumps(doc)))
        self.pub_markers.publish(self._markers(doc["objects"]))
        if self.pub_bev.get_num_connections() or self.pub_bev_c.get_num_connections():
            img = bev.render(doc["objects"], self.grid, self._robot(), self.bev_window)
            stamp = rospy.Time.now()
            if self.pub_bev.get_num_connections():
                msg = Image(height=img.shape[0], width=img.shape[1], encoding="bgr8", step=img.shape[1] * 3,
                            data=img.tobytes())
                msg.header.stamp, msg.header.frame_id = stamp, self.map_frame
                self.pub_bev.publish(msg)
            if self.pub_bev_c.get_num_connections():
                c = CompressedImage(format="jpeg", data=cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                c.header.stamp, c.header.frame_id = stamp, self.map_frame
                self.pub_bev_c.publish(c)

    def _markers(self, objects):
        arr = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for o in objects:
            b, g_, r = bev.label_color(o["label"])
            for kind in (Marker.SPHERE, Marker.TEXT_VIEW_FACING):
                mk = Marker(type=kind, action=Marker.ADD, id=o["id"] * 2 + (kind == Marker.TEXT_VIEW_FACING))
                mk.header.frame_id, mk.ns = self.map_frame, "semantic_map"
                mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = o["position"]
                mk.pose.orientation.w = 1.0
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = r / 255.0, g_ / 255.0, b / 255.0, 0.85
                if kind == Marker.SPHERE:
                    d = max(o["size"][0] if o["size"] else 0.3, 0.15)
                    mk.scale.x = mk.scale.y = mk.scale.z = d
                else:
                    mk.pose.position.z += 0.4
                    mk.scale.z = 0.2
                    mk.text = "#%d %s (%d)" % (o["id"], o["label"], o["observations"])
                arr.markers.append(mk)
        return arr


if __name__ == "__main__":
    rospy.init_node("semantic_map")
    SemanticMapNode()
    rospy.spin()
