#!/usr/bin/env python3
# ROS 1 node (python3, rospy). Reads our own topics and publishes JSON strings for the cloud side.
# No AWS code and no credentials here. The Pulse cloud_gateway forwards these Strings to MQTT UNCHANGED,
# so every message is wrapped in the envelope here (schemas/envelope.schema.json).
#   /cloud_gateway/telemetry  odin.pose.v1, odin.detections.yolo.v1     (MQTT QoS 0)
#   /cloud_gateway/status     odin.status.v1                            (QoS 1, retained: ONE slot, so one message holds
#                                                                        map + mission + health)
#   /cloud_gateway/events     odin.detections.florence.v1               (QoS 1)
# Settings: config/cloud_publisher.yaml (rates per kind). Use:  python3 cloud_publisher.py _config:=<yaml>
import array
import hashlib
import json
import math
import os
import threading
import time
import uuid

import rospy
import tf2_ros
import yaml
from cloud_logic import apply_tf, envelope, parse_class_names, yaw_of, yolo_signature
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

try:
    from nav_interfaces.msg import MissionStatus
except ImportError:  # ONS2 workspace not sourced
    MissionStatus = None


class CloudPublisher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.seq = {}
        self.last_seen = {}
        self.yolo_stamps = []
        self.class_names = {}
        self.speed = 0.0
        self.tf_stamp = None
        self.tf_changed = 0.0
        self.last = {"pose": 0.0, "yolo": 0.0, "status": 0.0}
        self.map_sha = None
        self.nav_sig = None
        self.yolo_sig = None
        self.map_part = None      # parts of odin.status.v1; None until the topic has been seen
        self.nav_part = None
        self.status_dirty = False
        self.map_name = cfg.get("map_name") or os.environ.get("ONS2_MAP") or "unknown"

        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        out = cfg["out"]
        self.pub = {
            "telemetry": rospy.Publisher(out["telemetry"], String, queue_size=10),
            "status": rospy.Publisher(out["status"], String, queue_size=10, latch=True),
            "events": rospy.Publisher(out["events"], String, queue_size=10),
        }
        t = cfg["topics"]
        rospy.Subscriber(t["odometry"], Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber(t["map"], OccupancyGrid, self.on_map, queue_size=1)
        if MissionStatus is not None:
            rospy.Subscriber(t["nav_status"], MissionStatus, self.on_nav, queue_size=1)
        else:
            rospy.logwarn("nav_interfaces not found: nav part of odin.status.v1 stays null")
        rospy.Subscriber(t["yolo_names"], String, self.on_names, queue_size=1)
        rospy.Subscriber(t["yolo_3d"], Detection3DArray, self.on_yolo, queue_size=1)
        rospy.Subscriber(t["florence"], String, self.on_florence, queue_size=5)

    # ---- helpers
    def send(self, channel, kind, body):
        now = time.time()
        with self.lock:
            n = self.seq.get(kind, 0)
            self.seq[kind] = n + 1
        payload = {"kind": kind, "seq": n, "t_robot": round(now, 3)}
        payload.update(body)
        msg = envelope(self.cfg["identity"], payload, now, str(uuid.uuid4()))
        self.pub[channel].publish(String(data=json.dumps(msg, separators=(",", ":"))))

    def resolve_map_name(self):
        """ONS2 keeps the map name in a ROS parameter; read it until we have it."""
        if self.map_name == "unknown":
            name = rospy.get_param(self.cfg.get("map_name_param", "/map_planner/map_file_name"), "")
            if name:
                self.map_name = str(name)

    def seen(self, name):
        self.last_seen[name] = time.time()

    def due(self, key, period):
        now = time.time()
        if period > 0 and now - self.last[key] >= period:
            self.last[key] = now
            return True
        return False

    def cam_to_map_tf(self):
        """Latest TF camera_frame -> map, like yolo_detector.py (same frame, same maths). None if missing."""
        try:
            return self.tfbuf.lookup_transform(
                self.cfg["map_frame"], self.cfg["camera_frame"], rospy.Time(0), rospy.Duration(0.05)).transform
        except Exception:
            return None

    # ---- subscribers
    def on_odom(self, m):
        self.seen("odometry")
        v = m.twist.twist.linear
        self.speed = math.hypot(v.x, v.y)

    def on_map(self, m):
        self.seen("map")
        self.resolve_map_name()
        sha = hashlib.sha256(array.array('b', m.data).tobytes()).hexdigest()
        o = m.info.origin
        self.map_part = {
            "name": self.map_name,
            "resolution_m": round(m.info.resolution, 4),
            "origin": {"x": round(o.position.x, 3), "y": round(o.position.y, 3), "yaw": round(yaw_of(o.orientation), 3)},
            "width": int(m.info.width),
            "height": int(m.info.height),
            "sha256": sha,
            "reloc_map": os.environ.get("ODIN_RELOC_MAP") or None,
            "s3_key": None,
        }
        if sha != self.map_sha:
            self.map_sha = sha
            self.status_dirty = True

    def on_nav(self, m):
        self.seen("nav_status")
        t = m.current_target.position
        self.nav_part = {
            "state": m.state,
            "mission_id": m.mission_id,
            "waypoint_idx": int(m.current_waypoint_idx),
            "total_waypoints": int(m.total_waypoints),
            "goal": {"x": round(t.x, 2), "y": round(t.y, 2)} if m.state not in ("", "IDLE") else None,
            "distance_to_target_m": round(max(0.0, m.distance_to_target_m), 2),
            "progress_percent": round(min(100.0, max(0.0, m.progress_percent)), 1),
            "linear_vel": round(m.linear_vel, 3),
            "angular_vel": round(m.angular_vel, 3),
        }
        sig = (m.state, m.mission_id, m.current_waypoint_idx)
        if sig != self.nav_sig:
            self.nav_sig = sig
            self.status_dirty = True

    def on_names(self, m):
        try:
            self.class_names = parse_class_names(m.data)
        except Exception as e:
            rospy.logwarn_throttle(30, "bad /yolo_class_names: %s" % e)

    def on_yolo(self, m):
        self.seen("yolo")
        now = time.time()
        self.yolo_stamps = [s for s in self.yolo_stamps if now - s < 5.0] + [now]
        objects = []
        tf_cam = self.cam_to_map_tf()
        for d in m.detections[: self.cfg["max_objects"]]:
            if not d.results:
                continue
            r = d.results[0]
            cid = int(getattr(r, "id", getattr(getattr(r, "hypothesis", None), "class_id", -1)))
            score = float(getattr(r, "score", getattr(getattr(r, "hypothesis", None), "score", 0.0)))
            p = d.bbox.center.position
            mx, my = apply_tf(tf_cam, p.x, p.y, p.z) if tf_cam else (None, None)
            objects.append({
                "class": self.class_names.get(cid, str(cid)),
                "confidence": round(score, 3),
                "distance_m": round(math.hypot(p.x, p.z), 2),
                "bearing_deg": round(math.degrees(math.atan2(p.x, p.z)), 1),
                "map_x": mx,
                "map_y": my,
            })
        sig = yolo_signature(objects)
        rates = self.cfg["rates_hz"]
        period = 1.0 / rates["yolo_max"] if rates["yolo_max"] > 0 else 0
        refresh = rates.get("yolo_refresh_s", 10.0)
        changed = sig != self.yolo_sig and period and now - self.last["yolo"] >= period
        if changed or (refresh > 0 and now - self.last["yolo"] >= refresh):
            self.yolo_sig = sig
            self.last["yolo"] = now
            self.send("telemetry", "odin.detections.yolo.v1",
                      {"image_stamp": round(m.header.stamp.to_sec(), 3), "objects": objects})

    def on_florence(self, m):
        self.seen("florence")
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        if not isinstance(d, dict):
            return
        if time.time() - self.t0 < 2.0:
            return  # the latched message from an earlier search arrives right at start
        stamp = float(d.get("stamp") or 0.0)
        objs = [{
            "n": int(o["n"]), "name": str(o.get("name", "")), "label": str(o.get("label", "")),
            "distance_m": round(float(o.get("dist", 0.0)), 2), "direction": str(o.get("direction", "")),
        } for o in d.get("objects", [])[: self.cfg["max_objects"]]]
        self.send("events", "odin.detections.florence.v1", {
            "phrase": str(d.get("phrase") or ""), "search_id": d.get("seq"),
            "image_stamp": round(stamp, 3), "objects": objs})

    # ---- periodic
    def pose(self):
        c = self.cfg
        self.resolve_map_name()
        body = {"map": self.map_name, "relocalized": False, "speed": round(self.speed, 3)}
        try:
            tr = self.tfbuf.lookup_transform(c["map_frame"], c["base_frame"], rospy.Time(0))
            stamp = tr.header.stamp.to_sec()
            if stamp != self.tf_stamp:  # age = time since the TF last changed, on our clock (the stamp may be the Odin1 clock)
                self.tf_stamp, self.tf_changed = stamp, time.time()
            age = time.time() - self.tf_changed
            t = tr.transform
            body.update({"x": round(t.translation.x, 3), "y": round(t.translation.y, 3),
                         "yaw": round(yaw_of(t.rotation), 3), "tf_age_ms": round(age * 1000, 1),
                         "relocalized": age < c["tf_max_age_s"]})
            self.seen("tf_map")
        except Exception:
            pass
        self.send("telemetry", "odin.pose.v1", body)

    def stack_part(self):
        now = time.time()
        inputs = {k: (round(now - self.last_seen[k], 2) if k in self.last_seen else None)
                  for k in ("odometry", "tf_map", "yolo", "nav_status", "florence")}
        images = dict(kv.split("=", 1) for kv in os.environ.get("STACK_IMAGES", "").split(",") if "=" in kv)
        part = {"inputs": inputs, "yolo_hz": round(len(self.yolo_stamps) / 5.0, 1)}
        if images:
            part["images"] = images
        if os.environ.get("STACK_COMMIT"):
            part["commit"] = os.environ["STACK_COMMIT"]
        return part

    def status_due(self):
        """Send the whole status when something changed (at most once per min_status_interval_s),
        every nav_active_refresh_s while NAVIGATING, otherwise every status_refresh_s."""
        r = self.cfg["rates_hz"]
        since = time.time() - self.last["status"]
        navigating = self.nav_part is not None and self.nav_part["state"] == "NAVIGATING"
        period = r["nav_active_refresh_s"] if navigating else r["status_refresh_s"]
        return since >= period or (self.status_dirty and since >= r["min_status_interval_s"])

    def send_status(self):
        self.resolve_map_name()
        if self.map_part:
            self.map_part["name"] = self.map_name
        self.status_dirty = False
        self.last["status"] = time.time()
        self.send("status", "odin.status.v1", {"map": self.map_part, "nav": self.nav_part, "stack": self.stack_part()})

    def spin(self):
        r = self.cfg["rates_hz"]
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if r["pose"] > 0 and self.due("pose", 1.0 / r["pose"]):
                self.pose()
            if self.status_due():
                self.send_status()
            # yolo_detector publishes nothing when it sees nothing: say so once, or viewers keep stale objects
            if self.yolo_sig not in (None, ()) and time.time() - self.last_seen.get("yolo", 0) > 1.5:
                self.yolo_sig = ()
                self.send("telemetry", "odin.detections.yolo.v1", {"objects": []})
            rate.sleep()


def main():
    rospy.init_node("cloud_publisher")
    path = rospy.get_param("~config")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    CloudPublisher(cfg).spin()


if __name__ == "__main__":
    main()
