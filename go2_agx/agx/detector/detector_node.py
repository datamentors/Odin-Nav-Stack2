#!/usr/bin/env python3
"""ROS 1 side of the GPU detector (process "yolo" in the container odin-agx, node name /yolo_detector). Works with any
detector/tracker of detector_service.py (YOLOv5, RF-DETR; none, IoU, BoT-SORT, SAM 2). Python 3.8, no torch here.

Per camera frame (only the newest is processed; older ones are dropped while the model runs):
  image -> resized to the model size -> POST /detect -> boxes back to full resolution -> 3D from the lidar depth image
  closest in time (depth3d.py) -> topics.

Inputs
  ~rgb_topic   /odin1/image/undistorted (Image, or CompressedImage if the name ends in /compressed)
  ~depth_topic /odin1/depth_img_competetion (pcd2depth: lidar projected into the same image)
  /yolo/set_conf             std_msgs/Float32   confidence threshold, live
  /perception/target/select  std_msgs/String    JSON, one of (boxes in full-resolution pixels):
        {"box": [x1, y1, x2, y2], "stamp": <s>, "name": "chair"}   e.g. a Florence-2 box (Florence command "track N")
        {"name": "person"}   {"track_id": 3}   {"clear": true}

Outputs, /perception/* (the complete set; topics nobody subscribes to cost nothing)
  /perception/objects        std_msgs/String    JSON, every frame (the full result, for any consumer / streaming):
        {"stamp", "frame_id", "image_size": [w, h], "detector", "tracker", "target": {"id", "visible"} | null,
         "objects": [{"name", "cls", "score", "box": [x1, y1, x2, y2] (px), "track_id" | null, "target",
                      "source": "detector" | "sam2", "mask_index" | null, "xyz": [x, y, z] (m, camera frame) | null,
                      "size": [w, h] (m, at that depth) | null}]}
  /perception/detections     vision_msgs/Detection2DArray  all objects (results[0].id = class id, -1 = unknown class)
  /perception/detections_3d  vision_msgs/Detection3DArray  the objects with depth: centre + metric width/height
  /perception/class_names    std_msgs/String (latched)     JSON {class id: name} of the running detector
  /perception/masks          sensor_msgs/Image mono8       instance label image at full resolution, same header as the
                                                           camera frame: pixel = mask_index of an object, 0 = none
  /perception/masks/compressed  sensor_msgs/CompressedImage  the same as PNG
  /perception/debug_image(/compressed)  the camera frame at full resolution with masks, boxes, class, score, track ID,
                                        distance and the target in red
  /perception/target_3d      geometry_msgs/PointStamped   the target's 3D point, when it is seen with depth
  /perception/target/status  std_msgs/String (latched)    result of the last selection (JSON)
Outputs, upstream yolo_ros topics (detector objects only; ~publish_yolo_topics, default true), so object_query_node,
yolo_cmd_bridge and the Foxglove layout keep working:
  /yolo_detections, /yolo_detections_3d (only frames with a 3D object, upstream behaviour), /yolo_class_names,
  /yolo_debug_image (= the full-resolution debug image), /yolo_debug_image/compressed (800 px wide, for Wi-Fi)
  /yolo/console              std_msgs/String               short answers (confidence, target selection)
3D outputs use the camera image's frame_id (or ~camera_frame when empty); x right, y down, z forward.

Params: ~service_url (http://127.0.0.1:8750), ~conf_thres (0.25), ~camera_frame (camera_link), ~calib_file
(/calib.yaml), ~depth_percentile (20), ~ignore_classes (comma separated, never reported unless it is the target),
~publish_yolo_topics (true).
"""
import base64
import collections
import json
import threading
import time
import urllib.request

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, String
from vision_msgs.msg import (Detection2D, Detection2DArray, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

import depth3d

# Classes never seen indoors: only false positives here (a chair from below -> "airplane", ceiling lights -> "car"...)
DEFAULT_IGNORE = ("airplane,bus,train,truck,boat,car,motorcycle,traffic light,fire hydrant,stop sign,parking meter,bird,"
                  "horse,sheep,cow,elephant,bear,zebra,giraffe,surfboard,skis,snowboard,kite,skateboard")


class Service:
    def __init__(self, url):
        self.url = url

    def get(self, path, timeout=3):
        return json.loads(urllib.request.urlopen(self.url + path, timeout=timeout).read())

    def post(self, path, data, headers=None, timeout=5):
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {}, method="POST")
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


class DetectorNode:
    def __init__(self):
        p = rospy.get_param
        self.svc = Service(p("~service_url", p("~gpu_url", "http://127.0.0.1:8750")))
        self.conf = float(p("~conf_thres", 0.25))
        self.camera_frame = p("~camera_frame", "camera_link")
        self.percentile = float(p("~depth_percentile", 20.0))
        self.ignore = {c.strip().lower() for c in str(p("~ignore_classes", DEFAULT_IGNORE)).split(",") if c.strip()}
        self.yolo_topics = bool(p("~publish_yolo_topics", True))
        self.bridge = CvBridge()
        while not rospy.is_shutdown():   # the model may still be loading
            try:
                self.info = self.svc.get("/info")
                break
            except Exception as e:
                rospy.logwarn_throttle(10, "[detector] waiting for the service %s (%s)" % (self.svc.url, e))
                time.sleep(2)
        self.names = {int(k): v for k, v in self.info["names"].items()}
        try:
            self.K = depth3d.load_intrinsics(p("~calib_file", "/calib.yaml"))
        except Exception as e:
            self.K = None
            rospy.logwarn("[detector] calib.yaml not read (%s): no 3D positions" % e)

        pub = lambda topic, cls, latch=False: rospy.Publisher(topic, cls, queue_size=1, latch=latch)
        names = String(data=json.dumps({str(k): v for k, v in self.names.items()}))
        self.pub_objects = pub("/perception/objects", String)
        self.pub2d = pub("/perception/detections", Detection2DArray)
        self.pub3d = pub("/perception/detections_3d", Detection3DArray)
        self.pub_mask = pub("/perception/masks", Image)
        self.pub_mask_c = pub("/perception/masks/compressed", CompressedImage)
        self.pub_dbg = pub("/perception/debug_image", Image)
        self.pub_dbg_c = pub("/perception/debug_image/compressed", CompressedImage)
        self.pub_target = pub("/perception/target_3d", PointStamped)
        self.pub_status = pub("/perception/target/status", String, True)
        pub("/perception/class_names", String, True).publish(names)
        self.pub_console = pub("/yolo/console", String, True)
        if self.yolo_topics:
            self.y2d = pub("/yolo_detections", Detection2DArray)
            self.y3d = pub("/yolo_detections_3d", Detection3DArray)
            self.ydbg = pub("/yolo_debug_image", Image)
            self.ydbg_c = pub("/yolo_debug_image/compressed", CompressedImage)
            pub("/yolo_class_names", String, True).publish(names)

        self.depth = collections.deque(maxlen=25)   # (stamp s, msg): ~5-10 s of lidar depth images
        self.scale = 1.0                             # image sent to the service = full image * scale
        self._latest, self._lock, self._event = None, threading.Lock(), threading.Event()
        rgb = p("~rgb_topic", "/odin1/image/undistorted")
        rospy.Subscriber(rgb, CompressedImage if rgb.endswith("/compressed") else Image, self._on_image,
                         queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(p("~depth_topic", "/odin1/depth_img_competetion"), Image,
                         lambda m: self.depth.append((m.header.stamp.to_sec(), m)), queue_size=2, buff_size=2 ** 25)
        rospy.Subscriber("/yolo/set_conf", Float32, self._on_conf)
        rospy.Subscriber("/perception/target/select", String, self._on_select)
        threading.Thread(target=self._worker, name="detector_worker", daemon=True).start()
        rospy.loginfo("[detector] %s | tracker %s | initialized with %d classes, input %d px"
                      % (self.info["detector"], self.info["tracker"], len(self.names), self.info["input_size"]))

    # ------------------------------------------------------------------ inputs
    def _on_image(self, msg):   # only keep the newest frame; the worker drops the ones it had no time for
        with self._lock:
            self._latest = msg
        self._event.set()

    def _on_conf(self, m):
        self.conf = min(max(float(m.data), 0.01), 0.99)
        rospy.loginfo("[detector] confidence threshold -> %.2f", self.conf)
        self.pub_console.publish(String(data="Detector confidence threshold set to %.2f (%d %%)"
                                        % (self.conf, round(self.conf * 100))))

    def _on_select(self, m):
        try:
            req = json.loads(m.data)
            if "box" in req:   # full-resolution pixels -> pixels of the images the service gets
                req["box"] = [float(v) * self.scale for v in req["box"]]
                ids = [k for k, v in self.names.items() if v.lower() == str(req.get("name", "")).lower()]
                if ids and "cls" not in req:
                    req["cls"] = ids[0]
            res = self.svc.post("/target", json.dumps(req).encode(), {"Content-Type": "application/json"})
        except Exception as e:
            res = {"ok": False, "target": None, "detail": "selection failed: %s" % e}
        self.pub_status.publish(String(data=json.dumps(res)))
        self.pub_console.publish(String(data="[detector] target: " + res.get("detail", "")))

    def _depth_at(self, stamp, shape):
        items = list(self.depth)
        if self.K is None or not items:
            return None
        _, m = min(items, key=lambda it: abs(it[0] - stamp))
        d = depth3d.depth_to_m(self.bridge.imgmsg_to_cv2(m, desired_encoding="passthrough"), m.encoding)
        if d.shape[:2] != shape:
            rospy.logwarn_throttle(30, "[detector] depth %dx%d does not match the image %dx%d: no 3D"
                                   % (d.shape[1], d.shape[0], shape[1], shape[0]))
            return None
        return d

    # ------------------------------------------------------------------ per frame
    def _worker(self):
        while not rospy.is_shutdown():
            if not self._event.wait(0.5):
                continue
            with self._lock:
                msg, self._latest = self._latest, None
                self._event.clear()
            if msg is not None:
                try:
                    self._process(msg)
                except Exception as e:
                    rospy.logerr_throttle(5, "[detector] processing error: %s" % e)

    def _process(self, msg):
        if isinstance(msg, CompressedImage):
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        else:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        H, W = img.shape[:2]
        s = min(1.0, float(self.info["input_size"]) / max(H, W))
        small = cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
        self.scale = s
        stamp = msg.header.stamp.to_sec()
        want_dbg = self._subs(self.pub_dbg, self.pub_dbg_c) or (self.yolo_topics and self._subs(self.ydbg, self.ydbg_c))
        want_mask = want_dbg or self._subs(self.pub_mask, self.pub_mask_c)
        try:
            res = self.svc.post("/detect?conf=%.3f&stamp=%.6f&masks=%d" % (self.conf, stamp, int(want_mask)),
                                np.ascontiguousarray(small).tobytes(),
                                {"X-Shape": "%d,%d" % small.shape[:2], "Content-Type": "application/octet-stream"})
        except Exception as e:
            rospy.logwarn_throttle(5, "[detector] request to %s failed: %s" % (self.svc.url, e))
            return
        depth = self._depth_at(stamp, (H, W))
        frame_id = msg.header.frame_id or self.camera_frame
        objs = []
        for o in res["objects"]:
            if o["name"].lower() in self.ignore and not o["target"]:
                continue
            o = dict(o, box=[round(v / s, 1) for v in o["box"]], xyz=None, size=None)   # full-resolution px
            p = depth3d.box_to_3d(self.K, depth, o["box"], self.percentile) if depth is not None else None
            if p is not None:
                o["xyz"] = [round(v, 3) for v in p]
                o["size"] = [round((o["box"][2] - o["box"][0]) * p[2] / self.K["fx"], 3),
                             round((o["box"][3] - o["box"][1]) * p[2] / self.K["fy"], 3)]
            objs.append(o)

        self.pub_objects.publish(String(data=json.dumps({
            "stamp": stamp, "frame_id": frame_id, "image_size": [W, H], "detector": self.info["detector"],
            "tracker": self.info["tracker"], "target": res["target"], "objects": objs})))
        tgt = next((o for o in objs if o["target"] and o["xyz"]), None)
        if tgt is not None:
            pt = PointStamped()
            pt.header = msg.header
            pt.header.frame_id = frame_id
            pt.point.x, pt.point.y, pt.point.z = tgt["xyz"]
            self.pub_target.publish(pt)
        a2, a3 = self._vision_msgs(objs, msg.header, frame_id)
        self.pub2d.publish(a2)
        self.pub3d.publish(a3)
        if self.yolo_topics:   # upstream topics: detector objects only, 3D only when there is one
            y2, y3 = self._vision_msgs([o for o in objs if o["source"] == "detector"], msg.header, frame_id)
            self.y2d.publish(y2)
            if y3.detections:
                self.y3d.publish(y3)
        label = None
        if want_mask and res.get("mask_png"):
            small_label = cv2.imdecode(np.frombuffer(base64.b64decode(res["mask_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
            lut = np.zeros(256, np.uint8)   # objects dropped here (ignored classes) leave no mask
            for o in objs:
                if o.get("mask_index"):
                    lut[o["mask_index"]] = o["mask_index"]
            label = cv2.resize(lut[small_label], (W, H), interpolation=cv2.INTER_NEAREST)
            self._publish_masks(label, msg.header)
        if want_dbg:
            self._publish_debug(img, objs, label, msg.header, res.get("ms", 0.0))

    @staticmethod
    def _subs(*pubs):
        return any(p.get_num_connections() > 0 for p in pubs)

    def _publish_masks(self, label, header):
        if self.pub_mask.get_num_connections():
            m = self.bridge.cv2_to_imgmsg(label, "mono8")
            m.header = header
            self.pub_mask.publish(m)
        if self.pub_mask_c.get_num_connections():
            c = CompressedImage(header=header, format="mono8; png compressed")
            c.data = cv2.imencode(".png", label)[1].tobytes()
            self.pub_mask_c.publish(c)

    def _vision_msgs(self, objs, header, frame_id):
        a2, a3 = Detection2DArray(header=header), Detection3DArray(header=header)
        a3.header.frame_id = frame_id
        for o in objs:
            b = o["box"]
            hyp = ObjectHypothesisWithPose(id=int(o["cls"]), score=float(o["score"]))
            d2 = Detection2D()
            d2.bbox.center.x, d2.bbox.center.y = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            d2.bbox.size_x, d2.bbox.size_y = b[2] - b[0], b[3] - b[1]
            d2.results.append(hyp)
            a2.detections.append(d2)
            if o["xyz"] is not None:
                d3 = Detection3D()
                d3.results.append(hyp)
                c = d3.bbox.center
                c.position.x, c.position.y, c.position.z = o["xyz"]
                c.orientation.w = 1.0
                d3.bbox.size.x, d3.bbox.size.y, d3.bbox.size.z = max(0.01, o["size"][0]), max(0.01, o["size"][1]), 0.01
                a3.detections.append(d3)
        return a2, a3

    @staticmethod
    def _color(o):
        if o["target"]:
            return (0, 0, 255)
        if o["track_id"] is None:
            return (0, 200, 0)
        r = np.random.RandomState(o["track_id"] * 7 + 3).randint(60, 255, 3)   # stable colour per track ID
        return int(r[0]), int(r[1]), int(r[2])

    def _publish_debug(self, img, objs, label, header, ms):
        """The camera frame at full resolution: masks (tinted), boxes, '#ID class score distance', target in red."""
        vis = img.copy()
        if label is not None:
            for o in objs:
                if o.get("mask_index"):
                    m = label == o["mask_index"]
                    vis[m] = (0.5 * vis[m] + 0.5 * np.array(self._color(o))).astype(np.uint8)
        fs, th = max(0.5, img.shape[1] / 1600.0), max(1, int(round(img.shape[1] / 800.0)))   # text/lines scale with size
        for o in sorted(objs, key=lambda o: o["target"]):   # the target last, on top
            c = self._color(o)
            text = ("#%d " % o["track_id"] if o["track_id"] is not None else "") + "%s %.2f" % (o["name"], o["score"])
            if o["xyz"]:
                text += " %.1fm" % o["xyz"][2]
            x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, th * (2 if o["target"] else 1))
            cv2.putText(vis, text, (x1, max(int(20 * fs), y1 - 6)), 0, 0.7 * fs, c, th, cv2.LINE_AA)
        cv2.putText(vis, "%s | %s | %.0f ms" % (self.info["detector"], self.info["tracker"], ms), (10, int(30 * fs)),
                    0, 0.7 * fs, (255, 255, 255), th, cv2.LINE_AA)
        raw = None
        for p_raw, p_jpg, width in ((self.pub_dbg, self.pub_dbg_c, None),
                                    (self.ydbg, self.ydbg_c, 800) if self.yolo_topics else (None, None, None)):
            if p_raw is not None and p_raw.get_num_connections():
                if raw is None:
                    raw = self.bridge.cv2_to_imgmsg(vis, "bgr8")
                    raw.header = header
                p_raw.publish(raw)
            if p_jpg is not None and p_jpg.get_num_connections():
                out = vis if width is None else cv2.resize(vis, (width, int(vis.shape[0] * width / vis.shape[1])))
                c = CompressedImage(header=header, format="jpeg")
                c.data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                p_jpg.publish(c)


if __name__ == "__main__":
    rospy.init_node("yolo_detector")
    DetectorNode()
    rospy.spin()
