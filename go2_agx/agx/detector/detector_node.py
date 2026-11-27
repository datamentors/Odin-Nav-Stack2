#!/usr/bin/env python3
"""ROS 1 side of the GPU detector (process "yolo" in the container odin-agx, node /yolo_detector). Python 3.8, no torch.
Topics, models and launch: README.md next to this file.

Per camera frame (only the newest; older ones are dropped while the model runs): image -> downsampled for the models
-> POST /detect (detector_service.py) -> boxes and masks back to full resolution -> 3D from the lidar depth image closest
in time (depth3d.py) -> /perception/* and the upstream /yolo_* topics.

/perception/objects (std_msgs/String) schema, one message per frame:
  {"stamp", "frame_id", "image_size": [w, h], "detector", "tracker", "target": {"id", "visible"} | null,
   "objects": [{"name", "cls", "score", "box": [x1, y1, x2, y2] (px), "track_id" | null, "target",
                "source": "detector" | "sam2", "mask_index" | null, "xyz": [x, y, z] (m, camera frame) | null,
                "size": [w, h] (m) | null}]}
3D outputs use the camera image's frame_id (or ~camera_frame when empty); x right, y down, z forward.

Params: ~service_url (http://127.0.0.1:8750), ~conf_thres (0.25), ~scale_percent (0 = auto: the detector's input size;
1-100 = % of the camera image), ~camera_frame (camera_link), ~calib_file (/calib.yaml), ~depth_percentile (20),
~ignore_classes (never reported unless it is the target; not applied to an open-vocabulary detector),
~publish_yolo_topics (true), ~rgb_topic (/odin1/image/undistorted), ~depth_topic (/odin1/depth_img_competetion).
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
TARGET_COLOR, UNTRACKED_COLOR = (0, 0, 255), (0, 200, 0)


class Service:
    def __init__(self, url):
        self.url = url

    def get(self, path, timeout=3):
        return json.loads(urllib.request.urlopen(self.url + path, timeout=timeout).read())

    def post(self, path, data, headers=None, timeout=5):
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {}, method="POST")
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def post_json(self, path, obj, timeout=5):
        return self.post(path, json.dumps(obj).encode(), {"Content-Type": "application/json"}, timeout)


class DetectorNode:
    def __init__(self):
        p = rospy.get_param
        self.svc = Service(p("~service_url", "http://127.0.0.1:8750"))
        self.conf = float(p("~conf_thres", 0.25))
        self.scale_percent = float(p("~scale_percent", 0) or 0)
        self.camera_frame = p("~camera_frame", "camera_link")
        self.percentile = float(p("~depth_percentile", 20.0))
        self.ignore_cfg = {c.strip().lower() for c in str(p("~ignore_classes", DEFAULT_IGNORE)).split(",") if c.strip()}
        self.yolo_topics = bool(p("~publish_yolo_topics", True))
        self.bridge = CvBridge()
        try:
            self.K = depth3d.load_intrinsics(p("~calib_file", "/calib.yaml"))
        except Exception as e:
            self.K = None
            rospy.logwarn("[detector] calib.yaml not read (%s): no 3D positions" % e)

        pub = lambda topic, cls, latch=False: rospy.Publisher(topic, cls, queue_size=1, latch=latch)
        self.pub_objects = pub("/perception/objects", String)
        self.pub2d = pub("/perception/detections", Detection2DArray)
        self.pub3d = pub("/perception/detections_3d", Detection3DArray)
        self.pub_mask = pub("/perception/masks", Image)
        self.pub_mask_c = pub("/perception/masks/compressed", CompressedImage)
        self.pub_target = pub("/perception/target_3d", PointStamped)
        self.pub_status = pub("/perception/target/status", String, True)
        self.pub_query = pub("/perception/query/status", String, True)
        self.pub_console = pub("/yolo/console", String, True)
        self.pub_names = [pub("/perception/class_names", String, True)]
        self.debug_outs = [(pub("/perception/debug_image", Image), pub("/perception/debug_image/compressed", CompressedImage), None)]
        if self.yolo_topics:
            self.y2d = pub("/yolo_detections", Detection2DArray)
            self.y3d = pub("/yolo_detections_3d", Detection3DArray)
            self.pub_names.append(pub("/yolo_class_names", String, True))
            self.debug_outs.append((pub("/yolo_debug_image", Image), pub("/yolo_debug_image/compressed", CompressedImage), 800))

        while not rospy.is_shutdown():   # the model may still be loading
            try:
                self._refresh_names()
                break
            except Exception as e:
                rospy.logwarn_throttle(10, "[detector] waiting for the service %s (%s)" % (self.svc.url, e))
                time.sleep(2)

        self.depth = collections.deque(maxlen=25)   # (stamp s, msg): ~5-10 s of lidar depth images
        self._colors = {}
        self._latest, self._lock, self._event = None, threading.Lock(), threading.Event()
        rgb = p("~rgb_topic", "/odin1/image/undistorted")
        rospy.Subscriber(rgb, CompressedImage if rgb.endswith("/compressed") else Image, self._on_image,
                         queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(p("~depth_topic", "/odin1/depth_img_competetion"), Image,
                         lambda m: self.depth.append((m.header.stamp.to_sec(), m)), queue_size=2, buff_size=2 ** 25)
        rospy.Subscriber("/yolo/set_conf", Float32, self._on_conf)
        rospy.Subscriber("/perception/set_scale", Float32, self._on_scale)
        rospy.Subscriber("/perception/target/select", String, self._on_select)
        rospy.Subscriber("/perception/query", String, self._on_query)
        threading.Thread(target=self._worker, name="detector_worker", daemon=True).start()
        rospy.loginfo("[detector] %s | tracker %s | initialized with %d classes, input %d px"
                      % (self.info["detector"], self.info["tracker"], len(self.names), self.info["input_size"]))

    def _refresh_names(self):
        """(Re)load the service info and publish the class names (start, and after a query changed the classes)."""
        self.info = self.svc.get("/info")
        self.names = {int(k): v for k, v in self.info["names"].items()}
        self.ignore = set() if self.info.get("open_vocab") else self.ignore_cfg   # open vocabulary: you chose the classes
        msg = String(data=json.dumps({str(k): v for k, v in self.names.items()}))
        for p_ in self.pub_names:
            p_.publish(msg)

    # ------------------------------------------------------------------ inputs
    def _on_image(self, msg):   # only keep the newest frame; the worker drops the ones it had no time for
        with self._lock:
            self._latest = msg
        self._event.set()

    def _say(self, text):
        self.pub_console.publish(String(data=text))

    def _on_conf(self, m):
        self.conf = min(max(float(m.data), 0.01), 0.99)
        self._say("Detector confidence threshold set to %.2f (%d %%)" % (self.conf, round(self.conf * 100)))

    def _on_scale(self, m):
        self.scale_percent = min(max(float(m.data), 0.0), 100.0)
        self._say("Detector image size set to " + ("auto (detector input size)" if self.scale_percent <= 0
                                                   else "%.0f %% of the camera image" % self.scale_percent))

    def _call(self, path, req, status_pub, timeout=5):
        """POST a JSON request to the service, publish the answer on status_pub and /yolo/console."""
        try:
            res = self.svc.post_json(path, req, timeout)
        except Exception as e:
            res = {"ok": False, "detail": "%s failed: %s" % (path, e)}
        status_pub.publish(String(data=json.dumps(res)))
        self._say("[detector] " + res.get("detail", ""))
        return res

    def _on_select(self, m):
        req = json.loads(m.data)
        if "box" in req:   # full-resolution px; the service converts them with the scale of that frame
            req["box_scale"] = "full"
            ids = [k for k, v in self.names.items() if v.lower() == str(req.get("name", "")).lower()]
            if ids and "cls" not in req:
                req["cls"] = ids[0]
        self._call("/target", req, self.pub_status)

    def _on_query(self, m):
        """'chair, fire extinguisher' replaces the classes, '+x, y' adds, '-x' removes (open-vocabulary detector)."""
        q = (m.data or "").strip()
        words = lambda t: [w.strip() for w in t.split(",") if w.strip()]
        req = {"add": words(q[1:])} if q.startswith("+") else {"remove": words(q[1:])} if q.startswith("-") else {"names": words(q)}
        if self._call("/classes", req, self.pub_query, timeout=60).get("ok"):
            self._refresh_names()

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

    def _depth_at(self, stamp, shape):
        """Raw lidar depth image closest in time + metres per unit, or (None, 1) (depth3d converts only the boxes)."""
        it = depth3d.nearest(self.depth, stamp) if self.K is not None else None
        if it is None:
            return None, 1.0
        m = it[1]
        d = self.bridge.imgmsg_to_cv2(m, desired_encoding="passthrough")
        if d.shape[:2] != shape:
            rospy.logwarn_throttle(30, "[detector] depth %dx%d does not match the image %dx%d: no 3D"
                                   % (d.shape[1], d.shape[0], shape[1], shape[0]))
            return None, 1.0
        return d, depth3d.units_of(m.encoding)

    def _process(self, msg):
        if isinstance(msg, CompressedImage):
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        else:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        H, W = img.shape[:2]
        s = min(1.0, self.scale_percent / 100.0 if self.scale_percent > 0 else float(self.info["input_size"]) / max(H, W))
        small = cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
        s = small.shape[1] / float(W)   # exact factor after rounding
        stamp = msg.header.stamp.to_sec()
        subs = lambda *pubs: any(p_.get_num_connections() > 0 for p_ in pubs)
        dbg = [(raw, jpg, w) for raw, jpg, w in self.debug_outs if subs(raw, jpg)]
        want_mask = bool(dbg) or subs(self.pub_mask, self.pub_mask_c)
        try:
            res = self.svc.post("/detect?conf=%.3f&stamp=%.6f&masks=%d&scale=%.6f" % (self.conf, stamp, int(want_mask), s),
                                memoryview(np.ascontiguousarray(small)).cast("B"),   # flat byte view, no copy
                                {"X-Shape": "%d,%d" % small.shape[:2], "Content-Type": "application/octet-stream"})
        except Exception as e:
            rospy.logwarn_throttle(5, "[detector] request to %s failed: %s" % (self.svc.url, e))
            return
        if res.get("names_version", 0) != self.info.get("names_version", 0):
            self._refresh_names()   # classes changed (a query from another client)
        frame_id = msg.header.frame_id or self.camera_frame
        depth = units = None
        objs = []
        for o in res["objects"]:
            if o["name"].lower() in self.ignore and not o["target"]:
                continue
            if depth is None:   # decoded only when there is something to place
                depth, units = self._depth_at(stamp, (H, W))
                depth = depth if depth is not None else False
            o = dict(o, box=[round(v / s, 1) for v in o["box"]], xyz=None, size=None)   # full-resolution px
            p = depth3d.box_to_3d(self.K, depth, o["box"], self.percentile, units) if depth is not False else None
            if p is not None:
                o["xyz"] = [round(v, 3) for v in p]
                o["size"] = [round(v, 3) for v in depth3d.box_size_m(self.K, o["box"], p[2])]
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
        self._publish_detections(objs, msg.header, frame_id)
        label = None
        if want_mask and res.get("mask_png"):
            small_label = cv2.imdecode(np.frombuffer(base64.b64decode(res["mask_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
            lut = np.zeros(256, np.uint8)   # objects dropped here (ignored classes) leave no mask
            for o in objs:
                if o["mask_index"]:
                    lut[o["mask_index"]] = o["mask_index"]
            label = cv2.resize(lut[small_label], (W, H), interpolation=cv2.INTER_NEAREST)
            self._publish_image(self.pub_mask, self.pub_mask_c, label, msg.header, "mono8", ".png", "mono8; png compressed")
        if dbg:
            self._publish_debug(img, objs, label, msg.header, res.get("ms", 0.0), dbg)

    def _publish_detections(self, objs, header, frame_id):
        """vision_msgs for every object; the /yolo_* copies hold the detector objects only (upstream consumers)."""
        a2, a3 = Detection2DArray(header=header), Detection3DArray(header=header)
        y2, y3 = Detection2DArray(header=header), Detection3DArray(header=header)
        a3.header.frame_id = y3.header.frame_id = frame_id
        for o in objs:
            b = o["box"]
            hyp = ObjectHypothesisWithPose(id=int(o["cls"]), score=float(o["score"]))
            d2 = Detection2D()
            d2.bbox.center.x, d2.bbox.center.y = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            d2.bbox.size_x, d2.bbox.size_y = b[2] - b[0], b[3] - b[1]
            d2.results.append(hyp)
            d3 = None
            if o["xyz"] is not None:
                d3 = Detection3D()
                d3.results.append(hyp)
                c = d3.bbox.center
                c.position.x, c.position.y, c.position.z = o["xyz"]
                c.orientation.w = 1.0
                d3.bbox.size.x, d3.bbox.size.y, d3.bbox.size.z = max(0.01, o["size"][0]), max(0.01, o["size"][1]), 0.01
            for arr2, arr3, keep in ((a2, a3, True), (y2, y3, o["source"] == "detector")):
                if keep:
                    arr2.detections.append(d2)
                    if d3 is not None:
                        arr3.detections.append(d3)
        self.pub2d.publish(a2)
        self.pub3d.publish(a3)
        if self.yolo_topics:
            self.y2d.publish(y2)
            if y3.detections:   # upstream behaviour: only frames with a 3D object
                self.y3d.publish(y3)

    def _publish_image(self, raw_pub, jpg_pub, img, header, enc, ext, fmt, width=None):
        if raw_pub.get_num_connections():
            m = self.bridge.cv2_to_imgmsg(img, enc)
            m.header = header
            raw_pub.publish(m)
        if jpg_pub.get_num_connections():
            out = img if width is None else cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
            c = CompressedImage(header=header, format=fmt)
            c.data = cv2.imencode(ext, out, [cv2.IMWRITE_JPEG_QUALITY, 75] if ext == ".jpg" else [])[1].tobytes()
            jpg_pub.publish(c)

    def _color(self, o):
        if o["target"]:
            return TARGET_COLOR
        if o["track_id"] is None:
            return UNTRACKED_COLOR
        tid = o["track_id"]
        if tid not in self._colors:   # stable colour per track ID
            self._colors[tid] = tuple(int(v) for v in np.random.RandomState(tid * 7 + 3).randint(60, 255, 3))
        return self._colors[tid]

    def _publish_debug(self, img, objs, label, header, ms, outs):
        """The camera frame at full resolution: masks (tinted), boxes, '#ID class score distance', target in red."""
        vis = img.copy()
        colors = [self._color(o) for o in objs]
        if label is not None:   # one pass over the image for all masks: 50 % tint
            lut = np.zeros((256, 3), np.uint16)
            for o, c in zip(objs, colors):
                if o["mask_index"]:
                    lut[o["mask_index"]] = c
            sel = label > 0
            vis[sel] = ((vis[sel].astype(np.uint16) + lut[label[sel]]) >> 1).astype(np.uint8)
        fs, th = max(0.5, img.shape[1] / 1600.0), max(1, int(round(img.shape[1] / 800.0)))   # text/lines scale with size
        for o, c in sorted(zip(objs, colors), key=lambda oc: oc[0]["target"]):   # the target last, on top
            text = ("#%d " % o["track_id"] if o["track_id"] is not None else "") + "%s %.2f" % (o["name"], o["score"])
            if o["xyz"]:
                text += " %.1fm" % o["xyz"][2]
            x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, th * (2 if o["target"] else 1))
            cv2.putText(vis, text, (x1, max(int(20 * fs), y1 - 6)), 0, 0.7 * fs, c, th, cv2.LINE_AA)
        cv2.putText(vis, "%s | %s | %.0f ms" % (self.info["detector"], self.info["tracker"], ms), (10, int(30 * fs)),
                    0, 0.7 * fs, (255, 255, 255), th, cv2.LINE_AA)
        for raw, jpg, width in outs:
            self._publish_image(raw, jpg, vis, header, "bgr8", ".jpg", "jpeg", width)


if __name__ == "__main__":
    rospy.init_node("yolo_detector")
    DetectorNode()
    rospy.spin()
