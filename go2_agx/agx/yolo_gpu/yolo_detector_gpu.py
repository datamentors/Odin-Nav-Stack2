#!/usr/bin/env python3
# OUR AGX GPU VERSION (16 Nov) of the ONS2 yolo_ros/scripts/yolo_detector.py (with our earlier fixes: BGR->RGB,
# conf_thres param + /yolo/set_conf). Generated from the Go2 copy; ONLY the model part changes: the YOLOv5 forward
# pass + NMS run on the AGX GPU in yolo_gpu_service.py (127.0.0.1:8750, YOLOv5l, ~60 ms instead of ~3 s on CPU).
# Everything else (letterbox, depth, fisheye 3D, topics, debug image) is their code, unchanged.
# Extra: publishes /yolo_debug_image/compressed itself (800 px, q70) and the raw debug image only if someone
# subscribes (it is 6 MB per frame: it would cross the cable to the Go2).
# DEPTH FIX (16 Nov): their code read the lidar depth in a 5x5 patch at the box CENTRE (a chair is hollow there:
# it measured the wall behind, 7.9 m instead of 1.5 m), unprojected the UNDISTORTED image with the FISHEYE model and
# used the calibration of another Odin1 (O1-N090100030). Now: 20th percentile of the lidar depth inside the box
# (nearest surfaces of the object), pinhole model with THIS Odin1's calibration (/calib.yaml = its calib.yaml,
# the same intrinsics pcd2depth uses to project the lidar into the image).
import torch  # noqa: E402  (first: torch.libs libgomp needs static TLS before cv2 loads)
import json as _json, time as _time, urllib.request as _urlreq, threading as _threading
# Copyright 2026 Manifold Tech Ltd.(www.manifoldtech.com.co)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from vision_msgs.msg import Detection3DArray, Detection3D
from geometry_msgs.msg import Pose2D, Twist, PointStamped
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import torch
import sys
import os
import message_filters
import numpy as np
import tf2_ros

# add yolov5 path
YOLOV5_PATH = "/opt/odin/yolov5"  # OUR: yolov5 code inside the odin-yolo image (only for scale_boxes/Annotator)
sys.path.append(YOLOV5_PATH)

from utils.general import scale_boxes
from utils.plots import Annotator


class YoloDetector:
    def __init__(self):
        self.bridge = CvBridge()
        # OUR: model on the AGX GPU (yolo_gpu_service.py); wait for it and take the class names from it
        self.gpu_url = rospy.get_param("~gpu_url", "http://127.0.0.1:8750")
        self.stride = 32
        while not rospy.is_shutdown():
            try:
                self.names = _json.loads(_urlreq.urlopen(self.gpu_url + "/names", timeout=3).read())
                self.names = {int(k): v for k, v in self.names.items()}
                break
            except Exception as e:
                rospy.logwarn_throttle(10, "[YOLO-GPU] waiting for the GPU service %s (%s)" % (self.gpu_url, e))
                _time.sleep(2)
        self._gpu_ms = 0.0
        # OUR (16 Nov): classes that are never inside a room/hall/auditorium. They were only false positives here
        # (a chair seen from below or the robot dog on the floor -> "airplane", ceiling lights -> "car"...).
        # Change with the param ~ignore_classes (comma separated; empty = keep all).
        default_ignore = ("airplane,bus,train,truck,boat,car,motorcycle,traffic light,fire hydrant,stop sign,"
                          "parking meter,bird,horse,sheep,cow,elephant,bear,zebra,giraffe,surfboard,skis,snowboard,"
                          "kite,skateboard")
        self.ignore = {c.strip().lower() for c in str(rospy.get_param("~ignore_classes", default_ignore)).split(",")
                       if c.strip()}
        unknown = self.ignore - {v.lower() for v in self.names.values()}
        if unknown:
            rospy.logwarn("[YOLO] ignore_classes not in the model: %s" % ", ".join(sorted(unknown)))
        rospy.loginfo("[YOLO] ignoring %d classes: %s" % (len(self.ignore), ", ".join(sorted(self.ignore))))

        self.rgb_topic = rospy.get_param("~rgb_topic", "/odin1/image/compressed")
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/odin1/depth_img_competetion"
        )
        self.auto_nav_mode = rospy.get_param("~auto_nav_mode", False)
        self.camera_frame = rospy.get_param("~camera_frame", "camera_link")
        self.target_frame = rospy.get_param("~target_frame", "map")
        self.use_latest_tf = rospy.get_param("~use_latest_tf", True)  # use latest TF
        self.depth_mode = rospy.get_param("~depth_mode", "z")
        # OUR change (12 Nov): confidence threshold was hard-coded to 0.25; now a param + live topic /yolo/set_conf
        from std_msgs.msg import Float32 as _F32, String as _Str
        self.conf_thres = float(rospy.get_param("~conf_thres", 0.25))
        self._conf_echo = rospy.Publisher("/yolo/console", _Str, queue_size=1, latch=True)
        def _set_conf(m):
            v = min(max(float(m.data), 0.01), 0.99)
            self.conf_thres = v
            rospy.loginfo("[YOLO] confidence threshold -> %.2f", v)
            self._conf_echo.publish(_Str(data="YOLO confidence threshold set to %.2f (%d %%)" % (v, round(v * 100))))
        rospy.Subscriber("/yolo/set_conf", _F32, _set_conf)
        rospy.loginfo(f"[YOLO] RGB topic: {self.rgb_topic}")
        rospy.loginfo(f"[YOLO] Depth topic: {self.depth_topic}")
        rospy.loginfo(f"[YOLO] Target frame: {self.target_frame}")
        rospy.loginfo(f"[YOLO] Use latest TF: {self.use_latest_tf}")
        rospy.loginfo(f"[YOLO] Depth mode: {self.depth_mode} (z|range)")

        # TF2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # FishPoly camera calibration parameters
        self.cam_params = {
            "image_width": 1600,
            "image_height": 1296,
            "k2": 2.2386681497553868e-02,
            "k3": -1.1005721600958770e-01,
            "k4": 1.9889594376167404e-01,
            "k5": -2.2469360826066142e-01,
            "k6": 1.2659501421970656e-01,
            "k7": -2.9594663777173985e-02,
            "p1": 0.0,
            "p2": 0.0,
            "A11": 7.3118660487066472e02,
            "A12": -3.1721597080498198e-01,
            "A22": 7.3103919402862072e02,
            "u0": 8.1792273080715278e02,
            "v0": 6.7415986395892662e02,
            "maxIncidentAngle": 120,
        }

        # OUR: real intrinsics of THIS Odin1 (pinhole model of /odin1/image/undistorted and of the lidar depth image)
        self.K = {"fx": self.cam_params["A11"], "fy": self.cam_params["A22"], "cx": self.cam_params["u0"],
                  "cy": self.cam_params["v0"], "skew": self.cam_params["A12"]}
        try:
            import re as _re
            txt = open(rospy.get_param("~calib_file", "/calib.yaml")).read()
            def _g(k):
                return float(_re.search(r"^\s*%s:\s*([-+0-9.eE]+)" % k, txt, _re.M).group(1))
            self.K = {"fx": _g("A11"), "fy": _g("A22"), "cx": _g("u0"), "cy": _g("v0"), "skew": _g("A12")}
            rospy.loginfo("[YOLO] intrinsics from calib.yaml: fx %.1f fy %.1f cx %.1f cy %.1f" %
                          (self.K["fx"], self.K["fy"], self.K["cx"], self.K["cy"]))
        except Exception as e:
            rospy.logwarn("[YOLO] calib.yaml not read (%s): using the built-in (other Odin1) intrinsics" % e)
        self.depth_percentile = float(rospy.get_param("~depth_percentile", 20.0))

        # depth image cache (None until first depth message received)
        self.depth_image = None

        # camera extrinsics Tcl (camera to lidar/body) - O1-N090100030
        self.Tcl = np.array(
            [
                [-0.01476, -0.99983, -0.01096, 0.04354],
                [0.00604, 0.01087, -0.99992, -0.01748],
                [0.99987, -0.01482, 0.00588, -0.02099],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # select message type based on topic suffix (support /compressed)
        rgb_msg_type = (
            CompressedImage if self.rgb_topic.endswith("/compressed") else Image
        )
        # OUR (17 Nov): newest-frame-only hand-off between the rospy receive thread and the YOLO worker thread
        self._img_lock = _threading.Lock()
        self._img_event = _threading.Event()
        self._img_latest = None
        self.image_sub = rospy.Subscriber(
            self.rgb_topic,
            rgb_msg_type,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.depth_callback, queue_size=1, buff_size=2**24
        )

        self.detection_pub = rospy.Publisher(
            "/yolo_detections", Detection2DArray, queue_size=1
        )
        self.detection3d_pub = rospy.Publisher(
            "/yolo_detections_3d", Detection3DArray, queue_size=1
        )
        self.debug_pub = rospy.Publisher("/yolo_debug_image", Image, queue_size=1)
        self.debug_c_pub = rospy.Publisher("/yolo_debug_image/compressed", CompressedImage, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.class_names_pub = rospy.Publisher(
            "/yolo_class_names", String, queue_size=1, latch=True
        )

        # publish class name mapping (latched topic, subscribers can get the latest value)
        import json

        # ensure class names are strings
        if isinstance(self.names, dict):
            class_mapping = {str(k): str(v) for k, v in self.names.items()}
        else:
            class_mapping = {str(k): str(v) for k, v in enumerate(self.names)}
        self.class_names_pub.publish(json.dumps(class_mapping))

        rospy.loginfo(f"YOLO detector (AGX GPU, {self.gpu_url}) initialized with {len(self.names)} classes.")
        rospy.loginfo(f"Classes: {list(self.names)[:10]}...")
        _threading.Thread(target=self._yolo_worker, name="yolo_worker", daemon=True).start()

    def letterbox(
        self,
        img,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        """adjust image size and padding to meet stride requirements"""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only shrink, no zoom
            r = min(r, 1.0)

        # calculate padding
        ratio = r, r  # width, height ratio
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratio

        dw /= 2  # divide to both sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return img, ratio, (dw, dh)

    def image_callback(self, msg):
        # OUR (17 Nov): LAG FIX. Before, all the work (~0.27 s) ran here, inside rospy's receive thread: meanwhile
        # the 10 Hz camera frames piled up in the socket/publisher queue, so YOLO always processed a frame ~1.2 s
        # old (laggy debug image in Foxglove, stale positions for go/follow, depth newer than the image).
        # Now this only keeps the NEWEST frame; _yolo_worker processes it and older ones are dropped.
        with self._img_lock:
            self._img_latest = msg
        self._img_event.set()

    def _yolo_worker(self):
        while not rospy.is_shutdown():
            if not self._img_event.wait(0.5):
                continue
            with self._img_lock:
                msg, self._img_latest = self._img_latest, None
                self._img_event.clear()
            if msg is None:
                continue
            try:
                self._process_image(msg)
            except Exception as e:
                rospy.logerr_throttle(5, "[YOLO] processing error: %s" % e)

    def _process_image(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                cv_image = self.bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding="bgr8"
                )
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        # save original image size for later coordinate mapping
        img_orig = cv_image.copy()
        h0, w0 = cv_image.shape[:2]

        # preprocess image: resize and padding
        img_resized, ratio, pad = self.letterbox(
            cv_image, new_shape=640, stride=self.stride, auto=True
        )

        # OUR: forward pass + NMS on the AGX GPU (same letterboxed BGR image their code builds; BGR->RGB done there)
        lb_shape = img_resized.shape[:2]
        try:
            _t = _time.time()
            req = _urlreq.Request(
                "%s/infer?conf=%.3f&iou=0.45" % (self.gpu_url, self.conf_thres),
                data=np.ascontiguousarray(img_resized).tobytes(),
                headers={"X-Shape": "%d,%d" % lb_shape, "Content-Type": "application/octet-stream"}, method="POST")
            rows = _json.loads(_urlreq.urlopen(req, timeout=5).read())
            self._gpu_ms = 0.9 * self._gpu_ms + 0.1 * (_time.time() - _t) * 1000
        except Exception as e:
            rospy.logwarn_throttle(5, "[YOLO-GPU] inference request failed: %s" % e)
            return
        pred = [torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)]

        det_msg = Detection2DArray()
        det_msg.header = msg.header

        det3d_msg = Detection3DArray()
        det3d_msg.header = msg.header

        annotator = Annotator(img_orig.copy(), line_width=2, example=str(self.names))

        for i, det in enumerate(pred):
            if len(det):
                # scale boxes to original image size
                det[:, :4] = scale_boxes(
                    lb_shape, det[:, :4], img_orig.shape
                ).round()

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = self.names[int(cls)]
                    if label.lower() in self.ignore:   # OUR: never indoors -> not published, not drawn
                        continue
                    confidence = float(conf)

                    # construct Detection2D
                    detection = Detection2D()
                    detection.bbox.center.x = (x1 + x2) / 2.0
                    detection.bbox.center.y = (y1 + y2) / 2.0
                    detection.bbox.size_x = x2 - x1
                    detection.bbox.size_y = y2 - y1

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.id = int(cls)
                    hypothesis.score = confidence
                    detection.results.append(hypothesis)

                    det_msg.detections.append(detection)

                    # draw box for debugging
                    annotator.box_label(xyxy, f"{label} {confidence:.2f}")

                    # calculate 3D position (using fisheye model)
                    if self.depth_image is not None:
                        # ensure coordinates are within original image range
                        u = int(np.clip((x1 + x2) / 2.0, 0, w0 - 1))
                        v = int(np.clip((y1 + y2) / 2.0, 0, h0 - 1))
                        depth_m = self.get_object_depth(x1, y1, x2, y2, u, v)  # OUR: lidar points ON the object
                        if depth_m is not None and np.isfinite(depth_m) and depth_m > 0:
                            # use FishPoly model to unproject pixel coordinates and depth to 3D camera coordinates
                            X_cam, Y_cam, Z_cam = self.unproject_pinhole(u, v, depth_m)  # OUR: undistorted image
                            if (
                                X_cam is not None
                                and Y_cam is not None
                                and Z_cam is not None
                            ):
                                # publish 3D results in camera frame
                                d3 = Detection3D()
                                d3.results.append(hypothesis)
                                d3.bbox.center.position.x = X_cam
                                d3.bbox.center.position.y = Y_cam
                                d3.bbox.center.position.z = Z_cam
                                d3.bbox.size.x = max(0.01, (x2 - x1) * depth_m / 800.0)
                                d3.bbox.size.y = max(0.01, (y2 - y1) * depth_m / 800.0)
                                d3.bbox.size.z = 0.01
                                det3d_msg.detections.append(d3)

        # publish detection results
        self.detection_pub.publish(det_msg)

        if len(det3d_msg.detections) > 0:
            # set 3D detection results frame id to camera frame
            self.detection3d_pub.publish(det3d_msg)

        # publish debug image (OUR: compressed 800 px always; the 6 MB raw image only if someone subscribes)
        debug_img = annotator.result()
        if self.debug_pub.get_num_connections() > 0:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        if self.debug_c_pub.get_num_connections() > 0:
            h_, w_ = debug_img.shape[:2]
            small = cv2.resize(debug_img, (800, int(h_ * 800 / w_)))
            cmsg = CompressedImage(header=msg.header, format="jpeg")
            cmsg.data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            self.debug_c_pub.publish(cmsg)

        if self.auto_nav_mode:
            self.simple_avoidance_control(det3d_msg)

    def depth_callback(self, msg):
        try:
            if msg.encoding == "32FC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="32FC1"
                )
            elif msg.encoding in ["16UC1", "mono16"]:
                di = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                self.depth_image = di.astype(np.float32) / 1000.0
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="32FC1"
                )
        except Exception as e:
            return

    def transform_to_target_frame(self, x_cam, y_cam, z_cam, timestamp):
        """
        transform point from camera frame to target frame
        return: (x_map, y_map, z_map) or (None, None, None) if transform fails
        """
        try:
            # if target frame is the same as camera frame, return directly
            if self.target_frame == self.camera_frame:
                return x_cam, y_cam, z_cam

            # try to get TF transform
            try:
                # always use rospy.Time(0) to get latest TF (to avoid timestamp mismatch)
                lookup_time = rospy.Time(0) if self.use_latest_tf else timestamp
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.camera_frame,
                    lookup_time,
                    rospy.Duration(0.5),
                )

                # manually execute coordinate transformation (do not use tf2_geometry_msgs)
                # extract translation
                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                tz = transform.transform.translation.z

                # extract rotation quaternion
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w

                # quaternion to rotation matrix
                R = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)

                # apply rotation and translation
                point_cam = np.array([x_cam, y_cam, z_cam])
                point_map = R @ point_cam + np.array([tx, ty, tz])

                return point_map[0], point_map[1], point_map[2]

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                # TF transform failed, return camera frame coordinates and warn
                if not hasattr(self, "_tf_warning_printed"):
                    rospy.logwarn(f"[TF] transform failed: {str(e)[:100]}")
                    rospy.logwarn(
                        f"[TF] required transform: {self.camera_frame} -> {self.target_frame}"
                    )
                    rospy.logwarn(
                        f"[TF] returning camera frame coordinates. Please check if TF tree is complete."
                    )
                    rospy.logwarn(
                        f"[TF] use command to check: rosrun tf tf_echo {self.target_frame} {self.camera_frame}"
                    )
                    self._tf_warning_printed = True
                # return camera frame coordinates as backup
                return x_cam, y_cam, z_cam

        except Exception as e:
            rospy.logerr(f"[Transform] coordinate transformation exception: {e}")
            return x_cam, y_cam, z_cam  # return camera frame coordinates as backup

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """convert quaternion to rotation matrix"""
        R = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ]
        )
        return R

    def unproject_fisheye(self, u, v, depth):
        """use FishPoly model to unproject pixel coordinates and depth to 3D camera coordinates"""
        try:
            # normalize pixel coordinates
            mx = (u - self.cam_params["u0"]) / self.cam_params["A11"]
            my = (v - self.cam_params["v0"]) / self.cam_params["A22"]

            # calculate radial distance
            r = np.sqrt(mx**2 + my**2)

            # FishPoly model: theta = r * (1 + k2*r^2 + k3*r^4 + k4*r^6 + k5*r^8 + k6*r^10 + k7*r^12)
            r2 = r * r
            r4 = r2 * r2
            r6 = r4 * r2
            r8 = r6 * r2
            r10 = r8 * r2
            r12 = r10 * r2

            theta = r * (
                1.0
                + self.cam_params["k2"] * r2
                + self.cam_params["k3"] * r4
                + self.cam_params["k4"] * r6
                + self.cam_params["k5"] * r8
                + self.cam_params["k6"] * r10
                + self.cam_params["k7"] * r12
            )

            # check incident angle limit
            max_angle_rad = np.deg2rad(self.cam_params["maxIncidentAngle"])
            if theta > max_angle_rad:
                return None, None, None

            # calculate 3D direction vector
            if r < 1e-6:
                # avoid division by zero
                X = 0.0
                Y = 0.0
                Z = depth
            else:
                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)
                # determine ray distance based on depth mode
                # depth_mode == 'z': depth is the Z direction of the optical axis, the distance along the ray is depth / cos(theta)
                # depth_mode == 'range': depth is the distance along the ray
                if getattr(self, "depth_mode", "z") == "z":
                    ray_dist = depth / max(1e-6, cos_theta)
                else:
                    ray_dist = depth

                # use ray distance to decompose to camera coordinate axis
                X = (mx / r) * sin_theta * ray_dist
                Y = (my / r) * sin_theta * ray_dist
                Z = cos_theta * ray_dist

            return float(X), float(Y), float(Z)
        except Exception as e:
            rospy.logwarn(f"Fisheye unprojection failed: {e}")
            return None, None, None

    def get_object_depth(self, x1, y1, x2, y2, u, v):
        """OUR: distance of the OBJECT, not of what is seen through it: the lidar depth image (pcd2depth) inside the
        box shrunk 10 % per side, 20th percentile of the valid points = its nearest surfaces (chair legs/back,
        person). The box centre of a chair, table or stool shows the background (7.9 m instead of 1.5 m)."""
        if self.depth_image is None:
            return None
        h, w = self.depth_image.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        xa, xb = int(np.clip(x1 + 0.1 * bw, 0, w - 1)), int(np.clip(x2 - 0.1 * bw, 1, w))
        ya, yb = int(np.clip(y1 + 0.1 * bh, 0, h - 1)), int(np.clip(y2 - 0.1 * bh, 1, h))
        patch = self.depth_image[ya:yb, xa:xb]
        vals = patch[np.isfinite(patch) & (patch > 0.15) & (patch < 40.0)]
        if vals.size >= 20:
            return float(np.percentile(vals, self.depth_percentile))
        return self.get_depth_m(u, v)  # tiny/far box: their centre patch

    def unproject_pinhole(self, u, v, depth):
        """OUR: /odin1/image/undistorted and the lidar depth image are pinhole images with this Odin1's intrinsics."""
        K = self.K
        my = (v - K["cy"]) / K["fy"]
        mx = (u - K["cx"] - K["skew"] * my) / K["fx"]
        z = depth if getattr(self, "depth_mode", "z") == "z" else depth / np.sqrt(1.0 + mx * mx + my * my)
        return float(mx * z), float(my * z), float(z)

    def get_depth_m(self, u, v):
        if self.depth_image is None:
            return None
        h, w = self.depth_image.shape[:2]
        u = np.clip(u, 0, w - 1)
        v = np.clip(v, 0, h - 1)
        patch = self.depth_image[
            max(0, v - 2) : min(h, v + 3), max(0, u - 2) : min(w, u + 3)
        ]
        val = np.median(patch[np.isfinite(patch)]) if patch.size > 0 else None
        return float(val) if val is not None else None

    def simple_avoidance_control(self, det3d_msg):
        if det3d_msg is None or len(det3d_msg.detections) == 0:
            twist = Twist()
            twist.linear.x = 0.2
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
        nearest = None
        min_z = 1e9
        for d in det3d_msg.detections:
            z = d.bbox.center.position.z
            if z > 0 and z < min_z:
                min_z = z
                nearest = d
        twist = Twist()
        if nearest is None:
            twist.linear.x = 0.2
            twist.angular.z = 0.0
        else:
            x = nearest.bbox.center.position.x
            z = nearest.bbox.center.position.z
            err = x / max(0.1, z)
            twist.angular.z = float(-0.8 * err)
            if z < 1.0:
                twist.linear.x = 0.0
            else:
                twist.linear.x = 0.15
        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    rospy.init_node("yolo_detector", anonymous=True)
    yolo = YoloDetector()
    rospy.spin()
