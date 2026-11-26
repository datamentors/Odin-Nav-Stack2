#!/usr/bin/env python3
# AGX ORIN, Odin-perception CUDA venv. OUR helper (not upstream): YOLOv5l on the AGX GPU for the ONS2 yolo_detector.
# ROS 1 (python 3.8, containers) and the CUDA torch (python 3.10, this venv) cannot live in one process, so the
# ROS node (yolo_detector_gpu.py, process yolo in the container odin-agx) letterboxes the image and asks this service for the
# raw YOLOv5 output + NMS (~60 ms on the GPU instead of ~3 s on a CPU). Same code path as their detector.
#   GET  /names                         -> {"0": "person", ...}
#   POST /infer?conf=0.25&iou=0.45      body = letterboxed BGR uint8 image, header X-Shape: "h,w"
#                                       -> JSON [[x1, y1, x2, y2, conf, cls], ...] in letterbox pixels
#   GET  /health                        -> "ok <model> <ms per image>"
# Started by ~/odin_test/start_yolo_gpu_agx.sh. Log: ~/odin_test/logs/yolo_gpu_service.log
import json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yolo5_stubs  # noqa: E402
yolo5_stubs.install()            # empty plotting/training modules + NO pip installs from this process
sys.path.insert(0, os.path.join(HERE, "yolov5"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from models.experimental import attempt_load  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402

WEIGHTS = os.environ.get("YOLO_WEIGHTS", os.path.expanduser(
    "~/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/yolov5l.pt"))
PORT = int(os.environ.get("YOLO_GPU_PORT", "8750"))
DEV = torch.device("cuda:0")
LOCK = threading.Lock()
STATS = {"n": 0, "ms": 0.0}

t0 = time.time()
MODEL = attempt_load(WEIGHTS, device=DEV)
MODEL.half().eval()
NAMES = MODEL.module.names if hasattr(MODEL, "module") else MODEL.names
NAMES = {str(k): str(v) for k, v in (NAMES.items() if isinstance(NAMES, dict) else enumerate(NAMES))}
with torch.no_grad():  # warm-up (first CUDA call is slow)
    MODEL(torch.zeros(1, 3, 512, 640, device=DEV).half())
print("YOLO GPU service: %s loaded in %.1f s, %d classes, listening on 127.0.0.1:%d"
      % (os.path.basename(WEIGHTS), time.time() - t0, len(NAMES), PORT), flush=True)


def infer(img_bgr, conf, iou):
    t = time.time()
    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1].transpose(2, 0, 1))).to(DEV)  # BGR->RGB, CHW
    x = x.half().unsqueeze(0) / 255.0
    with LOCK, torch.no_grad():
        pred = non_max_suppression(MODEL(x)[0], conf_thres=conf, iou_thres=iou)[0]
        out = pred.float().cpu().tolist()
    ms = (time.time() - t) * 1000
    STATS["n"] += 1
    STATS["ms"] = ms if STATS["n"] == 1 else 0.9 * STATS["ms"] + 0.1 * ms
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p == "/names":
            return self._send(200, json.dumps(NAMES))
        if p == "/health":
            return self._send(200, "ok %s %.0f ms/img (%d images)" % (os.path.basename(WEIGHTS), STATS["ms"], STATS["n"]),
                              "text/plain")
        self._send(404, "{}")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/infer":
            return self._send(404, "{}")
        q = urllib.parse.parse_qs(u.query)
        try:
            h, w = [int(v) for v in self.headers.get("X-Shape", "").split(",")]
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            out = infer(img, float(q.get("conf", ["0.25"])[0]), float(q.get("iou", ["0.45"])[0]))
            self._send(200, json.dumps(out))
        except Exception as e:  # never kill the service on one bad request
            self._send(500, json.dumps({"error": str(e)}))


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
