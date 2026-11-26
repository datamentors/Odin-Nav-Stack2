#!/usr/bin/env python3
# AGX ORIN, Odin-perception CUDA venv. TensorRT version of ~/odin_test/yolo_gpu/yolo_gpu_service.py with the SAME
# HTTP API, so the ROS node (yolo_detector_gpu.py, process yolo in the container odin-agx) does not change at all:
#   GET  /names                         -> {"0": "person", ...}
#   POST /infer?conf=0.25&iou=0.45      body = letterboxed BGR uint8 image, header X-Shape: "h,w"
#                                       -> JSON [[x1, y1, x2, y2, conf, cls], ...] in letterbox pixels
#   GET  /health                        -> "ok <engine> <ms per image>"
# Engine: yolov5l_fp16.engine, built once by trtexec from yolov5l.onnx (export_onnx.py), FP16, input 1x3xHxW
# (256..640, optimised for the live 544x640). NMS = the same YOLOv5 non_max_suppression as the PyTorch service.
# The tensorrt Python bindings of JetPack (system Python 3.10) are added to THIS process only: the venv is unchanged.
#   test:  YOLO_GPU_PORT=8751 ~/odin-perception/.venv/bin/python ~/odin_test/yolo_trt/yolo_trt_service.py
import json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_DIR = os.path.expanduser("~/odin_test/yolo_gpu")
sys.path.insert(0, GPU_DIR)
import yolo5_stubs  # noqa: E402
yolo5_stubs.install()            # same empty modules as the PyTorch service + NO pip installs from this process
sys.path.insert(0, os.path.join(GPU_DIR, "yolov5"))
sys.path.append("/usr/lib/python3.10/dist-packages")   # JetPack tensorrt (appended last: venv packages still win)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tensorrt as trt  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402

ENGINE = os.environ.get("YOLO_TRT_ENGINE", os.path.join(HERE, "yolov5l_fp16.engine"))
PORT = int(os.environ.get("YOLO_GPU_PORT", "8751"))
DEV = torch.device("cuda:0")
LOCK = threading.Lock()
STATS = {"n": 0, "ms": 0.0}
TORCH_DTYPE = {trt.float32: torch.float32, trt.float16: torch.float16}

t0 = time.time()
NAMES = json.load(open(os.path.join(HERE, "names.json")))
with open(ENGINE, "rb") as f:
    ENG = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
if ENG is None:
    sys.exit("cannot load the TensorRT engine %s (rebuild it with trtexec after a JetPack/TensorRT update)" % ENGINE)
CTX = ENG.create_execution_context()
IN, OUT = "images", "output0"
IN_DT, OUT_DT = TORCH_DTYPE[ENG.get_tensor_dtype(IN)], TORCH_DTYPE[ENG.get_tensor_dtype(OUT)]
STREAM = torch.cuda.Stream(DEV)   # own CUDA stream (TensorRT warns about extra syncs on the default stream)


def run_engine(x):
    """x: (1, 3, h, w) float on the GPU -> raw YOLOv5 predictions (1, N, 85), like MODEL(x)[0]."""
    x = x.to(IN_DT).contiguous()
    CTX.set_input_shape(IN, tuple(x.shape))
    y = torch.empty(tuple(CTX.get_tensor_shape(OUT)), dtype=OUT_DT, device=DEV)
    CTX.set_tensor_address(IN, x.data_ptr())
    CTX.set_tensor_address(OUT, y.data_ptr())
    STREAM.wait_stream(torch.cuda.current_stream(DEV))   # x and y are ready before TensorRT starts
    if not CTX.execute_async_v3(STREAM.cuda_stream):
        raise RuntimeError("TensorRT execution failed")
    STREAM.synchronize()                                 # y is complete before anyone reads it
    return y.float()


with torch.no_grad():  # warm-up at the live size
    run_engine(torch.zeros(1, 3, 544, 640, device=DEV))
print("YOLO TensorRT service: %s loaded in %.1f s, %d classes, listening on 127.0.0.1:%d"
      % (os.path.basename(ENGINE), time.time() - t0, len(NAMES), PORT), flush=True)


def infer(img_bgr, conf, iou):
    t = time.time()
    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1].transpose(2, 0, 1))).to(DEV)  # BGR->RGB, CHW
    x = x.float().unsqueeze(0) / 255.0
    with LOCK, torch.no_grad():
        pred = non_max_suppression(run_engine(x), conf_thres=conf, iou_thres=iou)[0]
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
            return self._send(200, "ok %s (TensorRT) %.0f ms/img (%d images)"
                              % (os.path.basename(ENGINE), STATS["ms"], STATS["n"]), "text/plain")
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
