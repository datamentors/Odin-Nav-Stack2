#!/usr/bin/env python3
# AGX ORIN, Odin-perception CUDA venv. Step 1 of the TensorRT test: YOLOv5l .pt -> ONNX (FP32, dynamic H/W),
# plus the class names as JSON. Nothing else is touched (new folder ~/odin_test/yolo_trt/).
#   ~/odin-perception/.venv/bin/python ~/odin_test/yolo_trt/export_onnx.py
import json, os, sys, time

GPU_DIR = os.path.expanduser("~/odin_test/yolo_gpu")
OUT_DIR = os.path.expanduser("~/odin_test/yolo_trt")
sys.path.insert(0, GPU_DIR)
import yolo5_stubs  # noqa: E402
yolo5_stubs.install()            # same empty modules as the GPU service, and NO pip installs from this process
sys.path.insert(0, os.path.join(GPU_DIR, "yolov5"))

import torch  # noqa: E402
import torch.onnx._internal.onnx_proto_utils as _opu  # noqa: E402
# torch 2.8 imports the "onnx" package only to insert onnxscript custom functions into the file; YOLOv5 has none,
# so that step is skipped here (no package to install). trtexec validates the resulting file anyway.
_opu._add_onnxscript_fn = lambda proto, custom_opsets: proto
from models.experimental import attempt_load  # noqa: E402
from models.yolo import Detect  # noqa: E402

WEIGHTS = os.environ.get("YOLO_WEIGHTS", os.path.expanduser(
    "~/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/yolov5l.pt"))
ONNX = os.path.join(OUT_DIR, "yolov5l.onnx")
t0 = time.time()
model = attempt_load(WEIGHTS, device=torch.device("cuda:0")).float().eval()
for m in model.modules():
    if isinstance(m, Detect):
        m.inplace = False
        m.export = True      # forward returns only the concatenated predictions (1, N, 85)
        m.dynamic = True     # grids rebuilt for any input size
names = model.module.names if hasattr(model, "module") else model.names
names = {str(k): str(v) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
json.dump(names, open(os.path.join(OUT_DIR, "names.json"), "w"))
x = torch.zeros(1, 3, 544, 640, device="cuda:0")   # live size: 1296x1600 letterboxed to 544x640
with torch.no_grad():
    y = model(x)[0]
print("pytorch output", tuple(y.shape))
torch.onnx.export(model, x, ONNX, opset_version=17, do_constant_folding=True, dynamo=False,
                  input_names=["images"], output_names=["output0"],
                  dynamic_axes={"images": {2: "height", 3: "width"}, "output0": {1: "anchors"}})
print("ONNX written: %s (%.0f MB) in %.1f s" % (ONNX, os.path.getsize(ONNX) / 1e6, time.time() - t0))
