#!/usr/bin/env python3
"""Exports a BoxMOT Re-ID model (default OSNet x0.25 MSMT17) to ONNX with a dynamic batch, then builds an FP16
TensorRT engine with trtexec, for the tracker botsort_reid. Input 'images' (N x 3 x 256 x 128), output 'output' (the
names BoxMOT's TensorRT backend looks for). Needs boxmot (odin-gpu image built with WITH_TRACKERS=1); BoxMOT downloads
the weights on first use. Normally run by get_models.sh reid.
    python export_reid.py ~/odin_test/models/osnet_x0_25_msmt17.pt   # -> .onnx + _fp16.engine next to it
"""
import os, subprocess, sys

import torch

w = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/odin_test/models/osnet_x0_25_msmt17.pt")
base = os.path.splitext(w)[0]
from boxmot.reid.core.runtime import ReID  # noqa: E402

rt = ReID(w, device="cuda", half=False)
model = next(m for m in (getattr(rt.model, "model", None), rt.model) if isinstance(m, torch.nn.Module)).eval()
h, wd = getattr(rt.model, "input_shape", (256, 128))[-2:]
x = torch.zeros(1, 3, h, wd, device="cuda")
with torch.no_grad():
    print("features", tuple(model(x).shape))
    torch.onnx.export(model, x, base + ".onnx", input_names=["images"], output_names=["output"], opset_version=17,
                      dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}})
print("onnx ->", base + ".onnx")
cmd = ["/usr/src/tensorrt/bin/trtexec", "--onnx=" + base + ".onnx", "--saveEngine=" + base + "_fp16.engine", "--fp16",
       "--minShapes=images:1x3x%dx%d" % (h, wd), "--optShapes=images:4x3x%dx%d" % (h, wd), "--maxShapes=images:32x3x%dx%d" % (h, wd)]
log = open(base + "_build.log", "w")
r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
print("engine ->", base + "_fp16.engine" if r.returncode == 0 else "FAILED, see " + base + "_build.log")
