#!/usr/bin/env python3
# ORIN, ~/odin_test/detbench/.venv. RF-DETR (COCO-pretrained) -> ONNX, fixed square input at the model resolution.
#   cd ~/odin_test/detbench/models && ../.venv/bin/python ../export_rfdetr.py medium large
import sys, shutil, glob, os, rfdetr
VARIANTS = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}
for v in sys.argv[1:] or ["medium", "large"]:
    m = VARIANTS[v]()
    out = m.export(output_dir="export_" + v, verbose=False)
    dst = "rfdetr_%s.onnx" % v
    shutil.move(str(out), dst)
    shutil.rmtree("export_" + v, ignore_errors=True)
    print("OK", v, dst, "%.0f MB" % (os.path.getsize(dst) / 1e6), "resolution", m.model_config.resolution, flush=True)
