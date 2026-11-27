#!/usr/bin/env python3
"""RF-DETR (COCO-pretrained, Roboflow) -> ONNX with a fixed square input at the model resolution (medium 576, large 704).
Needs the rfdetr package (get_models.sh rfdetr runs this in a throwaway venv; ONNX export works on the CPU).
    python export_rfdetr.py OUT_DIR medium large      -> OUT_DIR/rfdetr_medium.onnx, OUT_DIR/rfdetr_large.onnx
"""
import os
import shutil
import sys

import rfdetr

VARIANTS = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}
out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
for v in sys.argv[2:] or ["large"]:
    m = VARIANTS[v]()
    tmp = os.path.join(out_dir, "export_" + v)
    out = m.export(output_dir=tmp, verbose=False)
    dst = os.path.join(out_dir, "rfdetr_%s.onnx" % v)
    shutil.move(str(out), dst)
    shutil.rmtree(tmp, ignore_errors=True)
    print("OK %s %s (%.0f MB, resolution %d)" % (v, dst, os.path.getsize(dst) / 1e6, m.model_config.resolution), flush=True)
