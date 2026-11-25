#!/usr/bin/env python3
"""Pre-labels for human correction (CVAT / Label Studio): runs two detectors, keeps the classes asked for, merges
their boxes (union + class-wise NMS, so a box found by either model is proposed once) and writes COCO 1.0 JSON.

The output is a STARTING POINT for labelling, never ground truth: every frame must be checked by a person
(add missed objects, delete wrong boxes, tighten boxes) before it is used to score the models.

  V=~/odin_test/detbench/.venv/bin/python
  $V prelabel.py --images data/frames_label --out data/prelabels_person.json \
     --weights ~/odin_test/yolo_trt/yolov5l_fp16.engine models/rfdetr_medium_fp16.engine
"""
import argparse, glob, json, os, sys, time
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", nargs="+", required=True, help="one or more .engine/.onnx/.pt files")
    p.add_argument("--classes", default="person")
    p.add_argument("--conf", type=float, default=0.3, help="per-model confidence for a proposal")
    p.add_argument("--merge-iou", type=float, default=0.5)
    p.add_argument("--yolov5-dir", default=os.path.join(bench.HOME, "odin_test/yolo_gpu/yolov5"))
    p.add_argument("--save-vis", help="folder for frames with the proposed boxes drawn")
    a = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    dets = []
    for w in a.weights:
        o = SimpleNamespace(weights=w, yolov5_dir=a.yolov5_dir, imgsz=640, iou=0.45, device="cuda", fp32=False, threads=0)
        fam = "rfdetr" if "rfdetr" in os.path.basename(w).lower() else "yolov5"
        dets.append((os.path.splitext(os.path.basename(w))[0], (bench.RFDETR if fam == "rfdetr" else bench.YOLOv5)(o)))

    images, anns, per_model, t0 = [], [], {n: 0 for n, _ in dets}, time.time()
    for k, f in enumerate(files):
        img = cv2.imread(f)
        h, w = img.shape[:2]
        images.append({"id": k + 1, "file_name": os.path.basename(f), "width": w, "height": h})
        B, S, N = [], [], []
        for name, d in dets:
            b, s, n = d.detect(img, a.conf)
            keep = [i for i, c in enumerate(n) if c in classes]
            per_model[name] += len(keep)
            B.append(b[keep].reshape(-1, 4)); S.append(s[keep]); N += [n[i] for i in keep]
        if N:
            b, s, n = bench.nms_by_class(np.concatenate(B), np.concatenate(S), N, a.merge_iou)
        else:
            b, s, n = np.zeros((0, 4)), np.zeros(0), []
        for bb, ss, nn in zip(b, s, n):
            x1, y1, x2, y2 = [float(v) for v in bb]
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
            anns.append({"id": len(anns) + 1, "image_id": k + 1, "category_id": classes.index(nn) + 1,
                         "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                         "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0, "segmentation": [],
                         "attributes": {"occluded": False}})
        if a.save_vis:
            os.makedirs(a.save_vis, exist_ok=True)
            v = img.copy()
            for bb, ss in zip(b, s):
                cv2.rectangle(v, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 200, 0), 3)
                cv2.putText(v, "%.2f" % ss, (int(bb[0]), int(bb[1]) - 6), 0, 0.8, (0, 200, 0), 2)
            cv2.imwrite(os.path.join(a.save_vis, os.path.basename(f)), v)

    coco = {"info": {"description": "PRE-LABELS for human correction (models: %s, conf %.2f). Not ground truth."
                     % (", ".join(n for n, _ in dets), a.conf), "date_created": time.strftime("%Y-%m-%d %H:%M")},
            "licenses": [], "images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": c, "supercategory": ""} for i, c in enumerate(classes)]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(coco, open(a.out, "w"))
    with_obj = len({x["image_id"] for x in anns})
    print("%d frames, %d merged boxes (%s), %d frames with at least one box, %.0f s -> %s"
          % (len(files), len(anns), ", ".join("%s %d" % kv for kv in per_model.items()), with_obj, time.time() - t0, a.out))


if __name__ == "__main__":
    main()
