#!/usr/bin/env python3
"""Error analysis of one detector against the ground truth: which people it found (TP), which it missed (FN) and
which boxes are wrong (FP), frame by frame, with pictures.

Outputs in --out:
  errors.csv            one row per box: frame, kind (TP/FP/FN), score, IoU with the matched box, box, height in px
  vis/<frame>.jpg       every frame with at least one error (or all frames with --all):
                          green  = TP (model box, matched a labelled person)
                          red    = FP (model box, no labelled person)   + score
                          yellow = FN (labelled person the model missed), thick
  predictions_coco.json the model's boxes at --conf as COCO (e.g. for labelstudio.py to-tasks)
Matching: greedy by score, IoU >= --iou, one model box per labelled person (same rule as bench.py).

  .venv/bin/python errors.py --images data/frames_label --labels data/labels_person.json \
      --weights models/rfdetr_large_fp16.engine --conf 0.5 --out data/errors_rfdetr_large
"""
import argparse, csv, glob, json, os, sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--cls", default="person")
    p.add_argument("--out", required=True)
    p.add_argument("--all", action="store_true", help="draw every frame, not only frames with errors")
    p.add_argument("--yolov5-dir", default=os.path.join(bench.HOME, "odin_test/yolo_gpu/yolov5"))
    a = p.parse_args()

    o = SimpleNamespace(weights=a.weights, yolov5_dir=a.yolov5_dir, imgsz=640, iou=0.45, device="cuda", fp32=False, threads=0)
    fam = "rfdetr" if "rfdetr" in os.path.basename(a.weights).lower() else "yolov5"
    det = (bench.RFDETR if fam == "rfdetr" else bench.YOLOv5)(o)
    name = os.path.splitext(os.path.basename(a.weights))[0]

    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    gts, _ = bench.load_labels(a.labels, files)
    os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
    rows, coco_imgs, coco_anns = [], [], []
    tot = {"TP": 0, "FP": 0, "FN": 0}
    for k, (f, g) in enumerate(zip(files, gts)):
        img = cv2.imread(f)
        h, w = img.shape[:2]
        b, s, n = det.detect(img, a.conf)
        keep = [i for i, c in enumerate(n) if c == a.cls]
        pb, ps = b[keep].reshape(-1, 4), s[keep]
        gb = np.array([bb for c, bb in (g or []) if c == a.cls], np.float32).reshape(-1, 4)
        ious = bench.iou_matrix(pb, gb)
        order, used, kind = np.argsort(-ps), np.zeros(len(gb), bool), {}
        match = {}
        for i in order:
            j = int(np.argmax(ious[i])) if len(gb) else -1
            if j >= 0 and ious[i, j] >= a.iou and not used[j]:
                used[j] = True; kind[i] = "TP"; match[i] = j
            else:
                kind[i] = "FP"
        base = os.path.basename(f)
        for i in range(len(pb)):
            iou = ious[i, match[i]] if i in match else (ious[i].max() if len(gb) else 0.0)
            rows.append([base, kind[i], "%.3f" % ps[i], "%.2f" % iou] + ["%.0f" % v for v in pb[i]] + ["%.0f" % (pb[i, 3] - pb[i, 1])])
            tot[kind[i]] += 1
        for j in np.where(~used)[0]:
            iou = ious[:, j].max() if len(pb) else 0.0
            rows.append([base, "FN", "", "%.2f" % iou] + ["%.0f" % v for v in gb[j]] + ["%.0f" % (gb[j, 3] - gb[j, 1])])
            tot["FN"] += 1
        coco_imgs.append({"id": k + 1, "file_name": base, "width": w, "height": h})
        for i in range(len(pb)):
            x1, y1, x2, y2 = [float(v) for v in pb[i]]
            coco_anns.append({"id": len(coco_anns) + 1, "image_id": k + 1, "category_id": 1, "score": round(float(ps[i]), 3),
                              "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                              "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0, "segmentation": []})
        if a.all or "FP" in kind.values() or (~used).any():
            v = img.copy()
            for i in range(len(pb)):
                c = (0, 200, 0) if kind[i] == "TP" else (0, 0, 255)
                x1, y1, x2, y2 = [int(t) for t in pb[i]]
                cv2.rectangle(v, (x1, y1), (x2, y2), c, 3)
                cv2.putText(v, "%s %.2f" % (kind[i], ps[i]), (x1, max(20, y1 - 8)), 0, 0.8, c, 2)
            for j in np.where(~used)[0]:
                x1, y1, x2, y2 = [int(t) for t in gb[j]]
                cv2.rectangle(v, (x1, y1), (x2, y2), (0, 220, 255), 6)
                cv2.putText(v, "MISSED", (x1, min(h - 8, y2 + 28)), 0, 0.9, (0, 220, 255), 2)
            cv2.putText(v, "%s conf %.2f   green TP  red FP  yellow missed" % (name, a.conf), (20, 40), 0, 1.1, (255, 255, 255), 3)
            cv2.imwrite(os.path.join(a.out, "vis", base), v)
    with open(os.path.join(a.out, "errors.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "kind", "score", "iou", "x1", "y1", "x2", "y2", "height_px"])
        wr.writerows(rows)
    json.dump({"info": {"description": "%s detections, conf >= %.2f (model output, not ground truth)" % (name, a.conf)},
               "images": coco_imgs, "annotations": coco_anns, "categories": [{"id": 1, "name": a.cls}]},
              open(os.path.join(a.out, "predictions_coco.json"), "w"))
    gt = tot["TP"] + tot["FN"]
    print("%s @conf %.2f IoU %.2f: %d labelled people | TP %d  FN (missed) %d  FP (wrong) %d | recall %.3f precision %.3f"
          % (name, a.conf, a.iou, gt, tot["TP"], tot["FN"], tot["FP"], tot["TP"] / max(gt, 1), tot["TP"] / max(tot["TP"] + tot["FP"], 1)))
    print("frames drawn: %d -> %s" % (len(os.listdir(os.path.join(a.out, "vis"))), a.out))


if __name__ == "__main__":
    main()
