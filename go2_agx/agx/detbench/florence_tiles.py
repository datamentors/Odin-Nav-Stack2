#!/usr/bin/env python3
"""Times Florence-2's tiled 'far' search exactly as florence_service.py find_far() does it, stage by stage:
whole image | 6 tiles (3 x 2, 20 % overlap, full resolution) | merge | YOLO confirmation of tile-only boxes.
Each Florence-2 pass is split into pre-processing, inference (generate) and post-processing (decode + parse).
Imports detect_all-equivalent code from florence_service.py's logic (copied, so the service is not started).

  cd ~/odin-perception && .venv/bin/python ~/odin_test/detbench/florence_tiles.py ~/odin_test/detbench/data/frames_label \
      --phrase person --every 7 --yolo http://127.0.0.1:8751
"""
import argparse, glob, json, os, sys, time, urllib.request

import numpy as np
from PIL import Image

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("frames")
p.add_argument("--phrase", default="person")
p.add_argument("--every", type=int, default=7, help="use every Nth frame")
p.add_argument("--yolo", default="", help="YOLO service URL for the confirmation step ('' = skip)")
p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="float32 = as the service runs it")
p.add_argument("--tile", type=int, default=0, help="0 = Javier's 3x2 tiles with 20%% overlap; N = NxN tiles (3x2 for 1600x1296), evenly spread, no resize at N=768")
p.add_argument("--save", help="write each frame's boxes (whole image, and far search after YOLO confirm) to this JSON")
a = p.parse_args()
FRAMES = os.path.abspath(a.frames)
sys.path.insert(0, os.path.expanduser("~/odin-perception"))
os.chdir(os.path.expanduser("~/odin-perception"))
from inspection_perception.cli import make_backend  # noqa: E402

t0 = time.time()
BACKEND = make_backend("florence2", model_dir="models/florence2-base-ft", dtype=a.dtype, device="cuda")
print("Florence-2 base (%s) loaded in %.1f s" % (a.dtype, time.time() - t0), flush=True)


def _area(b):
    return max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])


def _inter(a, b):
    w = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    h = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    return max(0.0, w) * max(0.0, h)


def detect_all(rgb, phrase):
    """florence_service.detect_all (= Florence2Backend._run_task + box clipping), split into timed stages.
    Returns (boxes, (pre_ms, inference_ms, post_ms))."""
    import torch
    task = "<OPEN_VOCABULARY_DETECTION>"
    torch.cuda.synchronize(); t0 = time.perf_counter()
    image = Image.fromarray(rgb)                                   # pre: resize 768x768 + normalise + tokenise + upload
    inputs = BACKEND.processor(text=task + phrase, images=image, return_tensors="pt")
    inputs = {k: (v.to(BACKEND.device, BACKEND._dtype) if v.dtype.is_floating_point else v.to(BACKEND.device))
              for k, v in inputs.items()}
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.inference_mode():                                   # inference: generate (3 beams, as the service)
        ids = BACKEND.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                     max_new_tokens=1024, num_beams=3, do_sample=False)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    raw = BACKEND.processor.batch_decode(ids, skip_special_tokens=False)[0]   # post: tokens -> text -> boxes
    out = BACKEND.processor.post_process_generation(raw, task=task, image_size=(image.width, image.height)).get(task, {}) or {}
    H, W = rgb.shape[:2]
    boxes = []
    for b in out.get("bboxes") or []:
        if b is None or len(b) < 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in b[:4]]
        x0, x1 = max(0.0, min(x0, x1)), min(W - 1.0, max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(H - 1.0, max(y0, y1))
        if (x1 - x0) >= 6 and (y1 - y0) >= 6:
            boxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
    t3 = time.perf_counter()
    return boxes, ((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)


def yolo_confirm(rgb, b, cls_id):  # florence_service._yolo_confirms (1x then 2x margin crops)
    import cv2
    H, W = rgb.shape[:2]
    bw, bh = b["x1"] - b["x0"], b["y1"] - b["y0"]
    calls = 0
    for frac in (1.0, 2.0):
        m = max(frac * max(bw, bh), 40)
        x0, y0 = int(max(0, b["x0"] - m)), int(max(0, b["y0"] - m))
        x1, y1 = int(min(W, b["x1"] + m)), int(min(H, b["y1"] + m))
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1, ::-1])
        ch, cw = crop.shape[:2]
        r = 640.0 / max(ch, cw)
        nw, nh = int(round(cw * r)), int(round(ch * r))
        lb = cv2.resize(crop, (nw, nh))
        lb = cv2.copyMakeBorder(lb, 0, (32 - nh % 32) % 32, 0, (32 - nw % 32) % 32, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        req = urllib.request.Request(a.yolo + "/infer?conf=0.20", data=lb.tobytes(), headers={"X-Shape": "%d,%d" % lb.shape[:2]}, method="POST")
        rows = json.loads(urllib.request.urlopen(req, timeout=10).read()); calls += 1
        tb = {"x0": (b["x0"] - x0) * r, "y0": (b["y0"] - y0) * r, "x1": (b["x1"] - x0) * r, "y1": (b["y1"] - y0) * r}
        for x_0, y_0, x_1, y_1, conf, k in rows:
            if int(k) == cls_id and _inter(tb, {"x0": x_0, "y0": y_0, "x1": x_1, "y1": y_1}) > 0.3 * min(_area(tb), _area({"x0": x_0, "y0": y_0, "x1": x_1, "y1": y_1})):
                return True, calls
        if x0 == 0 and y0 == 0 and x1 == W and y1 == H:
            break
    return False, calls


cls_id = None
if a.yolo:
    names = json.loads(urllib.request.urlopen(a.yolo + "/names", timeout=3).read())
    cls_id = {v.lower(): int(k) for k, v in names.items()}.get(a.phrase.lower())

files = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))[::a.every]
rgb0 = np.array(Image.open(files[0]).convert("RGB"))
detect_all(rgb0, a.phrase); detect_all(np.ascontiguousarray(rgb0[:720, :615]), a.phrase)  # warm-up
whole, tile, far = [], [], []     # rows of (pre, inference, post, end to end) in ms
n_new, n_calls, saved = [], [], {}
for f in files:
    rgb = np.array(Image.open(f).convert("RGB"))
    H, W = rgb.shape[:2]
    t_start = time.perf_counter()
    full, st = detect_all(rgb, a.phrase)
    whole.append(st + (sum(st),))
    pre, inf, post = list(st)
    cols, rows_, ov = 3, 2, 0.20
    if a.tile:
        tw, th = min(a.tile, W), min(a.tile, H)
        cols, rows_ = -(-W // tw), -(-H // th)
        xs = [round(c * (W - tw) / max(1, cols - 1)) for c in range(cols)]
        ys = [round(r * (H - th) / max(1, rows_ - 1)) for r in range(rows_)]
    else:
        tw, th = int(W / (cols - (cols - 1) * ov)), int(H / (rows_ - (rows_ - 1) * ov))
        xs = [int(c * tw * (1 - ov)) for c in range(cols)]
        ys = [int(r * th * (1 - ov)) for r in range(rows_)]
    tiles = []
    for y0 in ys:
        for x0 in xs:
            x1, y1 = min(W, x0 + tw), min(H, y0 + th)
            bs, st = detect_all(np.ascontiguousarray(rgb[y0:y1, x0:x1]), a.phrase)
            tile.append(st + (sum(st),))
            pre += st[0]; inf += st[1]; post += st[2]
            for b in bs:
                tiles.append({"x0": b["x0"] + x0, "y0": b["y0"] + y0, "x1": b["x1"] + x0, "y1": b["y1"] + y0})
    t2 = time.perf_counter()
    tiles.sort(key=_area, reverse=True)
    merged = []
    for b in tiles:
        if all(_inter(b, m) < 0.5 * min(_area(b), _area(m)) for m in merged):
            merged.append(b)
    new = [b for b in merged if all(_inter(b, f_) < 0.3 * min(_area(b), _area(f_)) for f_ in full)]
    calls, kept = 0, []
    if a.yolo and cls_id is not None:
        for b in new:
            ok, c = yolo_confirm(rgb, b, cls_id); calls += c
            if ok:
                kept.append(b)
    t4 = time.perf_counter()
    post += (t4 - t2) * 1000                       # merge + YOLO confirmation count as post-processing
    far.append((pre, inf, post, (t4 - t_start) * 1000))
    n_new.append(len(new)); n_calls.append(calls)
    saved[os.path.basename(f)] = {"whole": [[b["x0"], b["y0"], b["x1"], b["y1"]] for b in full],
                                  "far": [[b["x0"], b["y0"], b["x1"], b["y1"]] for b in full + kept]}
    print("  %s  far search %.1f s (%d tile-only boxes, %d YOLO calls)"
          % (os.path.basename(f)[-18:], far[-1][3] / 1000, len(new), calls), flush=True)

if a.save:
    json.dump(saved, open(a.save, "w"))
print(("\nFlorence-2 base %s," % a.dtype) + " <OPEN_VOCABULARY_DETECTION> '%s', %d frames, conf: n/a (Florence-2 has no score)" % (a.phrase, len(files)))
print("Model | Input | Pre-processing | Inference | Post-processing | End to end (mean / p50 / p95) | FPS")
for name, inp, rows in (("Florence-2 whole image", "%dx%d -> 768x768" % (W, H), whole),
                        ("Florence-2 one tile", "%dx%d -> 768x768" % (tw, th), tile),
                        ("Florence-2 far search (1 + %d tiles)" % (len(xs) * len(ys)), "%dx%d, %d passes" % (W, H, 1 + len(xs) * len(ys)), far)):
    v = np.array(rows)
    print("%s | %s | %.0f ms | %.0f ms | %.0f ms | %.0f / %.0f / %.0f ms | %.2f"
          % (name, inp, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean(),
             np.percentile(v[:, 3], 50), np.percentile(v[:, 3], 95), 1000 / v[:, 3].mean()))
print("far search: %.1f tile-only boxes and %.1f YOLO calls per frame (merge + YOLO confirm are in its post-processing)"
      % (np.mean(n_new), np.mean(n_calls)))
