#!/usr/bin/env python3
"""Stage-by-stage latency of one detector on the GPU: where the time per frame goes.

Each stage is timed on the host with a CUDA synchronize before and after it, so the numbers add up to the real
per-frame latency (wall time). Inference is also timed with CUDA events on the engine's stream (pure GPU time).
Same preprocessing / postprocessing as bench.py and the live detector.

  YOLOv5 (TensorRT .engine or PyTorch .pt):
    letterbox (CPU) | to tensor BGR->RGB CHW (CPU) | upload uint8 (H2D) | normalise /255 (GPU) |
    inference | NMS (GPU) | boxes to CPU + rescale
  RF-DETR (TensorRT .engine):
    resize (CPU) | upload uint8 (H2D) | normalise RGB, mean/std (GPU) | inference | outputs to CPU (D2H) |
    decode: sigmoid, top-k, boxes (CPU)
JPEG decoding is reported separately: the robot receives raw images, so it is not part of the live latency.

  .venv/bin/python profile_stages.py --images data/frames_clip --weights models/rfdetr_medium_fp16.engine
"""
import argparse, csv, glob, os, sys, time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

S = torch.cuda.synchronize


class Timer:
    def __init__(self):
        self.t, self.rec = {}, None

    def start(self):
        S(); self.rec = time.perf_counter()

    def lap(self, name):
        S(); now = time.perf_counter()
        self.t.setdefault(name, []).append((now - self.rec) * 1000)
        self.rec = now


def gpu_events(stream):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return a, b


def run_yolo(d, img, T, conf, G):
    torch_ = d.torch
    T.start()
    lb = d.letterbox(img, d.imgsz, stride=32, auto=True)[0]
    T.lap("1 letterbox (CPU)")
    x = torch_.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)))
    T.lap("2 to tensor (CPU)")
    x = x.cuda()
    T.lap("3 upload (H2D)")
    x = (x.half() if d.half else x.float()).unsqueeze(0) / 255.0
    T.lap("4 normalise (GPU)")
    a, b = gpu_events(None)
    with torch_.no_grad():
        if d.backend == "tensorrt":
            a.record(d.trt.stream); raw = d.trt(x)[d.trt.outputs[0]].float(); b.record(d.trt.stream)
        else:
            a.record(); raw = d.model(x)[0]; b.record()
    T.lap("5 inference")
    S(); G.append(a.elapsed_time(b))
    with torch_.no_grad():
        p = d.nms(raw, conf_thres=conf, iou_thres=d.iou)[0]
    T.lap("6 NMS (GPU)")
    p[:, :4] = d.scale(x.shape[2:], p[:, :4], img.shape)
    p = p.float().cpu().numpy()
    T.lap("7 boxes to CPU + rescale")
    return p


def run_rfdetr(d, img, T, conf, G):
    h, w = img.shape[:2]
    T.start()
    small = cv2.resize(img, (d.res, d.res), interpolation=cv2.INTER_LINEAR)
    T.lap("1 resize (CPU)")
    x = torch.from_numpy(small).cuda()
    T.lap("2 upload (H2D)")
    x = x[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255
    x = (x - d.mean) / d.std
    T.lap("3 normalise (GPU)")
    a, b = gpu_events(None)
    a.record(d.trt.stream); out = d.trt(x); b.record(d.trt.stream)
    T.lap("4 inference")
    S(); G.append(a.elapsed_time(b))
    dets, logits = out["dets"][0].float().cpu().numpy(), out["labels"][0].float().cpu().numpy()
    T.lap("5 outputs to CPU (D2H)")
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -88, 88)))
    flat = scores.ravel()
    k = min(dets.shape[0], flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[flat[idx] > conf]
    idx = idx[np.argsort(-flat[idx])]
    q, c = np.divmod(idx, scores.shape[1])
    cx, cy, bw, bh = [dets[q, i] for i in range(4)]
    boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    T.lap("6 decode (CPU)")
    return boxes


def gpu_clock():
    for p in glob.glob("/sys/class/devfreq/*.gpu/cur_freq"):
        try:
            return int(open(p).read()) // 1000000
        except Exception:
            pass
    return -1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--yolov5-dir", default=os.path.join(bench.HOME, "odin_test/yolo_gpu/yolov5"))
    p.add_argument("--out", default="results/stages.csv")
    p.add_argument("--note", default="")
    a = p.parse_args()

    o = SimpleNamespace(weights=a.weights, yolov5_dir=a.yolov5_dir, imgsz=640, iou=0.45, device="cuda", fp32=a.fp32, threads=0)
    fam = "rfdetr" if "rfdetr" in os.path.basename(a.weights).lower() else "yolov5"
    d = (bench.RFDETR if fam == "rfdetr" else bench.YOLOv5)(o)
    name = os.path.splitext(os.path.basename(a.weights))[0] + ("" if d.backend == "tensorrt" else "_pytorch")
    run = run_rfdetr if fam == "rfdetr" else run_yolo

    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    if a.limit:
        files = files[:a.limit]
    dec = []
    imgs = []
    for f in files:
        t = time.perf_counter(); imgs.append(cv2.imread(f)); dec.append((time.perf_counter() - t) * 1000)
    T, G = Timer(), []
    for _ in range(a.warmup):
        run(d, imgs[0], T, a.conf, G)
    T, G, clocks = Timer(), [], []
    t0 = time.perf_counter()
    for i, img in enumerate(imgs):
        run(d, img, T, a.conf, G)
        if i % 50 == 0:
            clocks.append(gpu_clock())
    wall = (time.perf_counter() - t0) * 1000 / len(imgs)

    tot = np.sum([v for v in T.t.values()], axis=0)
    print("%s | %s | %d frames | GPU clock samples %s MHz" % (name, d.input_desc, len(imgs), sorted(set(clocks))))
    print("  %-28s %8s %8s %8s %7s" % ("stage", "mean ms", "p50", "p95", "share"))
    rows = []
    for k in sorted(T.t):
        v = np.array(T.t[k])
        print("  %-28s %8.2f %8.2f %8.2f %6.1f%%" % (k, v.mean(), np.percentile(v, 50), np.percentile(v, 95), 100 * v.mean() / tot.mean()))
        rows.append((k, v.mean(), np.percentile(v, 50), np.percentile(v, 95)))
    print("  %-28s %8.2f %8.2f %8.2f   -> %.1f FPS" % ("TOTAL per frame", tot.mean(), np.percentile(tot, 50), np.percentile(tot, 95), 1000 / tot.mean()))
    print("  %-28s %8.2f %8.2f %8.2f   (CUDA events on the engine stream)" % ("  of which pure GPU inference", np.mean(G), np.percentile(G, 50), np.percentile(G, 95)))
    print("  %-28s %8.2f   (offline only; the robot gets raw images)" % ("JPEG decode (not counted)", np.mean(dec)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["time", "model", "input", "frames", "stage", "mean_ms", "p50_ms", "p95_ms", "note"])
        stamp = time.strftime("%Y-%m-%d %H:%M")
        for k, m, p50, p95 in rows + [("TOTAL", tot.mean(), np.percentile(tot, 50), np.percentile(tot, 95)),
                                     ("GPU inference (events)", np.mean(G), np.percentile(G, 50), np.percentile(G, 95)),
                                     ("JPEG decode (offline)", np.mean(dec), np.percentile(dec, 50), np.percentile(dec, 95))]:
            wr.writerow([stamp, name, d.input_desc, len(imgs), k, "%.2f" % m, "%.2f" % p50, "%.2f" % p95, a.note])


if __name__ == "__main__":
    main()
