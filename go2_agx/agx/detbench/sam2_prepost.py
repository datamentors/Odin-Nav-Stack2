#!/usr/bin/env python3
"""Measures SAM 2's preprocessing and postprocessing per frame on their own (the tracking itself is in sam2_track.py).

Preprocessing, as on the robot (frame already decoded, BGR uint8 1600x1296):
  resize to SxS (CPU) | BGR->RGB + to tensor (CPU) | upload (H2D) | /255, ImageNet mean/std (GPU)
Preprocessing as SAM 2's offline loader does it (for reference): JPEG decode + PIL resize + to tensor + normalise.
Postprocessing, per tracked object: low-res mask logits (256x256) -> upsample to 1600x1296 (GPU; SAM 2 does this inside
propagate_in_video) | threshold | copy mask to CPU | mask -> box.

  .venv/bin/python sam2_prepost.py --images data/frames_clip --sizes 512 1024 --objects 1 2
"""
import argparse, glob, os, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(3, 1, 1)
S = torch.cuda.synchronize


def stats(v):
    v = np.array(v)
    return v.mean(), np.percentile(v, 50), np.percentile(v, 95)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--objects", type=int, nargs="+", default=[1, 2])
    p.add_argument("--frames", type=int, default=200)
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[:a.frames]
    imgs = [cv2.imread(f) for f in files]
    H, W = imgs[0].shape[:2]

    for size in a.sizes:
        t = {k: [] for k in ("resize", "to_tensor", "upload", "normalise", "offline_loader")}
        for rep in range(2):  # first pass = warm-up
            for f, img in zip(files, imgs):
                S(); t0 = time.perf_counter()
                small = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
                t1 = time.perf_counter()
                x = torch.from_numpy(np.ascontiguousarray(small[:, :, ::-1].transpose(2, 0, 1)))
                t2 = time.perf_counter()
                x = x.cuda(); S(); t3 = time.perf_counter()
                x = (x.float() / 255 - MEAN) / STD; S(); t4 = time.perf_counter()
                # SAM 2's own loader (sam2.utils.misc._load_img_as_tensor + normalise), includes JPEG decode
                y = Image.open(f).convert("RGB").resize((size, size))
                y = torch.from_numpy(np.array(y) / 255.0).permute(2, 0, 1).float().cuda()
                y = (y - MEAN) / STD; S(); t5 = time.perf_counter()
                if rep:
                    for k, v in zip(("resize", "to_tensor", "upload", "normalise", "offline_loader"),
                                    (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                        t[k].append(v * 1000)
        pre = np.array(t["resize"]) + np.array(t["to_tensor"]) + np.array(t["upload"]) + np.array(t["normalise"])
        print("SAM 2 input %d  (%d frames, %dx%d)" % (size, len(files), W, H))
        print("  PREPROCESS (robot)   %6.2f ms  p50 %.2f  p95 %.2f   = resize %.2f + to tensor %.2f + upload %.2f + normalise %.2f"
              % (*stats(pre), *[np.mean(t[k]) for k in ("resize", "to_tensor", "upload", "normalise")]))
        print("  (SAM 2 offline loader incl. JPEG decode: %.1f ms)" % np.mean(t["offline_loader"]))

        for n in a.objects:
            # realistic low-res mask logits (SAM 2: 256x256 for any input size): one person-sized blob per object
            low = torch.full((n, 1, 256, 256), -10.0, device="cuda")
            for k in range(n):
                low[k, 0, 100:150, 60 + 40 * k:80 + 40 * k] = 10.0   # ~ 125x260 px at 1600x1296, like a far standing person
            u, c, b, g = [], [], [], []
            for rep in range(2):
                for _ in files:
                    S(); t0 = time.perf_counter()
                    m = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False) > 0
                    S(); t1 = time.perf_counter()
                    mc = m[:, 0].cpu().numpy()
                    t2 = time.perf_counter()
                    boxes = []
                    for mm in mc:
                        ys, xs = np.nonzero(mm)
                        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None)
                    t3 = time.perf_counter()
                    # faster alternative: box from the mask on the GPU (row/column any), only 4 numbers per object to the CPU
                    S(); t4 = time.perf_counter()
                    rows, cols = m[:, 0].any(2), m[:, 0].any(1)
                    ar_h, ar_w = torch.arange(H, device="cuda"), torch.arange(W, device="cuda")
                    big = 1 << 30
                    gb = torch.stack([torch.where(cols, ar_w, big).min(1).values, torch.where(rows, ar_h, big).min(1).values,
                                      torch.where(cols, ar_w, -1).max(1).values, torch.where(rows, ar_h, -1).max(1).values], 1).cpu()
                    t5 = time.perf_counter()
                    if rep:
                        u.append((t1 - t0) * 1000); c.append((t2 - t1) * 1000); b.append((t3 - t2) * 1000); g.append((t5 - t4) * 1000)
            assert all(tuple(int(v) for v in gb[k]) == tuple(int(v) for v in boxes[k]) for k in range(n)), (gb, boxes)
            post = np.array(c) + np.array(b)
            print("  POSTPROCESS %d obj    %6.2f ms  p50 %.2f  p95 %.2f   = mask to CPU %.2f + mask->box %.2f"
                  "   | on GPU instead: %.2f ms (same boxes)   (upsample inside SAM 2's tracking: %.2f ms)"
                  % (n, *stats(post), np.mean(c), np.mean(b), np.mean(g), np.mean(u)))


if __name__ == "__main__":
    main()
