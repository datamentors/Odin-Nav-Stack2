#!/usr/bin/env python3
"""Prompt once, then track: SAM 2 follows one object through consecutive frames from a single starting box
(e.g. a Florence-2 box). Measures per-frame time and whether the track stays on the object.

Per frame:  preprocess (resize to 1024, normalise; done by SAM 2 when the frames are loaded, reported per frame)
            + track (image encoder + memory attention + mask decoder, timed with CUDA sync)
            + mask -> box (CPU)
With --labels, on frames that have ground truth the tracked box is compared with the labelled people:
"on person" = IoU >= 0.5 with one of them (identity is not checked).

  .venv/bin/python sam2_track.py --images data/frames_clip --start 454 --frames 130 --box 865 204 1077 862 \
      --out data/sam2_B
"""
import argparse, csv, json, os, shutil, sys, tempfile, time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
          "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder of consecutive frames (sorted by name)")
    p.add_argument("--start", type=int, required=True, help="index of the first frame (the prompt frame)")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--box", type=float, nargs=4, action="append", required=True, metavar=("X1", "Y1", "X2", "Y2"),
                   help="prompt box in pixels; repeat --box to track several objects (obj 1, 2, ...)")
    p.add_argument("--model", default="tiny", choices=list(MODELS))
    p.add_argument("--image-size", type=int, default=1024, help="SAM 2 input size (1024 = default; 512 = ~4x less compute)")
    p.add_argument("--labels", help="COCO ground truth to check the track on labelled frames")
    p.add_argument("--out", required=True)
    p.add_argument("--vis-every", type=int, default=10)
    p.add_argument("--video", help="write every frame with the masks to this .mp4 (10 fps, 960 px wide)")
    p.add_argument("--note", default="")
    a = p.parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(".jpg"))[a.start:a.start + a.frames]
    tmp = tempfile.mkdtemp(prefix="sam2_")  # SAM 2 wants frames named 00000.jpg, 00001.jpg, ...
    for i, f in enumerate(files):
        os.symlink(os.path.abspath(os.path.join(a.images, f)), os.path.join(tmp, "%05d.jpg" % i))
    cfg, ckpt = MODELS[a.model]
    os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)

    t0 = time.perf_counter()
    hydra_over = [] if a.image_size == 1024 else ["++model.image_size=%d" % a.image_size]
    pred = build_sam2_video_predictor(cfg, os.path.join(HERE, "models/sam2", ckpt), device="cuda", hydra_overrides_extra=hydra_over)
    t_load = time.perf_counter() - t0
    gts = None
    if a.labels:
        full = [os.path.join(a.images, f) for f in files]
        gts, _ = bench.load_labels(a.labels, full)
        labelled = {os.path.basename(im["file_name"]) for im in json.load(open(a.labels))["images"]}

    rows, track_ms, post_ms, writer = [], [], [], None
    colors = [(0, 0, 255), (255, 0, 255), (255, 160, 0), (0, 255, 255)]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        state = pred.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=False)
        torch.cuda.synchronize(); t_init = time.perf_counter() - t0
        H, W = state["video_height"], state["video_width"]
        t0 = time.perf_counter()
        for k, b in enumerate(a.box):
            pred.add_new_points_or_box(state, frame_idx=0, obj_id=k + 1, box=np.array(b, np.float32))
        torch.cuda.synchronize(); t_prompt = time.perf_counter() - t0
        torch.cuda.synchronize(); t_prev = time.perf_counter()
        for fi, obj_ids, logits in pred.propagate_in_video(state):
            masks = (logits[:, 0] > 0)
            torch.cuda.synchronize(); t_now = time.perf_counter()
            track_ms.append((t_now - t_prev) * 1000)
            ms = masks.cpu().numpy()
            boxes = []
            for m in ms:
                ys, xs = np.nonzero(m)
                boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else None)
            post_ms.append((time.perf_counter() - t_now) * 1000)
            labelled_here = gts is not None and files[fi] in labelled
            g = np.array([bb for c, bb in (gts[fi] or []) if c == "person"], np.float32).reshape(-1, 4) if labelled_here else None
            statuses = []
            for k, (oid, m, box) in enumerate(zip(obj_ids, ms, boxes)):
                status, iou = "", ""
                if labelled_here:
                    if box is None:
                        status = "lost" if len(g) else "absent-ok"
                    else:
                        best = bench.iou_matrix(np.array([box], np.float32), g).max() if len(g) else 0.0
                        iou = "%.2f" % best
                        status = "on person" if best >= 0.5 else ("partial" if best >= 0.2 else "off person")
                statuses.append((oid, status, iou))
                rows.append([a.start + fi, files[fi], oid, "%.1f" % track_ms[-1], "%.1f" % post_ms[-1], int(m.sum())] +
                            (["%.0f" % v for v in box] if box else ["", "", "", ""]) + [status, iou])
            save_jpg = fi % a.vis_every == 0 or any(st in ("lost", "off person") for _, st, _ in statuses)
            if save_jpg or a.video:
                img = cv2.imread(os.path.join(a.images, files[fi]))
                vis = img.copy()
                for k, (m, box) in enumerate(zip(ms, boxes)):
                    c = colors[k % len(colors)]
                    vis[m] = (0.5 * vis[m] + 0.5 * np.array(c)).astype(np.uint8)
                    if box:
                        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), c, 4)
                        cv2.putText(vis, "obj %d" % obj_ids[k], (int(box[0]), int(box[1]) - 8), 0, 1.0, c, 2)
                if labelled_here:
                    for c, bb in (gts[fi] or []):
                        cv2.rectangle(vis, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 220, 0), 2)
                txt = "frame %d  " % (a.start + fi) + "  ".join("obj%d %s %s" % x for x in statuses if x[1])
                cv2.putText(vis, txt, (20, 50), 0, 1.2, (255, 255, 255), 3)
                if save_jpg:
                    cv2.imwrite(os.path.join(a.out, "vis", "%04d.jpg" % (a.start + fi)), vis)
                if a.video:
                    small = cv2.resize(vis, (960, int(960 * H / W)))
                    if writer is None:
                        writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 10, small.shape[1::-1])
                    writer.write(small)
            t_prev = time.perf_counter()
    shutil.rmtree(tmp)
    if writer is not None:
        writer.release()

    tm = np.array(track_ms[1:])  # frame 0 is the prompt frame (already computed by add_new_points_or_box)
    pre = t_init * 1000 / len(files)
    with open(os.path.join(a.out, "track.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["clip_index", "frame", "obj", "track_ms", "post_ms", "mask_px", "x1", "y1", "x2", "y2", "check", "iou"])
        wr.writerows(rows)
    print("SAM 2.1 %s @%d | %d object(s) | frames %d-%d (%d) | %dx%d | %s"
          % (a.model, a.image_size, len(a.box), a.start, a.start + len(files) - 1, len(files), W, H, a.note))
    print("  model load %.1f s, prompt frame %.0f ms" % (t_load, t_prompt * 1000))
    print("  per frame: preprocess %.1f ms | track mean %.1f  p50 %.1f  p95 %.1f ms | mask->box %.1f ms"
          % (pre, tm.mean(), np.percentile(tm, 50), np.percentile(tm, 95), np.mean(post_ms)))
    e2e = pre + tm.mean() + np.mean(post_ms)
    print("  end to end ~%.1f ms per frame -> %.1f FPS (preprocess includes JPEG decode)" % (e2e, 1000 / e2e))
    for oid in sorted({r[2] for r in rows}):
        rr = [r for r in rows if r[2] == oid]
        lost_at = next((r[0] for r in rr if r[6] == ""), None)
        checks = [r[-2] for r in rr if r[-2]]
        print("  obj %d: mask first empty at frame %s | labelled frames: %s" % (oid, lost_at if lost_at is not None else "never",
              ", ".join("%s %d" % (k, checks.count(k)) for k in sorted(set(checks))) or "-"))
    print("  -> %s" % a.out)


if __name__ == "__main__":
    main()
