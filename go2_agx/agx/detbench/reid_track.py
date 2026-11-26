#!/usr/bin/env python3
"""RF-DETR (TensorRT) + BoT-SORT (BoxMOT) with or without appearance Re-ID, on consecutive frames.

Per frame:  detector (pre | inference | post, person boxes) -> BoT-SORT update, split into
            Re-ID embedding (crop + OSNet on the GPU) | camera-motion compensation (CPU) | association + Kalman.
Re-ID check (no identity labels): a "return" is a track ID that disappears for >= --gap seconds and comes back
with the SAME ID. Whether it came back on the SAME PERSON has to be checked by eye in the video.

  .venv/bin/python reid_track.py --images data/frames_clip --weights models/rfdetr_large_fp16.engine \
      --out data/reid_on --video data/reid_on/track.mp4
  .venv/bin/python reid_track.py ... --no-reid --out data/reid_off
"""
import argparse, csv, os, sys, time
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402


class Timer:
    """Wraps obj.<method> and adds its wall time (ms) to .ms; .take() returns and resets it."""

    def __init__(self, obj, method, cuda=False):
        self.ms, f = 0.0, getattr(obj, method)

        def wrapped(*args, **kw):
            if cuda:
                bench.sync("cuda")
            t = time.perf_counter()
            r = f(*args, **kw)
            if cuda:
                bench.sync("cuda")
            self.ms += (time.perf_counter() - t) * 1000
            return r
        setattr(obj, method, wrapped)

    def take(self):
        v, self.ms = self.ms, 0.0
        return v


def stats(v):
    v = np.asarray(v, float)
    return "%6.1f ms  p50 %6.1f  p95 %6.1f" % (v.mean(), np.percentile(v, 50), np.percentile(v, 95))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder of consecutive frames, named <bag>_<unix time>.jpg")
    p.add_argument("--weights", default="models/rfdetr_large_fp16.engine")
    p.add_argument("--conf", type=float, default=0.1, help="detections passed to the tracker (BoT-SORT uses low ones too)")
    p.add_argument("--no-reid", action="store_true", help="BoT-SORT with motion only (no appearance embeddings)")
    p.add_argument("--reid-model", default="osnet-x0-25-msmt17",
                   help="BoxMOT model name (PyTorch) or a TensorRT .engine from export_reid.py")
    p.add_argument("--lost-frames", type=int, default=40, help="frames a lost track is kept (BoxMOT default 40 = 4 s at 10 Hz)")
    p.add_argument("--gap", type=float, default=1.0, help="seconds missing to count as a 'return'")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=0, help="0 = all")
    p.add_argument("--out", required=True)
    p.add_argument("--video", help="write every frame with the track IDs to this .mp4 (960 px wide)")
    a = p.parse_args()

    from boxmot import BotSort
    from boxmot.reid.specs import ReIDConfig
    from boxmot.trackers.botsort.config import BotSortConfig
    from boxmot.components.timing import timing_event_sink

    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(".jpg"))
    files = files[a.start:a.start + a.frames] if a.frames else files[a.start:]
    stamps = np.array([float(os.path.splitext(f)[0].rsplit("_", 1)[-1]) for f in files])  # <bag>_<unix time>.jpg
    os.makedirs(a.out, exist_ok=True)

    det = bench.RFDETR(SimpleNamespace(weights=a.weights, threads=0))
    cfg = BotSortConfig(use_embeddings=not a.no_reid, max_lost_frames=a.lost_frames)
    if a.no_reid:
        reid = None
    elif a.reid_model.endswith(".engine"):
        # JetPack's TensorRT (not on PyPI for Jetson): skip BoxMOT's pip check for 'nvidia-tensorrt'
        sys.path.append("/usr/lib/python3.10/dist-packages")
        import boxmot.reid.backends.tensorrt_backend as trt_backend
        trt_backend.require_reid_backend_requirements = lambda *args, **kw: ()
        # engine I/O is fp32 (fp16 inside); 'fp16' here would hand it a half-precision batch
        reid = ReIDConfig(os.path.abspath(a.reid_model), device="cuda", precision="fp32")
    else:
        reid = ReIDConfig(a.reid_model, device="cuda", precision="fp16")
    tracker = BotSort(cfg, reid=reid)

    img0 = cv2.imread(os.path.join(a.images, files[0]))
    for _ in range(10):  # warm-up detector
        det.detect(img0, a.conf)
    enc_t = cmc_t = None

    T = {k: [] for k in ("read", "det pre", "det inference", "det post", "reid", "cmc", "association", "tracker total")}
    R = {k: [] for k in ("preprocess", "process", "postprocess")}   # BoxMOT's own Re-ID phase timers
    ev = {}
    sink = lambda e: ev.__setitem__(e.phase, ev.get(e.phase, 0.0) + e.elapsed_ms) if e.component == "reid" else None
    rows, seen, writer, n_dets = [], {}, None, []
    rng = np.random.RandomState(1)
    colors = {}
    for i, f in enumerate(files):
        t0 = time.perf_counter()
        img = cv2.imread(os.path.join(a.images, f))
        t1 = time.perf_counter()
        boxes, scores, names = det.detect(img, a.conf)
        keep = [k for k, nm in enumerate(names) if nm == "person"]
        dets = np.zeros((len(keep), 6), np.float32)
        if keep:
            dets[:, :4], dets[:, 4] = boxes[keep], scores[keep]
        n_dets.append(int((dets[:, 4] >= 0.5).sum()))
        t2 = time.perf_counter()
        ev.clear()
        with timing_event_sink(sink):
            tracks = tracker.update(dets, img)
        t3 = time.perf_counter()
        if cmc_t is None and getattr(tracker, "cmc", None) is not None:
            cmc_t = Timer(tracker.cmc, "apply")
            cmc_t.ms = 0.0
        enc = getattr(tracker, "_reid_encoder", None)  # BoxMOT 26 builds it on the first update that has detections
        if not a.no_reid and enc_t is None and enc is not None:
            enc_t = Timer(enc, "encode", cuda=True)     # this frame included building the encoder: not counted
            cmc_t.take() if cmc_t else None
        elif i > 0:
            r = enc_t.take() if enc_t else 0.0
            c = cmc_t.take() if cmc_t else 0.0
            for k, v in zip(T, ((t1 - t0) * 1000, *(x * 1000 for x in det.t), r, c, (t3 - t2) * 1000 - r - c, (t3 - t2) * 1000)):
                T[k].append(v)
            for k in R:
                R[k].append(ev.get(k, 0.0))
        tracks = np.asarray(tracks).reshape(-1, tracks.shape[1] if np.ndim(tracks) == 2 else 8)
        for tr in tracks:
            x1, y1, x2, y2, tid, conf = tr[:6]
            tid = int(tid)
            seen.setdefault(tid, []).append(i)
            rows.append([a.start + i, f, tid, "%.2f" % conf, "%.0f" % x1, "%.0f" % y1, "%.0f" % x2, "%.0f" % y2])
        if a.video:
            vis = img.copy()
            for tr in tracks:
                x1, y1, x2, y2, tid = [int(v) for v in tr[:5]]
                col = colors.setdefault(tid, tuple(int(v) for v in rng.randint(60, 255, 3)))
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 4)
                cv2.putText(vis, "ID %d" % tid, (x1, max(30, y1 - 10)), 0, 1.4, col, 3)
            cv2.putText(vis, "frame %d  t=%.1f s  %s" % (a.start + i, stamps[i] - stamps[0], "BoT-SORT + Re-ID" if not a.no_reid else "BoT-SORT, no Re-ID"),
                        (20, 50), 0, 1.2, (255, 255, 255), 3)
            small = cv2.resize(vis, (960, int(960 * img.shape[0] / img.shape[1])))
            if writer is None:
                writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 10, small.shape[1::-1])
            writer.write(small)
    if writer is not None:
        writer.release()

    with open(os.path.join(a.out, "tracks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_index", "frame", "track_id", "conf", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    # ID lifetimes and returns (same ID after >= gap seconds missing)
    lines, returns = [], []
    for tid in sorted(seen):
        idx = seen[tid]
        segs = [[idx[0], idx[0]]]
        for j in idx[1:]:
            if stamps[j] - stamps[segs[-1][1]] >= a.gap:
                returns.append((tid, stamps[j] - stamps[segs[-1][1]], a.start + segs[-1][1], a.start + j))
                segs.append([j, j])
            else:
                segs[-1][1] = j
        lines.append("  ID %3d  frames %4d-%4d  (%5.1f s, %4d frames tracked)%s"
                     % (tid, a.start + idx[0], a.start + idx[-1], stamps[idx[-1]] - stamps[idx[0]], len(idx),
                        "  returns: " + ", ".join("after %.1f s at frame %d" % (g, fr) for t_, g, _, fr in returns if t_ == tid)
                        if any(t_ == tid for t_, *_ in returns) else ""))

    dur = stamps[-1] - stamps[0]
    print("RF-DETR %s + BoT-SORT (%s) | %d frames, %.1f s (%.1f Hz) | %dx%d | lost track kept %d frames"
          % (os.path.basename(a.weights), "Re-ID " + a.reid_model if not a.no_reid else "no Re-ID", len(files), dur,
             (len(files) - 1) / dur, img0.shape[1], img0.shape[0], a.lost_frames))
    print("  people with conf >= 0.5 per frame: mean %.2f, max %d" % (np.mean(n_dets), max(n_dets)))
    if not a.no_reid and enc_t is None:
        print("  note: the Re-ID encoder was never built (no detections?), so there is no Re-ID time")
    print("  per frame:")
    for k in ("det pre", "det inference", "det post", "reid", "cmc", "association", "tracker total"):
        if k == "reid" and a.no_reid:
            continue
        print("    %-14s %s" % (k, stats(T[k])))
        if k == "reid":
            for kk, nm in (("preprocess", "crop+resize+upload"), ("process", "OSNet forward"), ("postprocess", "normalise")):
                print("      %-18s %s" % (nm, stats(R[kk])))
    e2e = np.array(T["det pre"]) + np.array(T["det inference"]) + np.array(T["det post"]) + np.array(T["tracker total"])
    print("    %-14s %s  -> %.1f FPS (JPEG read %.1f ms not included)" % ("END TO END", stats(e2e), 1000 / e2e.mean(), np.mean(T["read"])))
    print("  unique track IDs: %d | returns with the same ID after >= %.1f s missing: %d" % (len(seen), a.gap, len(returns)))
    print("\n".join(lines))
    print("  -> %s" % a.out)


if __name__ == "__main__":
    main()
