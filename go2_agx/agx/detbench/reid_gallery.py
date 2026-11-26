#!/usr/bin/env python3
"""Long-term Re-ID on top of a tracker's output (tracks.csv from reid_track.py), simulated as it would run live.

BoT-SORT only matches appearance near the predicted position, so a person who leaves and comes back gets a new ID.
Here every ENDED track keeps a gallery embedding (mean of its OSNet embeddings). When a NEW track has been seen for
--first frames, its mean embedding is compared with the galleries of tracks that ended before it started (appearance
only, no position); if the best cosine similarity >= threshold, the new track takes that old ID.

With --same (IDs a person checked by eye belong to ONE person), it also reports how well each threshold does.

  .venv/bin/python reid_gallery.py --tracks data/reid_on/tracks.csv --images data/frames_clip \
      --engine models/osnet_x0_25_msmt17_fp16.engine --same 2 13 14 23 41
"""
import argparse, csv, os, sys
from collections import defaultdict

import cv2
import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--tracks", required=True)
p.add_argument("--images", required=True)
p.add_argument("--engine", default="models/osnet_x0_25_msmt17_fp16.engine")
p.add_argument("--first", type=int, default=5, help="frames of a new track used to decide (live: decision after N frames)")
p.add_argument("--same", type=int, nargs="*", default=[], help="track IDs known to be the same person")
p.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
a = p.parse_args()

sys.path.append("/usr/lib/python3.10/dist-packages")
import boxmot.reid.backends.tensorrt_backend as tb  # noqa: E402
tb.require_reid_backend_requirements = lambda *args, **kw: ()
from boxmot.reid.core.runtime import ReID  # noqa: E402
reid = ReID(os.path.abspath(a.engine), device="cuda", half=False)

rows = defaultdict(list)  # frame -> [(tid, box)]
for r in csv.DictReader(open(a.tracks)):
    rows[r["frame"]].append((int(r["track_id"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
emb = defaultdict(list)   # tid -> [(clip_index, embedding)]
index = {}
for r in csv.DictReader(open(a.tracks)):
    index[r["frame"]] = int(r["clip_index"])
for f in sorted(rows, key=lambda k: index[k]):
    img = cv2.imread(os.path.join(a.images, f))
    tids = [t for t, _ in rows[f]]
    feats = np.asarray(reid(img, np.array([b for _, b in rows[f]], np.float32)), np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    for t, e in zip(tids, feats):
        emb[t].append((index[f], e))

tids = sorted(emb, key=lambda t: emb[t][0][0])
span = {t: (emb[t][0][0], emb[t][-1][0]) for t in tids}
unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
full = {t: unit(np.mean([e for _, e in emb[t]], 0)) for t in tids}            # gallery: whole track
head = {t: unit(np.mean([e for _, e in emb[t][:a.first]], 0)) for t in tids}  # new track: first N frames
same = set(a.same)

print("tracks: %d | embeddings per track = boxes the tracker output | new-track decision after %d frames" % (len(tids), a.first))
if same:
    print("\nsimilarity of each known-person track (first %d frames) to the earlier tracks' galleries:" % a.first)
    for t in tids:
        if t not in same:
            continue
        earlier = [(float(head[t] @ full[o]), o) for o in tids if span[o][1] < span[t][0]]
        earlier.sort(reverse=True)
        txt = ", ".join("%s%d %.2f" % ("*" if o in same else "", o, s) for s, o in earlier[:5])
        print("  ID %3d (frames %d-%d): %s" % (t, span[t][0], span[t][1], txt or "no earlier tracks"))
    print("  (* = same person)")


def run(thr):
    """Live simulation in start order: a new track takes the ID of the most similar ended track if sim >= thr."""
    label, ended = {}, []
    for t in tids:
        cands = [(float(head[t] @ full[o]), o) for o in ended if span[o][1] < span[t][0]]
        best = max(cands) if cands else (0.0, None)
        label[t] = label[best[1]] if best[1] is not None and best[0] >= thr else t
        ended.append(t)
    return label


print("\nthreshold | final IDs | merges | known person: IDs left (ideal 1) | merges into the known person from others")
for thr in a.thresholds:
    lab = run(thr)
    merges = [(t, lab[t]) for t in tids if lab[t] != t]
    known = sorted({lab[t] for t in same}) if same else []
    wrong = [t for t in tids if t not in same and same and lab[t] in {lab[s] for s in same}]
    print("   %.2f    |   %3d     |  %3d   | %s | %s" % (thr, len(set(lab.values())), len(merges),
          ("%d %s" % (len(known), known)) if same else "-", wrong if same else "-"))
    if merges:
        print("            merges (new -> old):", ", ".join("%d->%d" % m for m in merges))
