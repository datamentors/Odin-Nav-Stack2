#!/usr/bin/env python3
"""Offline tests of the semantic map store and BEV drawing (no ROS): python3 test_object_store.py [out_dir]
With out_dir, the two BEV test images are written there to look at."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bev  # noqa: E402
from object_store import ObjectStore, Params  # noqa: E402


def det(label, x, y, score=0.8, track=None, source="yolo", emb=None):
    return {"label": label, "score": score, "position": np.array([x, y, 0.5]), "source": source, "track_id": track,
            "size": [0.5, 0.9], "embedding": emb}


def test_merge_and_publish():
    s = ObjectStore(Params(min_observations=3))
    for k in range(5):   # the same chair seen 5 times with noise
        s.ingest(det("Chair", 2.0 + 0.05 * (-1) ** k, 1.0), 10.0 + k)
    s.ingest(det("chair", 5.0, 1.0), 15.0)          # another chair 3 m away: new object
    assert len(s.objects) == 2 and len(s.published()) == 1
    o = s.published()[0]
    assert o.observations == 5 and abs(o.position[0] - 2.0) < 0.06 and o.label == "chair"


def test_label_gate_and_votes():
    s = ObjectStore()
    a, _ = s.ingest(det("chair", 0, 0), 1)
    b, new = s.ingest(det("table", 0.1, 0), 2)       # a different label at the same place: separate object
    assert new and a != b
    s.ingest(det("couch", 0, 0, track=7), 3)         # a track carries over
    s.ingest(det("chair", 0, 0, track=7), 4)
    _, new = s.ingest(det("couch", 0, 0, score=0.5, track=7), 4.5)
    assert not new


def test_track_id_and_jump():
    s = ObjectStore(Params(assoc_dist_m=0.5))
    a, _ = s.ingest(det("person", 0, 0, track=3), 1)
    b, new = s.ingest(det("person", 0.9, 0, track=3), 1.5)   # walked 0.9 m: same track -> same object
    assert b == a and not new
    c, new = s.ingest(det("person", 9, 0, track=3), 2)       # 9 m jump: a reused ID, not the same object
    assert new and c != a
    d, new = s.ingest(det("person", 0.9, 0, track=3), 10)    # after track_memory_s: matched by distance + label
    assert d == a and not new


def test_embedding_gate():
    s = ObjectStore()
    e1, e2 = [1, 0, 0], [0, 1, 0]
    a, _ = s.ingest(det("bottle", 0, 0, emb=e1), 1)
    b, new = s.ingest(det("bottle", 0.1, 0, emb=e2), 2)      # same label, different look: separate
    assert new and b != a
    c, new = s.ingest(det("cup", 0.05, 0, emb=[0.99, 0.1, 0]), 3)   # other label, same look: merged
    assert c == a and not new and set(s.objects[a].labels) == {"bottle", "cup"}


def test_prune():
    s = ObjectStore(Params(ttl_s=0, dynamic_ttl_s=10))
    s.ingest(det("person", 0, 0), 0)
    s.ingest(det("chair", 3, 0), 0)
    assert s.prune(100) and [o.label for o in s.objects.values()] == ["chair"]


def test_bev(out=None):
    s = ObjectStore(Params(min_observations=1))
    for i, (lab, x, y) in enumerate([("chair", 2, 1), ("table", 3, 2.5), ("person", -1, 4), ("fire extinguisher", 6, -2)]):
        s.ingest(det(lab, x, y), 100 + i)
    objs = s.to_dict(104, "map")["objects"]
    img = bev.render(objs, None, (0.5, 0.5, 0.6))
    assert img.ndim == 3 and img.shape[0] > 100
    data = np.full((200, 300), -1, np.int8)   # 15 x 10 m map at 5 cm, origin (-5, -3): free room with walls
    data[20:180, 20:280] = 0
    data[20:22, 20:280] = data[178:180, 20:280] = data[20:180, 20:22] = data[20:180, 278:280] = 100
    img2 = bev.render(objs, {"data": data, "resolution": 0.05, "origin": (-5.0, -3.0)}, (0.5, 0.5, 0.6))
    assert img2.shape[1] == 750 and img2.shape[0] == 500   # 15 m x 10 m at 50 px/m
    if out:
        import cv2
        cv2.imwrite(os.path.join(out, "bev_grid.png"), img)
        cv2.imwrite(os.path.join(out, "bev_map.png"), img2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(sys.argv[1]) if name == "test_bev" and len(sys.argv) > 1 else fn()
            print("ok", name)
