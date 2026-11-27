#!/usr/bin/env python3
"""Object store of the semantic map: detections in the map frame -> persistent objects with metadata.
Pure Python + numpy (Python 3.8), no ROS: semantic_map_node.py feeds it, test_object_store.py tests it.

Adapted from go2-semantic-nav (https://github.com/yusufdxb/go2-semantic-nav, MIT License, Copyright (c) 2026 Yusuf
Guenena), ros2_ws/src/go2_scene_graph/go2_scene_graph/graph_store.py. Changes for our stack:
  - the embedding is optional (our detectors give none yet): without it, objects match by label + distance
  - a tracker's track ID is used first (same source + track ID seen recently = same object)
  - positions are a weighted running mean (score-weighted, capped so the object can still move), not a fixed EMA
  - integer IDs, sources, last box / track ID / distance kept as metadata; JSON output
  - "dynamic" classes (person, ...) expire after a short time; others stay (ttl 0 = forever)
  - all times are message stamps (seconds), never the wall clock, so bags and live data behave the same
"""
import math

import numpy as np


class Params:
    def __init__(self, **kw):
        self.assoc_dist_m = 0.6         # max distance between a detection and an object to merge them
        self.assoc_embed_min = 0.75     # min cosine similarity when both have an embedding
        self.track_memory_s = 3.0       # a (source, track ID) seen within this time keeps pointing at its object
        self.track_max_jump_m = 1.5     # ...unless the new position is farther than this from the object
        self.max_weight = 20.0          # cap of the accumulated weight: the position keeps following a moved object
        self.embed_alpha = 0.2          # running update of the embedding
        self.min_observations = 3       # objects seen fewer times are kept but not published
        self.ttl_s = 0.0                # forget objects not seen for this long (0 = never)
        self.dynamic_classes = ("person", "dog", "cat")
        self.dynamic_ttl_s = 10.0       # ...and dynamic classes after this long
        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError("unknown parameter %s" % k)
            setattr(self, k, v)


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


class SemanticObject:
    def __init__(self, oid, det, stamp):
        self.id = oid
        self.position = np.asarray(det["position"], np.float64)   # map frame, m
        self.weight = max(det["score"], 1e-3)
        self.size = list(det["size"]) if det.get("size") else None  # [w, h] m, from the box at the object's depth
        self.labels = {det["label"]: det["score"]}                   # label -> summed score (votes)
        self.label = det["label"]
        self.confidence = det["score"]
        self.embedding = _unit(det["embedding"]) if det.get("embedding") is not None else None
        self.observations = 1
        self.first_seen = self.last_seen = stamp
        self.sources = {det["source"]}
        self.last = {"source": det["source"], "track_id": det.get("track_id"), "box": det.get("box"),
                     "distance": det.get("distance"), "score": det["score"]}

    def update(self, det, stamp, p):
        w = max(det["score"], 1e-3)
        self.position = (self.position * self.weight + np.asarray(det["position"]) * w) / (self.weight + w)
        self.weight = min(self.weight + w, p.max_weight)
        if det.get("size"):
            self.size = det["size"] if self.size is None else [0.8 * a + 0.2 * b for a, b in zip(self.size, det["size"])]
        self.labels[det["label"]] = self.labels.get(det["label"], 0.0) + det["score"]
        self.label = max(self.labels.items(), key=lambda kv: kv[1])[0]
        self.confidence = 0.7 * self.confidence + 0.3 * det["score"]
        e = _unit(det["embedding"]) if det.get("embedding") is not None else None
        if e is not None:
            self.embedding = e if self.embedding is None else _unit((1 - p.embed_alpha) * self.embedding + p.embed_alpha * e)
        self.observations += 1
        self.last_seen = max(self.last_seen, stamp)
        self.sources.add(det["source"])
        self.last = {"source": det["source"], "track_id": det.get("track_id"), "box": det.get("box"),
                     "distance": det.get("distance"), "score": det["score"]}

    def to_dict(self, with_embedding=False):
        d = {"id": self.id, "label": self.label, "labels": {k: round(v, 3) for k, v in self.labels.items()},
             "confidence": round(self.confidence, 3), "position": [round(float(v), 3) for v in self.position],
             "size": [round(v, 3) for v in self.size] if self.size else None, "observations": self.observations,
             "first_seen": self.first_seen, "last_seen": self.last_seen, "sources": sorted(self.sources),
             "last": self.last, "has_embedding": self.embedding is not None}
        if with_embedding and self.embedding is not None:
            d["embedding"] = [round(float(v), 5) for v in self.embedding]
        return d


class ObjectStore:
    def __init__(self, params=None):
        self.p = params or Params()
        self.objects = {}        # id -> SemanticObject
        self.tracks = {}         # (source, track_id) -> (object id, last stamp)
        self.next_id = 1

    def ingest(self, det, stamp):
        """det: {"label", "score", "position" [x, y, z] (map), "source", optional "size", "track_id", "box",
        "distance", "embedding"}. Returns (object id, is_new)."""
        det = dict(det, label=str(det["label"]).strip().lower())
        oid = self._by_track(det, stamp) or self._nearest(det)
        if oid is None:
            oid, self.next_id = self.next_id, self.next_id + 1
            self.objects[oid] = SemanticObject(oid, det, stamp)
            new = True
        else:
            self.objects[oid].update(det, stamp, self.p)
            new = False
        if det.get("track_id") is not None:
            self.tracks[(det["source"], det["track_id"])] = (oid, stamp)
        return oid, new

    def _by_track(self, det, stamp):
        if det.get("track_id") is None:
            return None
        hit = self.tracks.get((det["source"], det["track_id"]))
        if hit is None or hit[0] not in self.objects or stamp - hit[1] > self.p.track_memory_s:
            return None
        o = self.objects[hit[0]]
        return o.id if np.linalg.norm(o.position - det["position"]) <= self.p.track_max_jump_m else None

    def _nearest(self, det):
        """Closest object within assoc_dist_m that agrees: same embedding (when both have one), else same label."""
        e = _unit(det["embedding"]) if det.get("embedding") is not None else None
        best, best_d = None, math.inf
        for o in self.objects.values():
            d = float(np.linalg.norm(o.position - det["position"]))
            if d > self.p.assoc_dist_m or d >= best_d:
                continue
            if e is not None and o.embedding is not None:
                if float(np.dot(e, o.embedding)) < self.p.assoc_embed_min:
                    continue
            elif det["label"] not in o.labels:
                continue
            best, best_d = o.id, d
        return best

    def prune(self, now):
        """Forget objects not seen for too long (times are message stamps). Returns the removed ids."""
        gone = []
        for oid, o in list(self.objects.items()):
            ttl = self.p.dynamic_ttl_s if o.label in self.p.dynamic_classes else self.p.ttl_s
            if ttl > 0 and now - o.last_seen > ttl:
                del self.objects[oid]
                gone.append(oid)
        self.tracks = {k: v for k, v in self.tracks.items() if v[0] in self.objects}
        return gone

    def published(self):
        return [o for o in self.objects.values() if o.observations >= self.p.min_observations]

    def clear(self):
        self.objects.clear()
        self.tracks.clear()

    def to_dict(self, stamp, frame_id, with_embeddings=False):
        objs = sorted(self.published(), key=lambda o: o.id)
        return {"stamp": stamp, "frame_id": frame_id, "count": len(objs), "stored": len(self.objects),
                "objects": [o.to_dict(with_embeddings) for o in objs], "edges": []}
