#!/usr/bin/env python3
"""GPU detector service: a light per-frame detector + an optional tracker, behind one HTTP API on 127.0.0.1.
Runs in the Odin-perception venv (Python 3.10, CUDA) in the container odin-gpu or on the host; the ROS side
(detector_node.py, Python 3.8, container odin-agx) sends it frames. Started by start_yolo_gpu_agx.sh (process yolosvc).

Selection (environment variables, normally set by start_yolo_gpu_agx.sh from ~/.config/go2_agx/detector.env):
  DETECTOR          yolov5 (default) | rfdetr | yoloe (open vocabulary)  see detectors.py
  YOLOE_CLASSES     yoloe only: initial classes, comma separated (default: the 80 COCO classes)
  DETECTOR_WEIGHTS  path, or comma-separated fallbacks              default: detectors.DEFAULT_WEIGHTS
  TRACKER           none (default) | iou | botsort | botsort_reid | sam2      see trackers.py
  DETECTOR_PORT     8750
  HISTORY_SEC       5      recent frames kept to place a target box given with an older time stamp (e.g. Florence-2)

API. Images are raw BGR uint8 bytes with the header X-Shape: "h,w"; boxes are [x1, y1, x2, y2] in pixels of the image sent.
  GET  /health   "ok <detector> | <tracker> | <ms> ms/img (<n> images)"
  GET  /info     {"detector", "tracker", "prompted", "open_vocab", "names_version", "input_size", "names": {"0": "person", ...}}
  GET  /names    {"0": "person", ...}   (class id -> name, as the upstream /yolo_class_names)
  POST /detect?conf=0.25[&stamp=<s>][&track=0][&masks=1][&scale=<sent image / original>]
       -> {"ms", "names_version", "target": {"id", "visible"} | null,
           "objects": [{"box", "score", "cls", "name", "track_id" (null = not tracked), "target", "source",
                        "mask_index" (null = no mask)}],
           "mask_png": base64 PNG, only with masks=1}
       source "detector" for detections (score >= conf, or the target), "sam2" for the object SAM 2 follows.
       track=0: detections only, the tracker state is not touched (one-off images such as Florence-2 crops).
       masks=1: one 8-bit label image of the image sent, pixel = mask_index of the object there, 0 = none (SAM 2's
       object today; a segmentation detector would add its masks the same way).
  POST /target   JSON, one of:
       {"box": [x1, y1, x2, y2], "stamp": <s>[, "cls": <id>]}  the track overlapping the box in the frame closest to
                                                               stamp (prompted tracker: starts on that box and frame)
           box in pixels of the image sent, or with "box_scale": "full" in pixels of the original image (converted
           with the scale that frame was sent with)
       {"name": "person"}                                      the most confident current track of that class
       {"track_id": <id>}  |  {"clear": true}
       -> {"ok", "target", "detail"}
  POST /classes  JSON {"names": ["chair", "fire extinguisher"]} (replace), {"add": [...]}, {"remove": [...]}
       open-vocabulary detectors only (yoloe). Class ids change, so the tracker and the target are reset.
       -> {"ok", "names", "names_version", "detail"}
"""
import base64
import collections
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import make_detector  # noqa: E402
from trackers import iou_matrix, make_tracker  # noqa: E402


class Pipeline:
    def __init__(self, detector, tracker, weights=None, history_sec=5.0):
        t0 = time.time()
        self.det = make_detector(detector, weights, log=lambda m: print(m, flush=True))
        self.trk = make_tracker(tracker)
        self.lock = threading.Lock()          # one GPU, one tracker state: requests run one at a time
        self.history = collections.deque()    # {"stamp", "objects" (tracked ones), "img" (prompted trackers only)}
        self.history_sec = history_sec
        self.target = None                    # {"id", "visible"}
        self.names_version = 0                # bumped when an open-vocabulary detector gets new classes
        self.stats = {"n": 0, "ms": 0.0}
        print("Detector service: %s | tracker %s | loaded in %.1f s" % (self.det.desc, self.trk.desc, time.time() - t0),
              flush=True)

    def info(self):
        return {"detector": self.det.desc, "tracker": self.trk.name, "prompted": self.trk.prompted,
                "open_vocab": self.det.open_vocab, "names_version": self.names_version,
                "input_size": self.det.input_size, "names": {str(k): v for k, v in self.det.names.items()}}

    def set_classes(self, req):
        if not self.det.open_vocab:
            return {"ok": False, "names": list(self.det.names.values()), "names_version": self.names_version,
                    "detail": "%s has fixed classes (use DETECTOR=yoloe for text classes)" % self.det.desc}
        cur = list(self.det.names.values())
        if "names" in req:
            new = list(req["names"])
        else:
            new = [n for n in cur if n not in set(req.get("remove", []))] + [n for n in req.get("add", []) if n not in cur]
        t = time.time()
        with self.lock:
            names = self.det.set_classes(new)
            self.trk.reset()                  # class ids changed meaning: old tracks and the target are void
            self.target, self.history = None, collections.deque()
            self.names_version += 1
        names = list(names.values())
        return {"ok": True, "names": names, "names_version": self.names_version,
                "detail": "now looking for: %s (%.1f s)" % (", ".join(names), time.time() - t)}

    def health(self):
        return "ok %s | %s | %.0f ms/img (%d images)" % (self.det.desc, self.trk.desc, self.stats["ms"], self.stats["n"])

    def _name(self, c):
        return self.det.names.get(int(c), "cls%d" % int(c)) if int(c) >= 0 else "object"

    def _obj(self, box, score, cls, tid, source):
        return {"box": [round(float(v), 1) for v in box], "score": round(float(score), 3), "cls": int(cls),
                "name": self._name(cls), "track_id": None if tid < 0 else int(tid),
                "target": bool(self.target and tid >= 0 and tid == self.target["id"]), "source": source}

    # ------------------------------------------------------------------ per frame
    def detect(self, img, conf, stamp=None, track=True, masks=False, scale=1.0):
        t = time.time()
        track = track and self.trk.name != "none"
        with self.lock:
            dconf = min(conf, self.trk.det_conf) if (track and self.trk.det_conf) else conf
            boxes, scores, cls, dmasks = self.det.detect(img, dconf)
            if dmasks is None:
                dmasks = [None] * len(boxes)
            ids, extra = self.trk.update(img, boxes, scores, cls) if track else (np.full(len(boxes), -1), [])
            if track and not self.trk.has_target:
                self.target = None            # SAM 2 dropped the object
            objs, mlist = [], []
            for b, s, c, i, m in zip(boxes, scores, cls, ids, dmasks):
                if s >= conf or (self.target and i == self.target["id"]):
                    objs.append(self._obj(b, s, c, i, "detector"))
                    mlist.append(m)
            for e in extra:
                objs.append(self._obj(e["box"], e["score"], e["cls"], e["id"], self.trk.name))
                mlist.append(e.get("mask"))
            if track:
                self._remember(stamp if stamp is not None else time.time(), objs, img, scale)
                if self.target is not None:
                    self.target["visible"] = any(o["target"] for o in objs)
            target = dict(self.target) if self.target else None
        res = {"names_version": self.names_version, "target": target, "objects": objs}
        for k, (o, m) in enumerate(zip(objs, mlist)):
            o["mask_index"] = k + 1 if m is not None and k < 255 else None
        if masks:   # the label image only when asked for
            label = np.zeros(img.shape[:2], np.uint8)
            for o, m in zip(objs, mlist):
                if o["mask_index"]:
                    label[np.asarray(m, bool)] = o["mask_index"]
            res["mask_png"] = base64.b64encode(cv2.imencode(".png", label)[1].tobytes()).decode()
        ms = (time.time() - t) * 1000
        self.stats["n"] += 1
        self.stats["ms"] = ms if self.stats["n"] == 1 else 0.9 * self.stats["ms"] + 0.1 * ms
        res["ms"] = round(ms, 1)
        return res

    def _remember(self, stamp, objs, img, scale):
        self.history.append({"stamp": stamp, "scale": scale, "objects": [o for o in objs if o["track_id"] is not None],
                             "img": img if self.trk.keeps_frames else None})
        while self.history and self.history[-1]["stamp"] - self.history[0]["stamp"] > self.history_sec:
            self.history.popleft()

    # ------------------------------------------------------------------ target selection
    def select(self, req):
        with self.lock:
            return self._select(req)

    def _set(self, tid, detail):
        self.target = {"id": int(tid), "visible": False}
        return {"ok": True, "target": int(tid), "detail": detail}

    def _select(self, req):
        fail = lambda detail: {"ok": False, "target": None, "detail": detail}
        if req.get("clear"):
            self.target = None
            if self.trk.prompted:
                self.trk.reset()
            return {"ok": True, "target": None, "detail": "target cleared"}
        if self.trk.name == "none":
            return fail("no tracker running (TRACKER=none)")
        if "box" not in req and self.trk.prompted:
            return fail("tracker %s needs a box" % self.trk.name)
        if "track_id" in req:
            return self._set(req["track_id"], "target = track %d" % int(req["track_id"]))
        if not self.history:
            return fail("no frames tracked yet")
        if "name" in req and "box" not in req:
            name = str(req["name"]).lower()
            cands = [o for o in self.history[-1]["objects"] if o["name"].lower() == name]
            if not cands:
                return fail("no %s tracked in the last frame" % name)
            best = max(cands, key=lambda o: o["score"])
            return self._set(best["track_id"], "target = track %d (%s %.2f)" % (best["track_id"], name, best["score"]))
        if "box" not in req:
            return fail("give box, name, track_id or clear")
        box, stamp = np.asarray(req["box"], np.float32), req.get("stamp")
        h = min(self.history, key=lambda e: abs(e["stamp"] - stamp)) if stamp else self.history[-1]
        dt = abs(h["stamp"] - stamp) if stamp else 0.0
        if req.get("box_scale") == "full":
            box = box * h["scale"]
        if self.trk.prompted:
            tid = self.trk.start(h["img"], box, int(req.get("cls", -1)))
            return self._set(tid, "%s started on the box (frame %.2f s from stamp)" % (self.trk.name, dt))
        if not h["objects"]:
            return fail("no tracks in the frame %.2f s from stamp" % dt)
        ious = iou_matrix(box[None], [o["box"] for o in h["objects"]])[0]
        if "cls" in req:   # prefer the same class when the caller knows it
            ious = ious * np.array([1.0 if o["cls"] == int(req["cls"]) else 0.5 for o in h["objects"]])
        k = int(np.argmax(ious))
        if ious[k] < 0.3:
            return fail("no track overlaps the box (best IoU %.2f, frame %.2f s from stamp)" % (ious[k], dt))
        tid = h["objects"][k]["track_id"]
        return self._set(tid, "target = track %d (IoU %.2f, frame %.2f s from stamp)" % (tid, ious[k], dt))


def serve(pipe, port):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            body = (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", "0")))

        def _image(self):   # read straight into a writable buffer (trackers may write into the image): no extra copy
            h, w = [int(v) for v in self.headers.get("X-Shape", "").split(",")]
            buf = bytearray(int(self.headers.get("Content-Length", "0")))
            view, got = memoryview(buf), 0
            while got < len(buf):
                n = self.rfile.readinto(view[got:])
                if not n:
                    raise ValueError("short image body")
                got += n
            return np.frombuffer(buf, np.uint8).reshape(h, w, 3)

        def do_GET(self):
            p = urllib.parse.urlparse(self.path).path
            if p == "/health":
                return self._send(200, pipe.health(), "text/plain")
            if p == "/info":
                return self._send(200, pipe.info())
            if p == "/names":
                return self._send(200, pipe.info()["names"])
            self._send(404, {})

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                if u.path == "/detect":
                    img = self._image()
                    stamp = float(q["stamp"]) if q.get("stamp") else None
                    return self._send(200, pipe.detect(img, float(q.get("conf", 0.25)), stamp, q.get("track", "1") != "0",
                                                       q.get("masks", "0") == "1", float(q.get("scale", 1.0))))
                if u.path == "/target":
                    return self._send(200, pipe.select(json.loads(self._body() or b"{}")))
                if u.path == "/classes":
                    return self._send(200, pipe.set_classes(json.loads(self._body() or b"{}")))
                self._send(404, {})
            except Exception as e:   # never kill the service on one bad request
                self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    print("listening on 127.0.0.1:%d" % port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    env = os.environ.get
    serve(Pipeline(env("DETECTOR", "yolov5"), env("TRACKER", "none"), env("DETECTOR_WEIGHTS") or None,
                   float(env("HISTORY_SEC", "5"))),
          int(env("DETECTOR_PORT", "8750")))
