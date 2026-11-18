#!/usr/bin/env python3
# AGX ORIN, Odin-perception venv. OUR helper (not upstream): keeps Florence-2 LOADED (their API: make_backend +
# InspectionPipeline.run_items) and answers on http://127.0.0.1:8740 so Foxglove can ask it something in ~1-3 s
# instead of reloading the model (~10 s) per check run.
#
# Commands (GET /run?q=...), sent from Foxglove on /florence/command:
#   chair                 -> NUMBERED detection: every chair gets a name and number ("chair 1", "chair 2" ...),
#                            drawn with its own colour on /inspection/annotated/compressed. Numbering: closest row
#                            first (bottom of the image), left to right inside each row. Numbers stay valid
#                            until the next search.
#   far chair             -> FAR search (~12 s): whole image + 6 overlapping tiles at full resolution; adds only the
#                            small objects the whole image missed; for classes YOLO knows, YOLOv5l (GPU service :8750)
#                            must confirm each one; others are marked "?" (unverified)
#   describe 3            -> describes object #3 of the last search (Florence-2 caption of that crop)
#   show                  -> publishes the last numbered image again and repeats the list
#   check chair           -> the old checklist answer: "Is there a chair" + "How many chair"
#   all                   -> ~/perception_test/checklist.yaml (full checklist; env FLORENCE_CHECKLIST)
#   /path/list.yaml       -> that checklist
# GET /last -> JSON of the last numbered search (boxes, frame stamp): florence_cmd_relay.py adds the lidar distance
#              and publishes /florence/objects_3d -> yolo_cmd_bridge (Go2) saves them as F1, F2 ... ("go F1").
# Testing without the camera: /run?q=chair&img=/path/photo.jpg
import json, math, os, sys, tempfile, threading, time, traceback, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.expanduser("~/odin-perception")
sys.path.insert(0, REPO)
os.chdir(REPO)
from inspection_perception.cli import make_frame_source, make_backend, InspectionPipeline, load_check_items  # noqa
from inspection_perception.annotation import _load_font, encode_jpeg  # noqa

DEFAULT_LIST = os.environ.get("FLORENCE_CHECKLIST", os.path.expanduser("~/perception_test/checklist.yaml"))
EVIDENCE = os.path.expanduser("~/perception_test/evidence")
ANNOTATE_URL = "http://127.0.0.1:8730"
YOLO_URL = "http://127.0.0.1:8750"   # our YOLOv5l GPU service (start_yolo_gpu_agx.sh)
FOV_DEG = 90.0            # fallback only: undistorted camera, about 90 deg wide


def _intrinsics():
    """This Odin1's pinhole intrinsics (same as the undistorted image and the lidar depth image)."""
    import re
    try:
        txt = open(os.path.expanduser("~/odin_test/yolo/calib.yaml")).read()
        g = lambda k: float(re.search(r"^\s*%s:\s*([-+0-9.eE]+)" % k, txt, re.M).group(1))
        return {"fx": g("A11"), "cx": g("u0"), "w": g("image_width")}
    except Exception:
        return None


CAM = _intrinsics()
LOCK = threading.Lock()
LAST = {}                 # last numbered search: phrase, rgb, objects, time, stamp_ns, seq
LAST_SEQ = [0]
COLORS = [(255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0), (175, 82, 222), (255, 204, 0),
          (90, 200, 250), (255, 45, 85), (162, 132, 94), (0, 199, 190), (88, 86, 214), (142, 142, 147)]

print("Loading Florence-2 on the GPU ...", flush=True)
t0 = time.time()
SOURCE = make_frame_source(backend="http", prefer="compressed", http_url=ANNOTATE_URL, file_path=None)
BACKEND = make_backend("florence2", model_dir="models/florence2-base-ft", dtype="float32", device="cuda")
PIPE = InspectionPipeline(source=SOURCE, backend=BACKEND, evidence_dir=EVIDENCE, annotate_url=ANNOTATE_URL)
print("Florence-2 ready in %.1f s, listening on 127.0.0.1:8740" % (time.time() - t0), flush=True)


# ---------------------------------------------------------------- numbered detection
def grab_rgb(img_path=None):
    """Returns (rgb, stamp_ns of the camera image; 0 for a file)."""
    if img_path:
        return np.array(Image.open(img_path).convert("RGB")), 0
    fr = SOURCE.get_latest(timeout_s=3.0, max_age_ms=1500.0)
    if fr is None:
        raise RuntimeError("no camera frame (is the Odin1 running? start_odin_agx.sh + start_perception_agx.sh)")
    return fr.as_rgb(), int(fr.stamp_ns or 0)


def detect_all(rgb, phrase):
    """Florence-2 open-vocabulary detection with room for many objects (their detect() stops at ~18 boxes)."""
    task = "<OPEN_VOCABULARY_DETECTION>"
    out = BACKEND._run_task(rgb, task, text_input=phrase, max_new_tokens=1024).get(task, {}) or {}
    labels = out.get("bboxes_labels") or out.get("labels") or []
    H, W = rgb.shape[:2]
    boxes = []
    for i, b in enumerate(out.get("bboxes") or []):
        if b is None or len(b) < 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in b[:4]]
        x0, x1 = max(0.0, min(x0, x1)), min(W - 1.0, max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(H - 1.0, max(y0, y1))
        if (x1 - x0) < 6 or (y1 - y0) < 6:
            continue
        boxes.append({"label": str(labels[i]) if i < len(labels) and labels[i] else phrase,
                      "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return clean_boxes(boxes, W, H)


def _area(b):
    return max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])


def _inter(a, b):
    w = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    h = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    return max(0.0, w) * max(0.0, h)


def clean_boxes(boxes, W, H):
    # drop boxes covering most of the image (Florence sometimes boxes "the whole scene")
    boxes = [b for b in boxes if _area(b) < 0.6 * W * H]
    # drop "group" boxes that contain 2+ other boxes (e.g. one box around a row of chairs)
    keep = []
    for b in boxes:
        inside = sum(1 for o in boxes if o is not b and _area(o) < _area(b) and _inter(o, b) > 0.8 * _area(o))
        if inside < 2:
            keep.append(b)
    # duplicates: same object boxed twice -> keep the bigger one
    keep.sort(key=_area, reverse=True)
    out = []
    for b in keep:
        if all(_inter(b, o) < 0.6 * min(_area(b), _area(o)) for o in out):
            out.append(b)
    return out


def number_boxes(boxes, W):
    """Closest row first (bottom of the image), then left to right. Adds n, name, direction."""
    if not boxes:
        return []
    hs = sorted(b["y1"] - b["y0"] for b in boxes)
    tol = 0.35 * hs[len(hs) // 2]
    rows = []
    for b in sorted(boxes, key=lambda b: -b["y1"]):
        for r in rows:
            if abs(r[0]["y1"] - b["y1"]) <= tol:
                r.append(b); break
        else:
            rows.append([b])
    ordered = [b for r in rows for b in sorted(r, key=lambda b: b["x0"] + b["x1"])]
    counts = {}
    for n, b in enumerate(ordered, 1):
        b["n"] = n
        base = b["label"].strip() or "object"
        counts[base] = counts.get(base, 0) + 1
        b["name"] = "%s %d" % (base, n)
        cxb = (b["x0"] + b["x1"]) / 2.0
        if CAM and abs(CAM["w"] - W) < 2:      # real calibration (pinhole): same angle YOLO reports
            ang = math.degrees(math.atan((cxb - CAM["cx"]) / CAM["fx"]))
        else:
            ang = (cxb / W - 0.5) * FOV_DEG
        b["angle_deg"] = round(ang, 1)
        b["direction"] = "ahead" if abs(ang) < 8 else "%d deg %s" % (round(abs(ang)), "right" if ang > 0 else "left")
        b["row"] = next(i for i, r in enumerate(rows, 1) if b in r)
    return ordered


def render(rgb, objs, title):
    img = Image.fromarray(rgb).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    fs = max(14, int(W / 38))
    font, small = _load_font(fs), _load_font(max(12, int(fs * 0.8)))
    for o in objs:
        c = COLORS[(o["n"] - 1) % len(COLORS)]
        d.rectangle([o["x0"], o["y0"], o["x1"], o["y1"]], outline=c, width=max(3, W // 300))
        tag = o["name"]
        tw, th = d.textbbox((0, 0), tag, font=font)[2:]
        tx, ty = o["x0"], o["y0"] - th - 6
        if ty < 0:
            ty = o["y0"] + 2
        d.rectangle([tx, ty, tx + tw + 8, ty + th + 6], fill=c)
        d.text((tx + 4, ty + 2), tag, fill=(0, 0, 0) if sum(c) > 400 else (255, 255, 255), font=font)
    tw, th = d.textbbox((0, 0), title, font=small)[2:]
    d.rectangle([0, 0, tw + 16, th + 12], fill=(0, 0, 0))
    d.text((8, 5), title, fill=(255, 255, 255), font=small)
    return img


def publish(img):
    try:
        req = urllib.request.Request(ANNOTATE_URL + "/annotate", data=encode_jpeg(img),
                                     headers={"Content-Type": "image/jpeg"}, method="POST")
        urllib.request.urlopen(req, timeout=2.0).read(0)
        return True
    except Exception:
        return False


def list_text(phrase, objs, secs=None):
    if not objs:
        return 'No "%s" found in the image.' % phrase
    lines = ['Found %d "%s" (numbers: closest row first, left to right):' % (len(objs), phrase)]
    for o in objs:
        w = o["x1"] - o["x0"]
        lines.append("  #%-2d %-18s %-14s row %d, box %d px wide" % (o["n"], o["name"], o["direction"], o["row"], w))
    lines.append("Next: describe N | show | another word to search again" + (" (%.1f s)" % secs if secs else ""))
    return "\n".join(lines)


def find(phrase, img_path=None):
    t = time.time()
    with LOCK:
        rgb, stamp_ns = grab_rgb(img_path)
        objs = number_boxes(detect_all(rgb, phrase), rgb.shape[1])
    title = '%s: %d found' % (phrase, len(objs))
    img = render(rgb, objs, title)
    shown = publish(img)
    LAST.clear(); LAST.update(phrase=phrase, rgb=rgb, objects=objs, time=time.time(), title=title, stamp_ns=stamp_ns,
                              seq=LAST_SEQ.__setitem__(0, LAST_SEQ[0] + 1) or LAST_SEQ[0])
    try:
        img.save(os.path.join(os.path.expanduser("~/perception_test"), "last_numbered.jpg"), quality=85)
    except Exception:
        pass
    txt = list_text(phrase, objs, time.time() - t)
    return txt if shown else txt + "\n(annotated image not published: frame server :8730 not running)"


def _inter(a, b):
    w = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    h = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    return max(0.0, w) * max(0.0, h)


def _yolo_names():
    try:
        return {v.lower(): int(k) for k, v in json.loads(urllib.request.urlopen(YOLO_URL + "/names", timeout=3).read()).items()}
    except Exception:
        return {}


def _yolo_confirms(rgb, b, cls_id):
    """YOLOv5l (GPU service) on crops around box b: does it see class cls_id overlapping b? None = service down.
    YOLO needs context: on a tight crop it misses a chair it sees at 0.89 with more room around it, so it tries a
    crop with 1x and then 2x the object size as margin before saying no."""
    import cv2
    H, W = rgb.shape[:2]
    bw, bh = b["x1"] - b["x0"], b["y1"] - b["y0"]
    for frac in (1.0, 2.0):
        m = max(frac * max(bw, bh), 40)
        x0, y0 = int(max(0, b["x0"] - m)), int(max(0, b["y0"] - m))
        x1, y1 = int(min(W, b["x1"] + m)), int(min(H, b["y1"] + m))
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1, ::-1])           # BGR, as their detector sends it
        ch, cw = crop.shape[:2]
        r = 640.0 / max(ch, cw)
        nw, nh = int(round(cw * r)), int(round(ch * r))
        lb = cv2.resize(crop, (nw, nh))
        pw, ph = (32 - nw % 32) % 32, (32 - nh % 32) % 32
        lb = cv2.copyMakeBorder(lb, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        try:
            req = urllib.request.Request(YOLO_URL + "/infer?conf=0.20", data=lb.tobytes(),
                                         headers={"X-Shape": "%d,%d" % lb.shape[:2]}, method="POST")
            rows = json.loads(urllib.request.urlopen(req, timeout=10).read())
        except Exception:
            return None
        tb = {"x0": (b["x0"] - x0) * r, "y0": (b["y0"] - y0) * r, "x1": (b["x1"] - x0) * r, "y1": (b["y1"] - y0) * r}
        for x_0, y_0, x_1, y_1, conf, k in rows:
            if int(k) == cls_id:
                yb = {"x0": x_0, "y0": y_0, "x1": x_1, "y1": y_1}
                if _inter(tb, yb) > 0.3 * min(_area(tb), _area(yb)):
                    return True
        if x0 == 0 and y0 == 0 and x1 == W and y1 == H:
            break
    return False


def find_far(phrase, img_path=None):
    t = time.time()
    with LOCK:
        rgb, stamp_ns = grab_rgb(img_path)
        H, W = rgb.shape[:2]
        full = detect_all(rgb, phrase)
        for b in full:
            b["far"], b["verified"] = False, True
        cols, rows_, ov = 3, 2, 0.20
        tw, th = int(W / (cols - (cols - 1) * ov)), int(H / (rows_ - (rows_ - 1) * ov))
        tiles = []
        for r in range(rows_):
            for c in range(cols):
                x0, y0 = int(c * tw * (1 - ov)), int(r * th * (1 - ov))
                x1, y1 = min(W, x0 + tw), min(H, y0 + th)
                for b in detect_all(np.ascontiguousarray(rgb[y0:y1, x0:x1]), phrase):
                    tiles.append({"label": b["label"], "x0": b["x0"] + x0, "y0": b["y0"] + y0,
                                  "x1": b["x1"] + x0, "y1": b["y1"] + y0})
        # same object seen in two overlapping tiles -> keep the bigger box
        tiles.sort(key=_area, reverse=True)
        merged = []
        for b in tiles:
            if all(_inter(b, m) < 0.5 * min(_area(b), _area(m)) for m in merged):
                merged.append(b)
        # only what the whole image did NOT see (near/big objects come from the whole image, never as tile pieces)
        new = [b for b in merged if all(_inter(b, f) < 0.3 * min(_area(b), _area(f)) for f in full)]
        ynames = _yolo_names()
        cls_id = ynames.get(phrase.strip().lower())
        kept, dropped, unverified = [], 0, 0
        for b in new:
            b["far"] = True
            ok = _yolo_confirms(rgb, b, cls_id) if cls_id is not None else None
            if ok is False:
                dropped += 1
                continue
            b["verified"] = bool(ok)
            unverified += 0 if ok else 1
            kept.append(b)
        objs = number_boxes(full + kept, W)
    for o in objs:
        if not o.get("verified", True):
            o["name"] += "?"
    title = "far %s: %d found (%d far)" % (phrase, len(objs), len(kept))
    img = render(rgb, objs, title)
    shown = publish(img)
    LAST.clear(); LAST.update(phrase=phrase, rgb=rgb, objects=objs, time=time.time(), title=title, stamp_ns=stamp_ns,
                              seq=LAST_SEQ.__setitem__(0, LAST_SEQ[0] + 1) or LAST_SEQ[0])
    try:
        img.save(os.path.join(os.path.expanduser("~/perception_test"), "last_numbered.jpg"), quality=85)
    except Exception:
        pass
    txt = list_text(phrase, objs, time.time() - t)
    if cls_id is not None:
        note = "Far search: whole image + 6 tiles -> %d extra far object(s); YOLOv5l confirmed them" % len(kept)
        note += ", %d discarded (YOLO did not see a %s there)." % (dropped, phrase) if dropped else "."
        if unverified:
            note += " %d could not be checked (YOLO service down): marked '?'." % unverified
    else:
        note = ("Far search: whole image + 6 tiles -> %d extra far object(s). YOLO does not know '%s', so they are NOT "
                "verified: marked '?' - check the image before 'go F'." % (len(kept), phrase)) if kept else \
               "Far search: whole image + 6 tiles -> nothing new far away."
    txt += "\n" + note
    return txt if shown else txt + "\n(annotated image not published: frame server :8730 not running)"


def describe(n):
    o = next((x for x in LAST.get("objects", []) if x["n"] == n), None)
    if o is None:
        return "No object #%d in the last search. Search first (e.g. chair)." % n
    rgb = LAST["rgb"]
    H, W = rgb.shape[:2]
    mx, my = 0.15 * (o["x1"] - o["x0"]), 0.15 * (o["y1"] - o["y0"])
    crop = rgb[int(max(0, o["y0"] - my)):int(min(H, o["y1"] + my)), int(max(0, o["x0"] - mx)):int(min(W, o["x1"] + mx))]
    with LOCK:
        cap = BACKEND.caption(np.ascontiguousarray(crop), length="detailed")
    publish(render(rgb, [o], "%s (%s)" % (o["name"], o["direction"])))
    age = time.time() - LAST["time"]
    return "#%d %s, %s, row %d:\n  %s\n(image taken %.0f s ago)" % (o["n"], o["name"], o["direction"], o["row"], cap, age)


def show():
    if not LAST.get("objects") and not LAST.get("phrase"):
        return "Nothing searched yet. Send a word, e.g. chair."
    publish(render(LAST["rgb"], LAST["objects"], LAST["title"]))
    return list_text(LAST["phrase"], LAST["objects"]) + "\n(image taken %.0f s ago)" % (time.time() - LAST["time"])


# ---------------------------------------------------------------- old checklist mode
def phrase_list(q):
    q = q.strip()
    return ("version: 1\ndefaults: {severity: normal}\nitems:\n"
            f"  - id: present\n    name: Is there a {q}\n    kind: yes_no\n    question: Is there a {q}\n    expect: \"yes\"\n"
            f"    rule: {{op: presence, target: {json.dumps(q)}, mode: ovd, max_area_frac: 0.9, min_count: 1}}\n"
            f"  - id: count\n    name: How many {q}\n    kind: count\n    question: How many {q}\n"
            f"    rule: {{op: count, target: {json.dumps(q)}, mode: ovd, max_area_frac: 0.9}}\n")


def checklist(q):
    if q.lower() == "all":
        path, tmp = DEFAULT_LIST, None
    elif q.endswith((".yaml", ".yml", ".json")):
        path, tmp = os.path.expanduser(q), None
    else:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(phrase_list(q)); tmp.close(); path = tmp.name
    try:
        items = load_check_items(path)
        names = {getattr(i, "id", None) or getattr(i, "item_id", None): getattr(i, "name", "") for i in items}
        t = time.time()
        with LOCK:
            results = PIPE.run_items(items, share_frame=True)
        lines = []
        for r in results:
            rid = getattr(r, "item_id", "")
            st = getattr(r, "status", "")
            st = str(getattr(st, "value", st)).split(".")[-1].upper()
            lines.append(f"[{st:4}] {names.get(rid) or rid}: {getattr(r, 'answer', '')}  ({getattr(r, 'latency_ms', 0):.0f} ms)")
        src = os.path.basename(path) if not tmp else f'phrase "{q}"'
        lines.append(f"{len(results)} checks in {time.time() - t:.1f} s ({src})")
        return "\n".join(lines)
    finally:
        if tmp:
            os.unlink(tmp.name)


def run(q, img_path=None):
    q = (q or "").strip()
    low = q.lower()
    parts = low.split()
    if low in ("", "help"):
        return ("Send an English word to find and number objects (chair, table, person, cup ...).\n"
                "Then: far <word> (looks farther, ~12 s) | describe N | show | check <word> | all")
    if low == "all" or low.endswith((".yaml", ".yml", ".json")):
        return checklist(q)
    if parts[0] == "far" and len(parts) > 1:
        return find_far(q[4:].strip(), img_path)
    if parts[0] == "check" and len(parts) > 1:
        return checklist(q[6:].strip())
    if len(parts) == 2 and parts[0] in ("describe", "desc") and parts[1].lstrip("#").isdigit():
        return describe(int(parts[1].lstrip("#")))
    if low in ("show", "list", "last"):
        return show()
    return find(q, img_path)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, text, ctype="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        if u.path == "/last":
            objs = [{k: v for k, v in o.items()} for o in LAST.get("objects", [])]
            shp = LAST["rgb"].shape if "rgb" in LAST else (0, 0)
            return self._send(200, json.dumps({"phrase": LAST.get("phrase"), "time": LAST.get("time"),
                                               "stamp_ns": LAST.get("stamp_ns", 0), "seq": LAST.get("seq", 0),
                                               "width": shp[1], "height": shp[0], "objects": objs}, indent=1),
                              "application/json")
        if u.path != "/run":
            return self._send(404, "not found")
        q = qs.get("q", [""])[0]
        try:
            text, code = run(q, qs.get("img", [None])[0]), 200
        except Exception as e:
            text, code = "ERROR: %s\n%s" % (e, traceback.format_exc(limit=2)), 500
        self._send(code, text)
        print("q=%r\n%s" % (q, text), flush=True)


ThreadingHTTPServer(("127.0.0.1", 8740), H).serve_forever()
