#!/usr/bin/env python3
"""Trackers for the GPU detector service (detector_service.py), one interface for all:

    trk = make_tracker("none" | "iou" | "botsort" | "botsort_reid" | "sam2")
    ids, extra = trk.update(img_bgr, boxes, scores, cls)
        ids    track ID of each detection (N ints, -1 = not tracked)
        extra  objects the tracker follows that are not detections: [{"id", "box", "score", "cls", "mask"}] (sam2 only;
               mask = bool array of the image size)
    trk.reset()

Two kinds (trk.prompted):
  * detection trackers (iou, botsort, botsort_reid): give an ID to every detection and keep it across frames.
    The service picks a target among their tracks (by a box, e.g. from Florence-2, a class name or a track ID).
  * prompted trackers (sam2): follow ONE object given by a box (trk.start(img, box, cls)), whatever its class,
    even if the detector does not know it. update() returns it in extra; ids are all -1.

  none          no tracking (detections only)
  iou           greedy IoU association per class, no dependencies (default for tracking)
  botsort       BoT-SORT with camera-motion compensation (BoxMOT, AGPL-3.0: check before product use)
  botsort_reid  BoT-SORT + OSNet x0.25 appearance embeddings (TensorRT engine from export_reid.py)
  sam2          SAM 2.1 (tiny by default) on a box prompt. Its cost is set by SAM2_IMAGE_SIZE (default 512, the size SAM 2
                works at whatever image it gets; 384 or 256 run faster with coarser masks) and SAM2_MODEL (tiny | small). Uses the streaming "camera predictor" when the installed
                sam2 package has it (build_sam2_camera_predictor), otherwise the official image predictor with the
                previous box as prompt (no memory: may drift in crowds; the target is dropped when the mask is lost)

Ported from the detector benchmark (go2_rf_detr_bench branch: detbench/reid_track.py, sam2_track.py).
"""
import contextlib
import os

import cv2
import numpy as np

from detectors import MODELS, add_jetpack_path


def iou_matrix(a, b):
    """a: N x 4, b: M x 4 (x1, y1, x2, y2) -> N x M IoU."""
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-6)


class Tracker:
    name = "none"
    has_target = True     # prompted trackers: False once the object is lost (the service then drops the target)
    prompted = False     # True: follows one object given by start(); False: tracks every detection
    keeps_frames = False  # True: the service keeps recent frames so start() can use the frame of an old box
    det_conf = None       # detector threshold the tracker wants (BoT-SORT also uses low-score boxes); None = same

    @property
    def desc(self):
        return self.name

    def update(self, img, boxes, scores, cls):
        return np.full(len(boxes), -1, np.int64), []

    def reset(self):
        pass

    def start(self, img, box, cls=-1):
        raise NotImplementedError("tracker %s is not prompted" % self.name)


class IoUTracker(Tracker):
    """Greedy IoU association, same class only. Cheap and dependency-free; fine at 10 Hz for walking speeds."""
    name = "iou"

    def __init__(self, min_iou=0.3, max_lost=20, min_score=0.3):
        self.min_iou, self.max_lost, self.min_score = min_iou, max_lost, min_score
        self.reset()

    def reset(self):
        self.tracks, self.next_id = {}, 1   # id -> {"box", "cls", "lost"}

    def update(self, img, boxes, scores, cls):
        ids = list(self.tracks)
        tb = np.array([self.tracks[i]["box"] for i in ids], np.float32).reshape(-1, 4)
        m = iou_matrix(tb, boxes)
        if m.size:
            m[np.array([self.tracks[i]["cls"] for i in ids])[:, None] != np.asarray(cls)[None, :]] = 0
        used_t, used_d = set(), set()
        out = np.full(len(boxes), -1, np.int64)
        for flat in np.argsort(-m, axis=None):
            ti, di = np.unravel_index(flat, m.shape)
            if m[ti, di] < self.min_iou:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(int(ti))
            used_d.add(int(di))
            self.tracks[ids[ti]].update(box=boxes[di], lost=0)
            out[di] = ids[ti]
        for ti, i in enumerate(ids):   # unmatched old tracks: lost one more frame
            if ti not in used_t:
                self.tracks[i]["lost"] += 1
                if self.tracks[i]["lost"] > self.max_lost:
                    del self.tracks[i]
        for di in range(len(boxes)):   # unmatched confident detections: new tracks
            if di not in used_d and scores[di] >= self.min_score:
                self.tracks[self.next_id] = {"box": boxes[di], "cls": int(cls[di]), "lost": 0}
                out[di] = self.next_id
                self.next_id += 1
        return out, []


class BotSortTracker(Tracker):
    """BoT-SORT (BoxMOT 26). Detections down to det_conf go in (BoT-SORT's second association uses low-score boxes)."""
    name = "botsort"
    det_conf = 0.1

    def __init__(self, reid=False, lost_frames=40, reid_weights=None):
        from boxmot import BotSort
        from boxmot.trackers.botsort.config import BotSortConfig
        self.name = "botsort_reid" if reid else "botsort"
        self._make = lambda: BotSort(BotSortConfig(use_embeddings=reid, max_lost_frames=lost_frames),
                                     reid=self._reid_cfg(reid_weights) if reid else None)
        self.reset()

    @staticmethod
    def _reid_cfg(weights):
        from boxmot.reid.specs import ReIDConfig
        w = weights or os.environ.get("REID_WEIGHTS") or os.path.join(MODELS, "osnet_x0_25_msmt17_fp16.engine")
        if w.endswith(".engine"):
            if not os.path.exists(w):
                raise RuntimeError("Re-ID engine %s missing: build it with detector/export_reid.py" % w)
            # JetPack's TensorRT is not on PyPI: skip BoxMOT's pip check for 'nvidia-tensorrt'
            add_jetpack_path()
            import boxmot.reid.backends.tensorrt_backend as trt_backend
            trt_backend.require_reid_backend_requirements = lambda *a, **k: ()
            return ReIDConfig(os.path.abspath(w), device="cuda", precision="fp32")   # engine I/O is fp32 (fp16 inside)
        return ReIDConfig(w, device="cuda", precision="fp16")   # BoxMOT model name/.pt (downloaded on first use)

    def reset(self):
        self.trk = self._make()

    def update(self, img, boxes, scores, cls):
        dets = np.concatenate([np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32).reshape(-1, 1),
                               np.asarray(cls, np.float32).reshape(-1, 1)], 1)
        res = np.asarray(self.trk.update(dets, img))
        res = res.reshape(-1, res.shape[1]) if res.ndim == 2 else np.zeros((0, 8))
        ids = np.full(len(dets), -1, np.int64)
        for r in res:   # rows: x1, y1, x2, y2, track id, conf, cls, index of the detection
            if 0 <= int(r[7]) < len(ids):
                ids[int(r[7])] = int(r[4])
        return ids, []


class SAM2Tracker(Tracker):
    """SAM 2.1 on a box prompt. One object at a time; start() again replaces it (new ID)."""
    name = "sam2"
    prompted = True
    keeps_frames = True
    CFG = {"tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
           "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt")}

    def __init__(self, model=None, image_size=None, max_lost=10, min_score=0.5, max_unanchored=30):
        import torch
        import sam2.build_sam as bs
        self.torch = torch
        model = model or os.environ.get("SAM2_MODEL", "tiny")
        image_size = int(image_size or os.environ.get("SAM2_IMAGE_SIZE", "512"))   # 512: ~4x less compute than 1024
        cfg, ckpt = self.CFG[model]
        ckpt = os.path.join(os.environ.get("SAM2_DIR", os.path.join(MODELS, "sam2")), ckpt)
        if not os.path.exists(ckpt):
            raise RuntimeError("SAM 2 checkpoint %s missing: detector/get_models.sh sam2" % ckpt)
        over = [] if image_size == 1024 else ["++model.image_size=%d" % image_size]
        if hasattr(bs, "build_sam2_camera_predictor"):   # streaming predictor (memory across frames)
            self.cam = bs.build_sam2_camera_predictor(cfg, ckpt, device="cuda", hydra_overrides_extra=over)
            self.img_pred, mode = None, "camera predictor"
        else:                                             # official package: image predictor, previous box as prompt
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.cam = None
            self.img_pred, mode = SAM2ImagePredictor(bs.build_sam2(cfg, ckpt, device="cuda", hydra_overrides_extra=over)), \
                "image predictor, box propagation"
            # the image predictor hard-codes the backbone feature sizes of a 1024 input: scale them to image_size
            self.img_pred._bb_feat_sizes = [(image_size // k, image_size // k) for k in (4, 8, 16)]
        self._desc = "SAM 2.1 %s @%d (%s)" % (model, image_size, mode)
        self.max_lost, self.min_score, self.max_unanchored = max_lost, min_score, max_unanchored
        self.next_id = 0
        self.reset()

    @property
    def desc(self):
        return self._desc

    @property
    def has_target(self):
        return self.obj is not None

    def reset(self):
        self.obj = None   # {"id", "box", "cls", "lost"}

    @contextlib.contextmanager
    def _ctx(self):
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            yield

    def start(self, img, box, cls=-1):
        self.next_id += 1
        self.obj = {"id": self.next_id, "box": [float(v) for v in box], "cls": int(cls), "lost": 0, "free": 0}
        if self.cam is not None:
            with self._ctx():
                self.cam.load_first_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                self.cam.add_new_prompt(frame_idx=0, obj_id=1,
                                        bbox=np.array([box[:2], box[2:]], np.float32))
        return self.next_id

    @staticmethod
    def _mask_box(mask):
        cols, rows = np.flatnonzero(mask.any(0)), np.flatnonzero(mask.any(1))
        return [float(cols[0]), float(rows[0]), float(cols[-1]), float(rows[-1])] if len(cols) else None

    @staticmethod
    def _largest_component(mask):
        """Keep the biggest connected blob: stray pieces of background would stretch the box."""
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if n <= 2:
            return mask
        return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    def update(self, img, boxes, scores, cls):
        """Prompt SAM 2 with the previous box, or with the detector's box of the same object when there is one
        (re-anchoring: stops the box from creeping onto the background). A mask that jumps away or changes size too
        much is treated as lost; after max_lost lost frames the object is dropped (the service then clears the target)."""
        none = np.full(len(boxes), -1, np.int64)
        o = self.obj
        if o is None:
            return none, []
        prev = np.asarray(o["box"], np.float32)
        prompt, anchored = prev, False
        if o["cls"] >= 0 and len(boxes):
            same = np.flatnonzero(np.asarray(cls) == o["cls"])
            if len(same):
                ious = iou_matrix(prev[None], np.asarray(boxes)[same])[0]
                k = int(np.argmax(ious))
                if ious[k] >= 0.3:
                    prompt, anchored = np.asarray(boxes)[same[k]], True
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self._ctx():
            if self.cam is not None:   # (streaming predictor: keeps its own memory, prompt not used)
                _, logits = self.cam.track(rgb)
                mask = (logits[0, 0] > 0).cpu().numpy()
                score = 1.0
            else:
                self.img_pred.set_image(rgb)
                masks, sc, _ = self.img_pred.predict(box=prompt.astype(np.float32), multimask_output=False)
                mask, score = masks[0] > 0, float(sc[0])
        mask = self._largest_component(mask)
        nb = self._mask_box(mask)
        ok = nb is not None and score >= self.min_score
        if ok and not anchored:   # plausibility against the last good box (an anchored box comes from the detector)
            area = lambda b: max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            ratio = area(nb) / area(prev)
            ok = 0.4 <= ratio <= 2.5 and iou_matrix(prev[None], np.asarray([nb]))[0, 0] >= 0.3
        # an object the detector knows (cls >= 0) must be confirmed by it now and then: SAM 2 alone slowly slides onto
        # whatever is there once the object has left the view
        o["free"] = 0 if anchored else o["free"] + 1
        if o["cls"] >= 0 and o["free"] > self.max_unanchored:
            self.obj = None
            return none, []
        if not ok:
            o["lost"] += 1
            if o["lost"] > self.max_lost:
                self.obj = None
            return none, []
        o.update(box=nb, lost=0)
        return none, [{"id": o["id"], "box": nb, "score": score, "cls": o["cls"], "mask": mask}]


TRACKERS = {
    "none": Tracker,
    "iou": IoUTracker,
    "botsort": lambda: BotSortTracker(reid=False),
    "botsort_reid": lambda: BotSortTracker(reid=True),
    "sam2": SAM2Tracker,
}


def make_tracker(kind):
    kind = kind or "none"
    if kind not in TRACKERS:
        raise ValueError("unknown tracker %r (choose from %s)" % (kind, ", ".join(TRACKERS)))
    return TRACKERS[kind]()
