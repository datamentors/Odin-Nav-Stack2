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
  sam2          SAM 2.1 (tiny by default) on a box prompt, with its video memory (SAM2_MODE=video, default). Its cost is set
                by SAM2_IMAGE_SIZE (default 512, the size SAM 2 works at whatever image it gets) and SAM2_MODEL
                (tiny | small). Detector boxes of the same class confirm and re-anchor it; see SAM2Tracker

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

    def __init__(self, reid=False, lost_frames=40, reid_weights=None, high=None, new=None):
        from boxmot import BotSort
        from boxmot.trackers.botsort.config import BotSortConfig
        self.name = "botsort_reid" if reid else "botsort"
        # BoxMOT's defaults (first association and new tracks from 0.62) are tuned for pedestrian benchmarks: here a
        # suitcase or a far chair often scores 0.3-0.6 and would never get a track. Keep them near the service threshold.
        high = float(high if high is not None else os.environ.get("BOTSORT_HIGH", "0.3"))
        new = float(new if new is not None else os.environ.get("BOTSORT_NEW", "0.35"))
        cfg = BotSortConfig(use_embeddings=reid, max_lost_frames=lost_frames, track_high_thresh=high,
                            new_track_thresh=new, det_thresh=min(high, 0.3))
        self._make = lambda: BotSort(cfg, reid=self._reid_cfg(reid_weights) if reid else None)
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
    """SAM 2.1 on a box prompt. One object at a time; start() again replaces it (new ID).

    Default (SAM2_MODE=video): the official video model run one frame at a time (track_step) with a small memory bank
    (the prompt frame, the last re-anchor frames and the last few frames), so SAM 2 keeps the object's appearance across
    frames and reports when it is not visible (object score). SAM2_MODE=image: the image predictor prompted with the last
    box each frame (no memory, the earlier behaviour)."""
    name = "sam2"
    prompted = True
    keeps_frames = True
    det_conf = 0.1   # low-score detector boxes still confirm / re-anchor the object
    CFG = {"tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
           "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt")}
    MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model=None, image_size=None, mode=None, max_lost=None, max_unanchored=None, reanchor=None):
        import torch
        import sam2.build_sam as bs
        self.torch = torch
        env = os.environ.get
        model = model or env("SAM2_MODEL", "tiny")
        self.size = int(image_size or env("SAM2_IMAGE_SIZE", "512"))   # 512: ~4x less compute than 1024
        self.mode = mode or env("SAM2_MODE", "video")
        # frames without the object before the target is dropped (SAM 2's own visibility score, or a rejected mask)
        self.max_lost = int(max_lost if max_lost is not None else env("SAM2_MAX_LOST", "30"))
        # an object the detector knows must be confirmed by a detector box this often (frames), or it is dropped
        self.max_unanchored = int(max_unanchored if max_unanchored is not None else env("SAM2_MAX_UNANCHORED", "50"))
        # video mode: re-prompt with the detector's box at most every N frames when SAM 2's box drifts from it
        self.reanchor = int(reanchor if reanchor is not None else env("SAM2_REANCHOR", "10"))
        cfg, ckpt = self.CFG[model]
        ckpt = os.path.join(env("SAM2_DIR", os.path.join(MODELS, "sam2")), ckpt)
        if not os.path.exists(ckpt):
            raise RuntimeError("SAM 2 checkpoint %s missing: detector/get_models.sh sam2" % ckpt)
        over = [] if self.size == 1024 else ["++model.image_size=%d" % self.size]
        if self.mode == "video":
            self.net = bs.build_sam2_video_predictor(cfg, ckpt, device="cuda", hydra_overrides_extra=over)
            self.mean = torch.tensor(self.MEAN, device="cuda").view(1, 3, 1, 1)
            self.std = torch.tensor(self.STD, device="cuda").view(1, 3, 1, 1)
            kind = "video model, memory"
        else:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.img_pred = SAM2ImagePredictor(bs.build_sam2(cfg, ckpt, device="cuda", hydra_overrides_extra=over))
            # the image predictor hard-codes the backbone feature sizes of a 1024 input: scale them to the size
            self.img_pred._bb_feat_sizes = [(self.size // k, self.size // k) for k in (4, 8, 16)]
            kind = "image predictor, box propagation"
        self._desc = "SAM 2.1 %s @%d (%s)" % (model, self.size, kind)
        self.next_id = 0
        self.reset()

    @property
    def desc(self):
        return self._desc

    @property
    def has_target(self):
        return self.obj is not None

    def reset(self):
        self.obj = None   # {"id", "box", "cls", "lost", "free", "t", "last_anchor"}
        self.mem = None   # video mode: {"cond_frame_outputs": {t: out}, "non_cond_frame_outputs": {t: out}}

    @contextlib.contextmanager
    def _ctx(self):
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            yield

    # ---------------------------------------------------------------- video mode
    def _features(self, img):
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = self.torch.from_numpy(x).cuda().permute(2, 0, 1)[None].float().div_(255.0)
        _, feats, pos, sizes = self.net._prepare_backbone_features(self.net.forward_image((x - self.mean) / self.std))
        return feats, pos, sizes

    def _step(self, img, box=None, init=False):
        """One frame through SAM 2 with the memory bank; box = a prompt (makes it a conditioning frame).
        Returns (mask at the image size, object visible)."""
        o, h, w = self.obj, img.shape[0], img.shape[1]
        feats, pos, sizes = self._features(img)
        pts = None
        if box is not None:
            k = self.torch.tensor([self.size / w, self.size / h], device="cuda")
            xy = self.torch.tensor(np.asarray(box, np.float32).reshape(1, 2, 2), device="cuda") * k
            pts = {"point_coords": xy, "point_labels": self.torch.tensor([[2, 3]], dtype=self.torch.int32, device="cuda")}
        out = self.net.track_step(frame_idx=o["t"], is_init_cond_frame=init, current_vision_feats=feats,
                                  current_vision_pos_embeds=pos, feat_sizes=sizes, point_inputs=pts, mask_inputs=None,
                                  output_dict=self.mem, num_frames=1 << 30, run_mem_encoder=True)
        keep = {k_: out[k_] for k_ in ("maskmem_features", "maskmem_pos_enc", "pred_masks", "obj_ptr", "object_score_logits")}
        cond, non = self.mem["cond_frame_outputs"], self.mem["non_cond_frame_outputs"]
        (cond if pts is not None else non)[o["t"]] = keep
        while len(cond) > 3:   # the first prompt and the two latest re-anchors
            del cond[sorted(cond)[1]]
        for t in [t for t in non if t < o["t"] - 16]:   # the memory reads the last 6 frames, object pointers the last 15
            del non[t]
        visible = float(out["object_score_logits"].max()) > 0
        m = self.torch.nn.functional.interpolate(out["pred_masks"].float(), (h, w), mode="bilinear", align_corners=False)
        return (m[0, 0] > 0).cpu().numpy(), visible

    # ---------------------------------------------------------------- common
    def start(self, img, box, cls=-1):
        self.next_id += 1
        self.obj = {"id": self.next_id, "box": [float(v) for v in box], "cls": int(cls), "lost": 0, "free": 0, "t": 0,
                    "last_anchor": 0}
        if self.mode == "video":
            self.mem = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            with self._ctx():
                self._step(img, box, init=True)
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

    def _detector_box(self, boxes, cls, ref, min_iou):
        """The detector's box of the same class that overlaps ref most (IoU >= min_iou), or None."""
        o = self.obj
        if o["cls"] < 0 or not len(boxes):
            return None, 0.0
        same = np.flatnonzero(np.asarray(cls) == o["cls"])
        if not len(same):
            return None, 0.0
        ious = iou_matrix(np.asarray(ref, np.float32)[None], np.asarray(boxes)[same])[0]
        k = int(np.argmax(ious))
        return (np.asarray(boxes)[same[k]], float(ious[k])) if ious[k] >= min_iou else (None, float(ious[k]))

    def update(self, img, boxes, scores, cls):
        """Follow the object one more frame. A detector box of the same object confirms it (and re-anchors SAM 2 when
        they disagree); after max_lost frames without the object, or max_unanchored frames without a confirming detector
        box (objects the detector knows), the object is dropped (the service then clears the target)."""
        none = np.full(len(boxes), -1, np.int64)
        o = self.obj
        if o is None:
            return none, []
        o["t"] += 1
        prev = o["box"]
        with self._ctx():
            if self.mode == "video":
                mask, ok = self._step(img)
                score = 1.0
                nb = self._mask_box(self._largest_component(mask)) if ok else None
                det, iou = self._detector_box(boxes, cls, nb if nb else prev, 0.3)
                if det is not None and iou < 0.75 and o["t"] - o["last_anchor"] >= self.reanchor:
                    o["t"] += 1   # re-prompt on the detector's box (a new conditioning frame of the same image)
                    o["last_anchor"] = o["t"]
                    mask, ok = self._step(img, det)
                    nb = self._mask_box(self._largest_component(mask)) if ok else None
                anchored = det is not None
            else:
                det, _ = self._detector_box(boxes, cls, prev, 0.3)
                anchored = det is not None
                self.img_pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                masks, sc, _ = self.img_pred.predict(box=np.asarray(det if anchored else prev, np.float32),
                                                     multimask_output=False)
                mask, score = self._largest_component(masks[0] > 0), float(sc[0])
                nb = self._mask_box(mask) if score >= 0.5 else None
                if nb is not None and not anchored:   # plausibility against the last good box
                    area = lambda b: max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                    ratio = area(nb) / area(prev)
                    if not (0.4 <= ratio <= 2.5 and iou_matrix(np.asarray([prev]), np.asarray([nb]))[0, 0] >= 0.3):
                        nb = None
        o["free"] = 0 if anchored else o["free"] + 1
        if (o["cls"] >= 0 and o["free"] > self.max_unanchored) or (nb is None and o["lost"] + 1 > self.max_lost):
            self.reset()
            return none, []
        if nb is None:
            o["lost"] += 1
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
