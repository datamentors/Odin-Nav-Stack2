#!/usr/bin/env python3
"""Light per-frame object detectors for the GPU detector service (detector_service.py), one interface for all:

    det = make_detector("yolov5" | "rfdetr" | "yoloe", weights)
    boxes, scores, cls = det.detect(img_bgr, conf)     # boxes N x 4 (x1, y1, x2, y2) in img pixels, scores N, cls N (int)
                                                       # a segmentation model may return a 4th item: masks N x h x w (bool)
    det.names        {cls id: name}
    det.input_size   longest side the model works at (the ROS node sends images about that size, not the 1600 px original)
    det.desc         one line for logs / /health
    det.open_vocab   True when det.set_classes(["chair", "fire extinguisher", ...]) can change what it looks for (YOLOE)

Backends, chosen by the weights file extension:
    yolov5  .engine  TensorRT (FP16 engine from yolo_trt/build_engine.sh)
            .pt      PyTorch on the GPU (needs the yolov5 repo: ~/odin_test/yolo_gpu/yolov5, or $YOLOV5_DIR)
    rfdetr  .engine  TensorRT (FP16 engine from detector/get_models.sh rfdetr, fixed square input)
            .onnx    onnxruntime on the CPU (slow; for tests)
    yoloe   .pt      YOLOE (Ultralytics, open vocabulary, with instance masks): any text classes, changeable at run
                     time (set_classes, ~0.5 s each; the text encoder mobileclip_blt.ts must be next to the weights).
                     A "-pf" (prompt-free) model uses its built-in vocabulary. Needs ultralytics (image WITH_YOLOE=1).

Runs in the Odin-perception venv (Python 3.10, Jetson torch, JetPack TensorRT); nothing here needs ROS.
Ported from the detector benchmark (go2_rf_detr_bench branch, detbench/bench.py) without the benchmark parts.
"""
import os
import sys

import cv2
import numpy as np

HOME = os.path.expanduser("~")
JETPACK_PY = "/usr/lib/python3.10/dist-packages"   # JetPack's tensorrt bindings (appended last: the venv still wins)

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]
# RF-DETR (COCO-pretrained) output slots are the sparse COCO category ids (1 = person ... 90 = toothbrush)
COCO91_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
              33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
              60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class TRTEngine:
    """TensorRT engine with one input, run on torch CUDA tensors: engine(x) -> {output name: tensor}."""

    def __init__(self, path):
        if JETPACK_PY not in sys.path:
            sys.path.append(JETPACK_PY)
        import tensorrt as trt
        import torch
        self.torch, self.dev = torch, torch.device("cuda:0")
        with open(path, "rb") as f:
            self.eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if self.eng is None:
            raise RuntimeError("cannot load the TensorRT engine %s (rebuild it after a JetPack/TensorRT update)" % path)
        self.ctx = self.eng.create_execution_context()
        dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64}
        names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inputs = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dtype = {n: dt[self.eng.get_tensor_dtype(n)] for n in names}
        self.in_shape = tuple(self.eng.get_tensor_shape(self.inputs[0]))   # -1 for dynamic axes
        self.stream = torch.cuda.Stream(self.dev)   # own stream (TensorRT warns about syncs on the default one)

    def __call__(self, x):
        torch, name = self.torch, self.inputs[0]
        x = x.to(self.dtype[name]).contiguous()
        self.ctx.set_input_shape(name, tuple(x.shape))
        self.ctx.set_tensor_address(name, x.data_ptr())
        out = {}
        for n in self.outputs:
            out[n] = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=self.dtype[n], device=self.dev)
            self.ctx.set_tensor_address(n, out[n].data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream(self.dev))   # inputs ready before TensorRT starts
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()                                       # outputs complete before anyone reads them
        return out


def letterbox(img, size=640, stride=32, color=(114, 114, 114)):
    """YOLOv5 letterbox (auto=True): longest side -> size, pad to a multiple of stride. Returns img, ratio, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    pw, ph = (stride - nw % stride) % stride / 2, (stride - nh % stride) % stride / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(ph - 0.1)), int(round(ph + 0.1))
    left, right = int(round(pw - 0.1)), int(round(pw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (left, top)


class Detector:
    names, input_size, desc, open_vocab = {}, 640, "", False

    def detect(self, img, conf):
        raise NotImplementedError

    def set_classes(self, names):
        raise NotImplementedError("%s has a fixed set of classes" % self.desc)


class YOLOv5(Detector):
    """YOLOv5 (COCO, 80 classes). Letterbox 640 + YOLOv5 decoding and class-wise NMS done here, so the TensorRT path
    does not need the yolov5 repo; only the .pt path loads it (attempt_load)."""

    def __init__(self, weights, iou=0.45, max_det=300):
        import torch
        import torchvision
        self.torch, self.nms = torch, torchvision.ops.batched_nms
        self.iou, self.max_det, self.input_size = iou, max_det, 640
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ext = os.path.splitext(weights)[1]
        if ext == ".engine":
            trt = TRTEngine(weights)
            self.run = lambda x: trt(x.float())[trt.outputs[0]].float()
            names, backend = COCO80, "TensorRT"
            nf = os.path.join(os.path.dirname(weights), "names.json")   # written by yolo_trt/export_onnx.py
            if os.path.exists(nf):
                import json
                names = [v for _, v in sorted(json.load(open(nf)).items(), key=lambda kv: int(kv[0]))]
        elif ext == ".pt":
            ydir = os.environ.get("YOLOV5_DIR", os.path.join(HOME, "odin_test/yolo_gpu/yolov5"))
            sys.path.insert(0, os.path.dirname(ydir))
            import yolo5_stubs   # empty plotting/training modules, no pip installs from this process
            yolo5_stubs.install()
            sys.path.insert(0, ydir)
            from models.experimental import attempt_load
            m = attempt_load(weights, device=self.dev).eval()
            half = self.dev.type == "cuda"
            m = m.half() if half else m.float()
            self.run = lambda x: m(x.half() if half else x.float())[0].float()
            n = m.module.names if hasattr(m, "module") else m.names
            names, backend = ([n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)), "PyTorch"
        else:
            raise ValueError("YOLOv5 weights must be .engine or .pt: %s" % weights)
        self.names = {i: str(v) for i, v in enumerate(names)}
        self.desc = "YOLOv5 %s (%s)" % (os.path.basename(weights), backend)
        self.detect(np.zeros((544, 640, 3), np.uint8), 0.5)   # warm-up (first CUDA call is slow)

    def detect(self, img, conf):
        torch = self.torch
        lb, r, (px, py) = letterbox(img, self.input_size)
        x = torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))).to(self.dev)  # BGR->RGB, CHW
        x = x.unsqueeze(0).float() / 255.0
        with torch.no_grad():
            p = self.run(x)[0]                                  # N x (4 box + 1 obj + classes)
            score, cls = (p[:, 5:] * p[:, 4:5]).max(1)          # same as YOLOv5 non_max_suppression, single label
            k = score > conf
            p, score, cls = p[k], score[k], cls[k]
            xy, wh = p[:, :2], p[:, 2:4] / 2
            boxes = torch.cat([xy - wh, xy + wh], 1)
            keep = self.nms(boxes, score, cls, self.iou)[: self.max_det]
            boxes, score, cls = boxes[keep].cpu().numpy(), score[keep].cpu().numpy(), cls[keep].cpu().numpy()
        boxes = (boxes - [px, py, px, py]) / r
        h, w = img.shape[:2]
        boxes = np.clip(boxes, 0, [w, h, w, h])
        return boxes.astype(np.float32), score.astype(np.float32), cls.astype(np.int64)


class RFDETR(Detector):
    """RF-DETR (COCO-pretrained, Roboflow). The frame is stretched to the square model resolution (as rfdetr does),
    ImageNet-normalised; output decoded like rfdetr's PostProcess (per-class sigmoid, top-k, threshold). No NMS."""

    def __init__(self, weights, threads=0):
        ext = os.path.splitext(weights)[1]
        if ext == ".engine":
            import torch
            self.torch = torch
            self.trt = TRTEngine(weights)
            self.res = int(self.trt.in_shape[2])
            self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
            self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
            backend = "TensorRT"
        elif ext == ".onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads or len(os.sched_getaffinity(0))   # explicit: no thread pinning
            so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(weights, so, providers=["CPUExecutionProvider"])
            self.iname = self.sess.get_inputs()[0].name
            self.onames = [o.name for o in self.sess.get_outputs()]
            self.res, self.trt, backend = int(self.sess.get_inputs()[0].shape[2]), None, "onnxruntime CPU"
        else:
            raise ValueError("RF-DETR weights must be .engine or .onnx: %s" % weights)
        self.input_size = self.res
        self.names = {cid: n for cid, n in zip(COCO91_IDS, COCO80)}
        self.desc = "RF-DETR %s (%s, %dx%d)" % (os.path.basename(weights), backend, self.res, self.res)
        self.detect(np.zeros((self.res, self.res, 3), np.uint8), 0.5)   # warm-up

    def detect(self, img, conf):
        h, w = img.shape[:2]
        small = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        if self.trt is not None:   # upload only the small uint8 image; normalise on the GPU
            torch = self.torch
            with torch.no_grad():
                x = torch.from_numpy(small).cuda()[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255
                out = self.trt((x - self.mean) / self.std)
                dets, logits = out["dets"][0].float().cpu().numpy(), out["labels"][0].float().cpu().numpy()
        else:
            x = small[:, :, ::-1].astype(np.float32) / 255
            x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
            o = dict(zip(self.onames, self.sess.run(None, {self.iname: np.ascontiguousarray(x)})))
            dets, logits = o["dets"][0], o["labels"][0]
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -88, 88)))
        flat = scores.ravel()
        k = min(dets.shape[0], flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[flat[idx] > conf]
        idx = idx[np.argsort(-flat[idx])]
        q, c = np.divmod(idx, scores.shape[1])
        cx, cy, bw, bh = [dets[q, i] for i in range(4)]
        boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        keep = np.isin(c, COCO91_IDS)   # slots without a COCO class (unused ids) are never reported
        return boxes.reshape(-1, 4)[keep].astype(np.float32), flat[idx][keep].astype(np.float32), c[keep].astype(np.int64)


class YOLOE(Detector):
    """YOLOE segmentation model (Ultralytics) prompted with text classes. Returns boxes, scores, class ids and masks
    (at the size of the image given). Classes from YOLOE_CLASSES (comma separated) or COCO by default."""
    open_vocab = True

    def __init__(self, weights, classes=None):
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # never pip-install from this process
        os.environ.setdefault("YOLO_OFFLINE", "true")        # never download: models come from get_models.sh
        os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(weights), ".ultralytics"))
        import threading
        import torch
        from ultralytics import YOLOE as _YOLOE
        self.lock = threading.Lock()   # set_classes and predict must not overlap
        self.half = torch.cuda.is_available()
        self.dir = os.path.dirname(os.path.abspath(weights))
        self.m = _YOLOE(weights)
        self.prompt_free = "-pf" in os.path.basename(weights)
        self.input_size = int(os.environ.get("YOLOE_IMGSZ", "640"))
        self.desc = "YOLOE %s (PyTorch%s)" % (os.path.basename(weights), " FP16" if self.half else "")
        if self.prompt_free:
            self.open_vocab = False
            self.names = {int(k): str(v) for k, v in self.m.names.items()}
        else:
            env = os.environ.get("YOLOE_CLASSES", "")
            self.set_classes(classes or [c.strip() for c in env.split(",") if c.strip()] or COCO80)
        self._warm()

    def _warm(self):   # first predict at the live size (518 x 640 for the 1296 x 1600 camera) is slow: do it now
        self.detect(np.zeros((int(self.input_size * 1296 / 1600), self.input_size, 3), np.uint8), 0.5)

    def set_classes(self, names):
        if self.prompt_free:
            raise NotImplementedError("a prompt-free YOLOE model has a fixed vocabulary")
        names = [str(n).strip() for n in names if str(n).strip()]
        if not names:
            raise ValueError("no class names")
        cwd = os.getcwd()
        with self.lock:
            os.chdir(self.dir)   # Ultralytics finds the text encoder (mobileclip_blt.ts) in the working directory
            try:
                self.m.set_classes(names, self.m.get_text_pe(names))
            finally:
                os.chdir(cwd)
            self.names = {i: n for i, n in enumerate(names)}
        self._warm()
        return self.names

    def detect(self, img, conf):
        with self.lock:
            r = self.m.predict(img, conf=conf, imgsz=self.input_size, half=self.half, retina_masks=True, verbose=False)[0]
        b = r.boxes
        boxes, scores, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(np.int64)
        masks = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else [None] * len(boxes)
        return boxes.astype(np.float32), scores.astype(np.float32), cls, masks


DETECTORS = {"yolov5": YOLOv5, "rfdetr": RFDETR, "yoloe": YOLOE}


def default_weights(kind):
    """Candidate weights in order; the service uses the first one that exists and loads."""
    if kind == "yolov5":
        return [os.path.join(HOME, "odin_test/yolo_trt/yolov5l_fp16.engine"),
                os.path.join(HOME, "odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/yolov5l.pt")]
    if kind == "rfdetr":
        return [os.path.join(HOME, "odin_test/models/rfdetr_large_fp16.engine"),
                os.path.join(HOME, "odin_test/models/rfdetr_medium_fp16.engine")]
    if kind == "yoloe":
        return [os.path.join(HOME, "odin_test/models/yoloe-11l-seg.pt"),
                os.path.join(HOME, "odin_test/models/yoloe-11m-seg.pt"),
                os.path.join(HOME, "odin_test/models/yoloe-11s-seg.pt")]
    raise ValueError("unknown detector %r (choose from %s)" % (kind, ", ".join(DETECTORS)))


def make_detector(kind, weights=None, log=print):
    """weights: a path, a comma-separated list of fallbacks, or None for default_weights(kind)."""
    if kind not in DETECTORS:
        raise ValueError("unknown detector %r (choose from %s)" % (kind, ", ".join(DETECTORS)))
    cands = [os.path.expanduser(w.strip()) for w in weights.split(",")] if weights else default_weights(kind)
    errors = []
    for w in cands:
        if not os.path.exists(w):
            errors.append("%s: missing" % w)
            continue
        try:
            return DETECTORS[kind](w)
        except Exception as e:   # e.g. an engine built by another TensorRT version: try the next candidate
            errors.append("%s: %s" % (w, e))
            log("detector %s: %s failed (%s), trying the next weights" % (kind, w, e))
    raise RuntimeError("no usable weights for %s:\n  %s" % (kind, "\n  ".join(errors)))
