#!/usr/bin/env python3
"""Detector benchmark: speed (pre / infer / post / end-to-end latency, FPS) and accuracy (recall, precision,
AP50, AP50-95) of YOLOv5 and RF-DETR on a folder of frames, optionally with 2x2 tiling.

Backend is chosen from the weights file extension:
  .pt      YOLOv5 in PyTorch (cuda or cpu)               needs torch + the yolov5 repo (--yolov5-dir)
  .engine  TensorRT (YOLOv5 or RF-DETR)                  needs JetPack tensorrt + torch with CUDA
  .onnx    onnxruntime on CPU (YOLOv5 or RF-DETR)        needs onnxruntime (+ torch for YOLOv5 NMS)
Family (yolov5 / rfdetr) is taken from the file name unless --family is given.

Labels (optional) are COCO JSON, e.g. CVAT "COCO 1.0" export; matched to frames by file name and to model
classes by category NAME (default: person). Without labels only speed and detection counts are reported.

Latency is measured from a decoded BGR frame in memory to boxes in original pixel coordinates (JPEG decode
excluded). Accuracy uses one extra untimed pass per frame at --eval-conf; recall/precision are read at --conf.

Python 3.8 compatible (runs in the ROS Noetic containers on the Go2 too).

Examples (Orin):
  V=~/odin_test/detbench/.venv/bin/python; M=~/odin_test/detbench/models
  $V bench.py --images frames/ --labels labels.json --weights ~/odin_test/yolo_trt/yolov5l_fp16.engine
  $V bench.py --images frames/ --labels labels.json --weights $M/rfdetr_medium_fp16.engine --tiles 4
"""
import argparse, csv, glob, json, os, platform, socket, subprocess, sys, time

import cv2
import numpy as np

HOME = os.path.expanduser("~")
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]
# RF-DETR COCO-pretrained output slots use the sparse COCO category ids (1 = person ... 90 = toothbrush)
COCO91_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
              33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
              60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
RFDETR_NAMES = {i: n for i, n in zip(COCO91_IDS, COCO80)}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------------------------------------- runtimes
class TRTEngine:
    """Generic TensorRT engine runner on torch CUDA tensors (same approach as yolo_trt_service.py)."""

    def __init__(self, path):
        sys.path.append("/usr/lib/python3.10/dist-packages")  # JetPack tensorrt, appended last
        import tensorrt as trt
        import torch
        self.torch = torch
        self.dev = torch.device("cuda:0")
        with open(path, "rb") as f:
            self.eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if self.eng is None:
            sys.exit("cannot load TensorRT engine %s (rebuild it after a JetPack/TensorRT update)" % path)
        self.ctx = self.eng.create_execution_context()
        dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64}
        names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inputs = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dtype = {n: dt[self.eng.get_tensor_dtype(n)] for n in names}
        self.in_shape = tuple(self.eng.get_tensor_shape(self.inputs[0]))
        self.stream = torch.cuda.Stream(self.dev)

    def __call__(self, x):
        """x: torch CUDA tensor for the single input -> dict name -> torch CUDA tensor."""
        torch = self.torch
        name = self.inputs[0]
        x = x.to(self.dtype[name]).contiguous()
        self.ctx.set_input_shape(name, tuple(x.shape))
        self.ctx.set_tensor_address(name, x.data_ptr())
        out = {}
        for n in self.outputs:
            out[n] = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=self.dtype[n], device=self.dev)
            self.ctx.set_tensor_address(n, out[n].data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream(self.dev))
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return out


def ort_session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    # explicit count = no thread pinning (on the Orin some cores are offline in 30 W mode and pinning crashes ORT)
    so.intra_op_num_threads = threads or len(os.sched_getaffinity(0))
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return s


def sync(dev):
    if dev == "cuda":
        import torch
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------------- detectors
class Detector:
    """detect(img_bgr, conf) -> (boxes Nx4 xyxy px, scores N, class names list), and .t = [pre, infer, post] s."""
    family = backend = precision = input_desc = ""
    dev = "cpu"


class YOLOv5(Detector):
    family = "yolov5"

    def __init__(self, a):
        import torch
        self.torch = torch
        ydir = os.path.expanduser(a.yolov5_dir)
        stubs = os.path.join(os.path.dirname(ydir), "yolo5_stubs.py")
        if os.path.exists(stubs):  # empty plotting/training modules, no pip installs (same as the GPU service)
            sys.path.insert(0, os.path.dirname(ydir))
            import yolo5_stubs
            yolo5_stubs.install()
        sys.path.insert(0, ydir)
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes
        self.letterbox, self.nms, self.scale = letterbox, non_max_suppression, scale_boxes
        self.imgsz, self.iou = a.imgsz, a.iou
        ext = os.path.splitext(a.weights)[1]
        self.names = COCO80
        if ext == ".pt":
            from models.experimental import attempt_load
            self.dev = a.device
            self.half = a.device == "cuda" and not a.fp32
            m = attempt_load(a.weights, device=torch.device("cuda:0" if self.dev == "cuda" else "cpu"))
            m = m.half() if self.half else m.float()
            self.model = m.eval()
            n = m.module.names if hasattr(m, "module") else m.names
            self.names = [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)
            self.backend, self.precision = "pytorch", ("fp16" if self.half else "fp32")
            self.run = lambda x: self.model(x)[0]
        elif ext == ".engine":
            self.dev, self.half = "cuda", False
            self.trt = TRTEngine(a.weights)
            self.backend, self.precision = "tensorrt", ("fp16" if "fp16" in a.weights else "engine")
            self.run = lambda x: self.trt(x)[self.trt.outputs[0]].float()
        elif ext == ".onnx":
            self.dev, self.half = "cpu", False
            self.sess = ort_session(a.weights, a.threads)
            self.backend, self.precision = "onnxruntime", "fp32"
            iname = self.sess.get_inputs()[0].name
            self.run = lambda x: torch.from_numpy(self.sess.run(None, {iname: x.numpy()})[0])
        else:
            sys.exit("unknown weights type %s" % a.weights)
        self.input_desc = "letterbox %d" % self.imgsz

    def detect(self, img, conf):
        torch = self.torch
        t0 = time.perf_counter()
        lb = self.letterbox(img, self.imgsz, stride=32, auto=True)[0]
        x = torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)))  # BGR->RGB, CHW
        if self.dev == "cuda":
            x = x.cuda()
        x = (x.half() if self.half else x.float()).unsqueeze(0) / 255.0
        sync(self.dev)
        t1 = time.perf_counter()
        with torch.no_grad():
            raw = self.run(x)
        sync(self.dev)
        t2 = time.perf_counter()
        with torch.no_grad():
            p = self.nms(raw, conf_thres=conf, iou_thres=self.iou)[0]
            p[:, :4] = self.scale(x.shape[2:], p[:, :4], img.shape)
            p = p.float().cpu().numpy()
        t3 = time.perf_counter()
        self.t = [t1 - t0, t2 - t1, t3 - t2]
        return p[:, :4], p[:, 4], [self.names[int(c)] for c in p[:, 5]]


class RFDETR(Detector):
    family = "rfdetr"

    def __init__(self, a):
        ext = os.path.splitext(a.weights)[1]
        if ext == ".engine":
            import torch
            self.torch, self.dev = torch, "cuda"
            self.trt = TRTEngine(a.weights)
            self.res = self.trt.in_shape[2]
            self.backend, self.precision = "tensorrt", ("fp16" if "fp16" in a.weights else "engine")
            self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
            self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
        elif ext == ".onnx":
            self.dev = "cpu"
            self.sess = ort_session(a.weights, a.threads)
            self.iname = self.sess.get_inputs()[0].name
            self.onames = [o.name for o in self.sess.get_outputs()]
            self.res = self.sess.get_inputs()[0].shape[2]
            self.backend, self.precision = "onnxruntime", "fp32"
        else:
            sys.exit("RF-DETR: use the exported .onnx (CPU) or .engine (TensorRT); got %s" % a.weights)
        self.input_desc = "resize %dx%d" % (self.res, self.res)

    def detect(self, img, conf):
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        # stretch-resize (bilinear, half-pixel, no antialias, as rfdetr predict()) on the CPU, then normalise
        small = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
        if self.dev == "cuda":  # upload only the small uint8 image; normalise on the GPU
            torch = self.torch
            x = torch.from_numpy(small).cuda()[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255
            x = (x - self.mean) / self.std
            sync("cuda")
            t1 = time.perf_counter()
            out = self.trt(x)
            dets, logits = out["dets"][0].float().cpu().numpy(), out["labels"][0].float().cpu().numpy()
        else:
            x = small[:, :, ::-1].astype(np.float32) / 255
            x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
            t1 = time.perf_counter()
            o = dict(zip(self.onames, self.sess.run(None, {self.iname: np.ascontiguousarray(x)})))
            dets, logits = o["dets"][0], o["labels"][0]
        t2 = time.perf_counter()
        # decode like rfdetr PostProcess: per-class sigmoid, top-k over (query, class) pairs, threshold
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -88, 88)))
        flat = scores.ravel()
        k = min(dets.shape[0], flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[flat[idx] > conf]
        idx = idx[np.argsort(-flat[idx])]
        q, c = np.divmod(idx, scores.shape[1])
        cx, cy, bw, bh = [dets[q, i] for i in range(4)]
        boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        names = [RFDETR_NAMES.get(int(i), "cls%d" % i) for i in c]
        t3 = time.perf_counter()
        self.t = [t1 - t0, t2 - t1, t3 - t2]
        return boxes.reshape(-1, 4).astype(np.float32), flat[idx].astype(np.float32), names


# ----------------------------------------------------------------------------------------------- tiling
def nms_by_class(boxes, scores, names, iou_thr):
    keep = []
    for n in set(names):
        ids = np.array([i for i, m in enumerate(names) if m == n])
        ids = ids[np.argsort(-scores[ids])]
        while ids.size:
            i = ids[0]
            keep.append(i)
            if ids.size == 1:
                break
            ious = iou_matrix(boxes[i:i + 1], boxes[ids[1:]])[0]
            ids = ids[1:][ious < iou_thr]
    keep = sorted(keep, key=lambda i: -scores[i])
    return boxes[keep], scores[keep], [names[i] for i in keep]


def tiled(det, img, conf, mode, overlap, iou_thr):
    """mode 4: 2x2 overlapping tiles; mode 5: full frame + 2x2 tiles. Merged with class-wise NMS."""
    h, w = img.shape[:2]
    tw, th = int(round(w * (1 + overlap) / 2)), int(round(h * (1 + overlap) / 2))
    crops = [(x0, y0) for y0 in (0, h - th) for x0 in (0, w - tw)]
    B, S, N, T = [], [], [], np.zeros(3)
    if mode == 5:
        b, s, n = det.detect(img, conf)
        B.append(b); S.append(s); N += n; T += det.t
    for x0, y0 in crops:
        b, s, n = det.detect(np.ascontiguousarray(img[y0:y0 + th, x0:x0 + tw]), conf)
        B.append(b + np.array([x0, y0, x0, y0], np.float32)); S.append(s); N += n; T += det.t
    t = time.perf_counter()
    b, s, n = nms_by_class(np.concatenate(B), np.concatenate(S), N, iou_thr) if N else (np.zeros((0, 4)), np.zeros(0), [])
    T[2] += time.perf_counter() - t
    det.t = list(T)
    return b, s, n


def detect(det, img, conf, a):
    return tiled(det, img, conf, a.tiles, a.tile_overlap, a.iou) if a.tiles else det.detect(img, conf)


# ----------------------------------------------------------------------------------------------- metrics
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(preds, gts, thr):
    """preds: list per image of (boxes, scores); gts: list per image of boxes -> (scores, tp flags, n_gt)."""
    S, TP, n = [], [], 0
    for (pb, ps), gb in zip(preds, gts):
        n += len(gb)
        order = np.argsort(-ps)
        used = np.zeros(len(gb), bool)
        ious = iou_matrix(pb[order], gb)
        for r, i in enumerate(order):
            j = int(np.argmax(ious[r])) if len(gb) else -1
            ok = j >= 0 and ious[r, j] >= thr and not used[j]
            if ok:
                used[j] = True
            S.append(ps[i]); TP.append(ok)
    return np.array(S), np.array(TP, bool), n


def ap(scores, tp, n_gt):
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-scores)
    tp = tp[o]
    rec = np.cumsum(tp) / n_gt
    prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    rec, prec = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    prec = np.flip(np.maximum.accumulate(np.flip(prec)))
    x = np.linspace(0, 1, 101)  # COCO 101-point interpolation
    return float(np.mean(np.interp(x, rec, prec, right=0)))


def load_labels(path, files):
    d = json.load(open(path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_name = {os.path.basename(im["file_name"]): im["id"] for im in d["images"]}
    gts = {}
    for an in d["annotations"]:
        if an.get("iscrowd"):
            continue
        x, y, w, h = an["bbox"]
        gts.setdefault(an["image_id"], []).append((cats[an["category_id"]], [x, y, x + w, y + h]))
    out, missing = [], 0
    for f in files:
        iid = by_name.get(os.path.basename(f))
        missing += iid is None
        out.append(gts.get(iid, []) if iid is not None else None)
    return out, missing


# ----------------------------------------------------------------------------------------------- main
def hw_info():
    rel = ""
    if os.path.exists("/etc/nv_tegra_release"):
        rel = "jetson"
        try:
            pm = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
            rel += " " + [l for l in pm.splitlines() if "Power Mode" in l][0].split(":")[-1].strip()
        except Exception:
            pass
    else:
        rel = platform.machine()
    return socket.gethostname(), rel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder of .jpg/.png frames")
    p.add_argument("--labels", help="COCO JSON with ground truth (optional)")
    p.add_argument("--weights", required=True, help=".pt (YOLOv5 PyTorch), .engine (TensorRT) or .onnx (onnxruntime CPU)")
    p.add_argument("--family", choices=["yolov5", "rfdetr"], help="default: from the file name")
    p.add_argument("--name", help="model name in the results (default: file name)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="for .pt only")
    p.add_argument("--fp32", action="store_true", help=".pt on cuda: run FP32 instead of FP16")
    p.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads (0 = default)")
    p.add_argument("--yolov5-dir", default=os.path.join(HOME, "odin_test/yolo_gpu/yolov5"))
    p.add_argument("--imgsz", type=int, default=640, help="YOLOv5 letterbox size (live: 640)")
    p.add_argument("--conf", type=float, default=0.25, help="operating confidence (live YOLO: 0.25)")
    p.add_argument("--eval-conf", type=float, default=0.001, help="confidence for the AP pass")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU (YOLOv5 and tile merge)")
    p.add_argument("--tiles", type=int, default=0, choices=[0, 4, 5], help="0 full frame, 4 = 2x2 tiles, 5 = full + 2x2")
    p.add_argument("--tile-overlap", type=float, default=0.2)
    p.add_argument("--classes", default="person", help="comma list of class names to score")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=3, help="timed runs per frame")
    p.add_argument("--limit", type=int, default=0, help="use only the first N frames")
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--hw", help="hardware label (default: hostname + power mode)")
    p.add_argument("--note", default="")
    p.add_argument("--save-vis", help="folder to write frames with boxes (operating conf)")
    a = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit("no images in %s" % a.images)
    fam = a.family or ("rfdetr" if "rfdetr" in os.path.basename(a.weights).lower() or "rf-detr" in a.weights.lower() else "yolov5")
    det = (RFDETR if fam == "rfdetr" else YOLOv5)(a)
    name = a.name or os.path.splitext(os.path.basename(a.weights))[0]
    host, hwrel = hw_info()
    hw = a.hw or ("%s (%s)" % (host, hwrel))
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    print("%s | %s %s %s | %s | tiles %d | %d frames | %s" % (name, fam, det.backend, det.precision, det.input_desc, a.tiles, len(files), hw), flush=True)

    imgs = [cv2.imread(f) for f in files]
    for _ in range(a.warmup):
        detect(det, imgs[0], a.conf, a)

    E, T, ndet, evalp = [], [], 0, []
    for k, (f, img) in enumerate(zip(files, imgs)):
        for _ in range(a.repeat):
            t = time.perf_counter()
            b, s, n = detect(det, img, a.conf, a)
            E.append(time.perf_counter() - t)
            T.append(det.t)
        ndet += sum(1 for c in n if c in classes)
        if a.save_vis:
            os.makedirs(a.save_vis, exist_ok=True)
            v = img.copy()
            for bb, ss, nn in zip(b, s, n):
                c = (0, 200, 0) if nn in classes else (160, 160, 160)
                cv2.rectangle(v, tuple(int(x) for x in bb[:2]), tuple(int(x) for x in bb[2:]), c, 2)
                cv2.putText(v, "%s %.2f" % (nn, ss), (int(bb[0]), int(bb[1]) - 4), 0, 0.6, c, 2)
            cv2.imwrite(os.path.join(a.save_vis, os.path.basename(f)), v)
        if a.labels:
            evalp.append(detect(det, img, a.eval_conf, a))
        if (k + 1) % 25 == 0:
            print("  %d/%d  %.1f ms" % (k + 1, len(files), 1000 * np.mean(E[-25 * a.repeat:])), flush=True)

    E, T = np.array(E) * 1000, np.array(T) * 1000
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "hw": hw, "model": name, "family": fam, "backend": det.backend,
           "precision": det.precision, "input": det.input_desc, "tiles": a.tiles, "frames": len(files),
           "pre_ms": "%.1f" % T[:, 0].mean(), "infer_ms": "%.1f" % T[:, 1].mean(), "post_ms": "%.1f" % T[:, 2].mean(),
           "e2e_ms": "%.1f" % E.mean(), "p50_ms": "%.1f" % np.percentile(E, 50), "p95_ms": "%.1f" % np.percentile(E, 95),
           "fps": "%.1f" % (1000 / E.mean()), "conf": a.conf, "classes": "+".join(classes),
           "dets_per_frame": "%.2f" % (ndet / len(files)), "gt": "", "recall": "", "precision_at_conf": "",
           "ap50": "", "ap50_95": "", "note": a.note}

    if a.labels:
        gts, missing = load_labels(a.labels, files)
        if missing:
            print("WARNING: %d frames have no entry in %s (scored as empty)" % (missing, a.labels))
        gts = [g or [] for g in gts]
        res = {}
        for cls in classes:
            pr = [(b[[i for i, m in enumerate(n) if m == cls]], s[[i for i, m in enumerate(n) if m == cls]]) for b, s, n in evalp]
            pr = [(b.reshape(-1, 4), s) for b, s in pr]
            gb = [np.array([bb for c, bb in g if c == cls], np.float32).reshape(-1, 4) for g in gts]
            S, TP, ng = match(pr, gb, 0.5)
            op = S >= a.conf
            aps = [ap(*match(pr, gb, thr)) for thr in np.arange(0.5, 0.96, 0.05)]
            res[cls] = dict(gt=ng, recall=TP[op].sum() / max(ng, 1), prec=TP[op].sum() / max(op.sum(), 1),
                            ap50=aps[0], ap5095=float(np.mean(aps)))
            print("  %-10s gt %4d | recall %.3f  precision %.3f @conf %.2f | AP50 %.3f  AP50-95 %.3f"
                  % (cls, ng, res[cls]["recall"], res[cls]["prec"], a.conf, res[cls]["ap50"], res[cls]["ap5095"]))
        f = lambda k, fmt="%.3f": "/".join(fmt % res[c][k] for c in classes)
        row.update(gt=f("gt", "%d"), recall=f("recall"), precision_at_conf=f("prec"), ap50=f("ap50"), ap50_95=f("ap5095"))

    print("  latency e2e %s ms (p50 %s, p95 %s) = %s FPS | pre %s  infer %s  post %s ms"
          % (row["e2e_ms"], row["p50_ms"], row["p95_ms"], row["fps"], row["pre_ms"], row["infer_ms"], row["post_ms"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    print("  -> %s" % a.out)


if __name__ == "__main__":
    main()
