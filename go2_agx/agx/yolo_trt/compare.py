#!/usr/bin/env python3
# AGX ORIN, Odin-perception CUDA venv. TensorRT test, step 3: the SAME letterboxed images to the PyTorch service
# (8750, the one in use) and the TensorRT service (8751); compares detections and time. Read-only for the robot.
#   ~/odin-perception/.venv/bin/python ~/odin_test/yolo_trt/compare.py [images...]
import glob, json, os, sys, time, urllib.request

GPU_DIR = os.path.expanduser("~/odin_test/yolo_gpu")
sys.path.insert(0, GPU_DIR)
import yolo5_stubs  # noqa: E402
yolo5_stubs.install()
sys.path.insert(0, os.path.join(GPU_DIR, "yolov5"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402

NAMES = json.load(open(os.path.expanduser("~/odin_test/yolo_trt/names.json")))
SERVICES = {"pytorch": "http://127.0.0.1:8750", "tensorrt": "http://127.0.0.1:8751"}
REPEAT = 10


def ask(url, img):
    req = urllib.request.Request(url + "/infer?conf=0.25&iou=0.45", data=img.tobytes(), method="POST",
                                 headers={"X-Shape": "%d,%d" % img.shape[:2], "Content-Type": "application/octet-stream"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


files = sys.argv[1:] or sorted(glob.glob(os.path.expanduser("~/odin_test/yolo/bench/*.jpg")))
tot = {k: [] for k in SERVICES}
same = diff = 0
for f in files:
    im = cv2.imread(f)
    if im is None:
        continue
    lb = np.ascontiguousarray(letterbox(im, 640, stride=32, auto=True)[0])
    res = {}
    for k, url in SERVICES.items():
        res[k] = ask(url, lb)                       # warm
        t = time.time()
        for _ in range(REPEAT):
            ask(url, lb)
        tot[k].append((time.time() - t) * 1000 / REPEAT)
    print("\n%s  (%dx%d)" % (os.path.basename(f), lb.shape[0], lb.shape[1]))
    used = set()
    for d in sorted(res["pytorch"], key=lambda r: -r[4]):
        best, bj = 0, None
        for j, e in enumerate(res["tensorrt"]):
            if j not in used and int(e[5]) == int(d[5]) and iou(d, e) > best:
                best, bj = iou(d, e), j
        name = NAMES[str(int(d[5]))]
        if bj is not None and best > 0.5:
            used.add(bj); same += 1
            print("  %-14s pytorch %.2f | tensorrt %.2f   (box overlap %.2f)" % (name, d[4], res["tensorrt"][bj][4], best))
        else:
            diff += 1
            print("  %-14s pytorch %.2f | tensorrt  ---   ONLY PYTORCH" % (name, d[4]))
    for j, e in enumerate(res["tensorrt"]):
        if j not in used:
            diff += 1
            print("  %-14s pytorch  --- | tensorrt %.2f   ONLY TENSORRT" % (NAMES[str(int(e[5]))], e[4]))
print("\nSUMMARY: %d images, %d detections in both, %d in only one" % (len(tot["pytorch"]), same, diff))
for k in SERVICES:
    print("  %-8s %.0f ms per image (HTTP round trip, average)" % (k, sum(tot[k]) / max(1, len(tot[k]))))
