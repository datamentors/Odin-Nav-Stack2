# detbench — YOLOv5l vs RF-DETR vs 4-tile, Go2 CPU vs Orin

| File | What |
|---|---|
| `extract_frames.py` | ROS 1 bags → evenly spaced, de-duplicated JPEGs for labelling (needs ROS Noetic python) |
| `prelabel.py` | Two detectors → merged COCO pre-labels, a starting point for human labelling (never ground truth) |
| `errors.py` | Error analysis of one model vs the ground truth: TP / missed / wrong boxes per frame (CSV + pictures) and its detections as COCO |
| `labelstudio.py` | COCO pre-labels → Label Studio tasks + config, and Label Studio export → COCO ground truth |
| `bench.py` | The benchmark: latency (pre/infer/post/e2e, p50/p95), FPS, recall/precision @conf, AP50, AP50-95 per class. One CSV row per run |
| `update_from_github.sh` | On the Orin: pull this folder from GitHub (sparse, shallow clone < 1 MB) and copy the code into `~/odin_test/detbench` |
| `run_orin.sh` | Full Orin matrix: YOLOv5l PyTorch + TensorRT, RF-DETR medium/large TensorRT, each with and without 2x2 tiles |
| `export_rfdetr.py`, `build_engines.sh` | RF-DETR → ONNX → TensorRT FP16 (already done) |
| `data/labels_person.json` | Ground truth of the 25 Nov restaurant set: 138 frames, 121 `person` boxes (frames are not in git) |
| `results/2026-11-25_orin.csv` | All Orin runs on that set (see §5) |
| `models/` *(Orin only)* | `rfdetr_{medium,large}.onnx` (fixed 576/704 input) and `*_fp16.engine`. Nano/Small were dropped on 25 Nov: no speed gain over Medium on the Orin and lower accuracy |
| `.venv/` *(Orin only)* | Python 3.10: Jetson torch 2.8.0 (CUDA 12.6), rfdetr 1.12.1, onnx 1.17, onnxruntime 1.18.1 (newer ORT crashes on this Orin) |

YOLOv5l engine/ONNX are reused from `~/odin_test/yolo_trt/`, weights from `~/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/`.

## 1. Record (Go2)
Only the ROS master on the Go2 and the Odin1 driver on the AGX are needed (no ONS2, no chassis bridge):
`~/odin_nav_stack/drv0144/start_odin_remote.sh` on the Go2, autostart or `~/odin_test/start_odin_agx.sh go2` on the AGX.
The Odin1 must be on **USB 3** (`cat /sys/bus/usb/devices/*/speed` → 5000 for vendor 2207; a USB 2 cable gives 480).

Record the **undistorted** image (what the live detector sees; `/odin1/image/compressed` is the fisheye image).
It is raw, ~60 MB/s (3.6 GB per minute). One command, so it can be pasted as is:
```bash
docker run -it --rm --name rec --network host \
  -e ROS_MASTER_URI=http://192.168.123.18:11311 -e ROS_IP=192.168.123.18 -e ROS_HOSTNAME=192.168.123.18 \
  -v $HOME/bags:/bags -w /bags --entrypoint bash ons2:go2 \
  -c "source /opt/ros/noetic/setup.bash && exec rosbag record -b 4096 -O restaurant_$(date +%F_%H%M) /odin1/image/undistorted /odin1/image/compressed /odin1/odometry /tf /tf_static"
```
Ctrl+C once to stop → `~/bags/restaurant_<date>.bag`. Cover people near/far/partly hidden, backlight, empty scenes.

## 2. Frames
Inside the `ons2:go2` image the system Python cannot import cv2 (CUDA OpenCV mismatch) and the conda Python lacks the
`Cryptodome`/`gnupg` modules `rosbag` imports (only used for encrypted bags), so run the script with the conda Python
and empty stub modules, the bags mounted read-only, and stream the frames out (nothing is written on the Go2):
```bash
cat extract_frames.py | ssh unitree@<go2> 'docker run --rm -i -v $HOME/bags:/bags:ro --entrypoint bash ons2:go2 -c "
  source /opt/ros/noetic/setup.bash >/dev/null; S=/tmp/stubs; mkdir -p \$S/Cryptodome/Cipher \$S/Cryptodome/Random
  touch \$S/Cryptodome/__init__.py; echo AES=None > \$S/Cryptodome/Cipher/__init__.py
  echo get_random_bytes=None > \$S/Cryptodome/Random/__init__.py; echo > \$S/gnupg.py; cat > /tmp/x.py
  PYTHONPATH=\$S:\$PYTHONPATH /opt/conda/envs/neupan/bin/python /tmp/x.py --topic /odin1/image/undistorted \
    --every 1.0 --out /tmp/frames_label /bags/*.bag >&2; tar -C /tmp -cf - frames_label"' \
  | ssh <orin> 'tar -C ~/odin_test/detbench/data -xf -'
```
`--every 1.0` → labelling set; `--every 0 --min-change 0` → every frame (continuous clip for speed only).

## 3. Labels
1. Pre-label on the Orin with both families, so neither model's blind spots go unnoticed:
   `.venv/bin/python prelabel.py --images data/frames_label --out data/prelabels_person.json --weights ~/odin_test/yolo_trt/yolov5l_fp16.engine models/rfdetr_medium_fp16.engine`
2. `python3 labelstudio.py to-tasks prelabels_person.json --image-dir frames_label` → tasks + config.
3. Label Studio (`uv tool install label-studio --with "sentry-sdk<3"`: label-studio 1.23.2 crashes with sentry-sdk 3):
   `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$PWD label-studio start`.
   Project: paste `label_studio_config.xml`; Cloud Storage → Local files → absolute path of `frames_label` (do **not**
   sync); Import `label_studio_tasks.json`. Review **every** frame: add missed people, delete wrong boxes, Submit (also
   empty frames). Export → JSON.
4. `python3 labelstudio.py to-coco project-*.json --image-dir frames_label --out labels_person.json` (refuses while a
   frame is unsubmitted or missing).

Rules used: reflections, posters, humanoid robots and the mechanical legs on the trolley are **not** `person`.

## 4. Run
Orin:
```bash
cd ~/odin_test/detbench
./run_orin.sh data/frames_label data/labels_person.json                       # whole matrix
.venv/bin/python bench.py --images F --labels L --weights models/rfdetr_medium_fp16.engine --conf 0.5   # one run
```
Keep power mode MODE_30W. Stop the robot services first for clean timings, or start Florence and `--note "with florence"`.

Go2 CPU: copy `bench.py`, `models/rfdetr_*.onnx`, `~/odin_test/yolo_trt/yolov5l.onnx`, frames and labels. In an env with
torch + the yolov5 repo (the ONS2 container) and `pip install onnxruntime==1.18.1` (or 1.17.3 on Python 3.8):
```bash
python3 bench.py --images F --labels L --weights yolov5l.pt --device cpu --yolov5-dir ~/ons2/yolov5 --hw "Go2 CPU"
python3 bench.py --images F --labels L --weights yolov5l.onnx --family yolov5 --yolov5-dir ~/ons2/yolov5 --hw "Go2 CPU"
python3 bench.py --images F --labels L --weights rfdetr_medium.onnx --hw "Go2 CPU"
```

## 5. Results — restaurant, 25 Nov 2026, Orin (MODE_30W, TensorRT FP16 unless noted)
138 frames (66 with people, 72 empty), 121 people. Full rows in `results/2026-11-25_orin.csv`.

| Model | Time/frame | FPS | Recall / precision @0.5 | @0.4 | AP50 |
|---|---|---|---|---|---|
| YOLOv5l (live today) | 54 ms | 18.4 | 0.909 / 0.940 | **0.959 / 0.928** | **0.971** |
| YOLOv5l PyTorch FP16 | 68 ms | 14.7 | — | — | 0.971 |
| RF-DETR Medium | **32 ms** | **31.6** | 0.917 / 0.854 | 0.942 / 0.745 | 0.945 |
| RF-DETR Large | 41 ms | 24.2 | 0.917 / 0.841 | 0.942 / 0.740 | 0.941 |
| any model + 4 tiles | 120–203 ms | 4.9–8.3 | precision 0.34–0.55 @0.25 | | 0.65–0.79 |

Reading: YOLOv5l is the most accurate and already above the camera rate (10 Hz); RF-DETR Medium is ~1.7× faster with
lower precision at the same recall; Large adds latency but no accuracy here; tiling is slower and worse.

Caveats: the ground truth was made by correcting the pre-labels and no person was added by hand, so recall is an upper
bound for every model (people missed by both models are not counted). The kept boxes are unchanged model boxes (78 from RF-DETR
Medium, 43 from YOLOv5l), so AP50-95 (box tightness) favours those two models — compare AP50 and recall/precision instead. Small set (1 person ≈ 0.8 % recall),
one site, GPU clocks not locked, Go2 CPU and "with Florence" not measured yet.

## 6. Errors of one model
```bash
.venv/bin/python errors.py --images data/frames_label --labels data/labels_person.json \
    --weights models/rfdetr_large_fp16.engine --conf 0.5 --out data/errors_rfdetr_large
```
`vis/`: green = found, red = wrong box (+ score), yellow = labelled person it missed. `errors.csv`: one row per box.
`predictions_coco.json` → `labelstudio.py to-tasks ... --config ""` to view the model's boxes in Label Studio.
On 25 Nov, RF-DETR Large @0.5 missed 10 / 121 (seated people boxed differently, hands at the frame edge, dark corner)
and made 21 wrong boxes, about half on the mechanical legs on the trolley; YOLOv5l @0.4 missed 5 and made 9.

## Updating the Orin
`~/odin_test/detbench/update_from_github.sh` (first run creates `~/Odin-Nav-Stack2`). Edit code on GitHub / your
laptop, not in the clone: the update resets it to GitHub.

## Notes
- Latency = decoded frame in memory → boxes in original pixels (JPEG decode excluded), after 10 warm-up runs.
- Accuracy: one extra untimed pass at conf 0.001 per frame; recall/precision read at `--conf` (0.25 by default, as live).
- `--tiles 4`: 2x2 tiles with 20 % overlap, merged by class-wise NMS; `--tiles 5` adds the full frame.
- YOLOv5 uses the live letterbox 640; RF-DETR stretches the frame to its square resolution (as `rfdetr` does).
