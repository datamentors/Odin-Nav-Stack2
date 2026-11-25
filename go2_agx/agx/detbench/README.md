# detbench — YOLOv5l vs RF-DETR vs 4-tile, Go2 CPU vs Orin

| File | What |
|---|---|
| `extract_frames.py` | ROS 1 bags → evenly spaced, de-duplicated JPEGs for labelling (needs ROS Noetic python) |
| `bench.py` | The benchmark: latency (pre/infer/post/e2e, p50/p95), FPS, recall/precision @conf, AP50, AP50-95 per class. One CSV row per run |
| `run_orin.sh` | Full Orin matrix: YOLOv5l PyTorch + TensorRT, RF-DETR medium/large TensorRT, each with and without 2x2 tiles |
| `export_rfdetr.py`, `build_engines.sh` | RF-DETR → ONNX → TensorRT FP16 (already done) |
| `models/` | `rfdetr_{medium,large}.onnx` (fixed 576/704 input) and `*_fp16.engine`. Nano/Small were dropped on 25 Nov: no speed gain over Medium on the Orin and lower accuracy |
| `.venv/` | Python 3.10: Jetson torch 2.8.0 (CUDA 12.6), rfdetr 1.12.1, onnx 1.17, onnxruntime 1.18.1 (newer ORT crashes on this Orin) |
| `smoke/` | 9 old test frames + pseudo-labels, only to check the code runs |

YOLOv5l engine/ONNX are reused from `~/odin_test/yolo_trt/`, weights from `~/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/`.

## 1. Frames (Go2)
Record with the stack running: `rosbag record -O bench_$(date +%F_%H%M).bag /odin1/image/compressed`
(people near/far/partly hidden, bad light, empty scenes). Then in a ROS Noetic environment:
`python3 extract_frames.py --out bench_frames --every 1.5 *.bag` → aim for 150–200 frames, copy to the Orin.

## 2. Labels
CVAT or Label Studio, class `person` (add others if wanted), export **COCO 1.0** JSON. Same file names as the frames.
Decide up front: humanoid robots / mannequins / people on posters = not person (RF-DETR calls a humanoid robot a person).

## 3. Orin
```bash
cd ~/odin_test/detbench
./run_orin.sh ~/bench_frames ~/bench_frames/labels.json          # whole matrix
.venv/bin/python bench.py --images F --labels L --weights models/rfdetr_medium_fp16.engine --tiles 4   # one run
```
Keep power mode MODE_30W (same as the README numbers). Optional "under load" run: start Florence first, `--note "with florence"`.

## 4. Go2 CPU
Copy `bench.py`, `models/rfdetr_*.onnx`, `~/odin_test/yolo_trt/yolov5l.onnx`, frames and labels. In an env with
torch + the yolov5 repo (the ONS2 container) and `pip install onnxruntime==1.18.1` (or 1.17.3 on Python 3.8):
```bash
python3 bench.py --images F --labels L --weights yolov5l.pt --device cpu --yolov5-dir ~/ons2/yolov5 --hw "Go2 CPU"
python3 bench.py --images F --labels L --weights yolov5l.onnx --family yolov5 --yolov5-dir ~/ons2/yolov5 --hw "Go2 CPU"
python3 bench.py --images F --labels L --weights rfdetr_medium.onnx --hw "Go2 CPU"
```

## Notes
- Latency = decoded frame in memory → boxes in original pixels (JPEG decode excluded), after 10 warm-up runs.
- Accuracy: one extra untimed pass at conf 0.001 per frame; recall/precision read at `--conf` (0.25, as live).
- `--tiles 4`: 2x2 tiles with 20 % overlap, merged by class-wise NMS; `--tiles 5` adds the full frame.
- YOLOv5 uses the live letterbox 640; RF-DETR stretches the frame to its square resolution (as `rfdetr` does).
