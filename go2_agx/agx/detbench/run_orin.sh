#!/bin/bash
# ORIN. Full benchmark matrix on the labelled set -> results/results.csv
#   ~/odin_test/detbench/run_orin.sh FRAMES_DIR LABELS.json [extra bench.py args, e.g. --limit 20]
# Run with the robot stack stopped (no Odin/YOLO/Florence containers) unless you want "under load" numbers;
# then add --note "with florence" and run the Florence service first.
set -e
cd "$(dirname "$0")"
IMG=${1:?frames folder}; LAB=${2:?COCO labels json}; shift 2
V=.venv/bin/python; M=models
Y=~/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models
B() { $V -W ignore bench.py --images "$IMG" --labels "$LAB" "$@" 2>&1 | grep -vE "\[TRT\]|^YOLOv5. summary"; }
echo "power mode: $(nvpmodel -q 2>/dev/null | head -1)"
# YOLOv5l: PyTorch FP16 (today's fallback) and TensorRT FP16 (today's live detector)
B --weights $Y/yolov5l.pt "$@"
B --weights ~/odin_test/yolo_trt/yolov5l_fp16.engine "$@"
B --weights ~/odin_test/yolo_trt/yolov5l_fp16.engine --tiles 4 "$@"
# RF-DETR (COCO-pretrained), TensorRT FP16
for v in nano small medium; do
  [ -f $M/rfdetr_${v}_fp16.engine ] || { echo "missing $M/rfdetr_${v}_fp16.engine (build_engines.sh)"; continue; }
  B --weights $M/rfdetr_${v}_fp16.engine "$@"
  B --weights $M/rfdetr_${v}_fp16.engine --tiles 4 "$@"
done
echo; column -s, -t < results/results.csv | cut -c1-260
