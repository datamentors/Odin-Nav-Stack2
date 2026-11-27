#!/bin/bash
# Build the TensorRT FP16 engine of YOLOv5l inside the container odin-gpu (its own TensorRT; the host needs none).
# Run it on the Go2 with the robot idle: about 15 minutes of GPU. Needs the image odin-gpu:go2 and internet only for the weights.
#   1. weights (get_weights.sh)  2. .pt -> ONNX (export_onnx.py, in the Odin-perception venv)  3. trtexec -> yolov5l_fp16.engine
# The engine only loads with the TensorRT version that built it: rebuild it after a JetPack / image update.
#   ~/odin_test/yolo_trt/build_engine.sh        output: ~/odin_test/yolo_trt/yolov5l_fp16.engine (the old one is kept as .bak)
set -euo pipefail
D=$HOME/odin_test; . $D/agx_lib.sh
T=$D/yolo_trt
"$(dirname "$0")/get_weights.sh"
gpu_ensure || { echo "container $GPU_CTR did not start (image $GPU_IMAGE built?)"; exit 1; }
echo "== 1/2 .pt -> ONNX"
docker exec -e HOME=$HOME $GPU_CTR $(gpu_perc)/.venv/bin/python $T/export_onnx.py
echo "== 2/2 trtexec (about 15 minutes)"
docker exec -e HOME=$HOME $GPU_CTR bash -c "cd $T && /usr/src/tensorrt/bin/trtexec --onnx=yolov5l.onnx --saveEngine=yolov5l_fp16.engine.new --fp16 \
  --minShapes=images:1x3x256x256 --optShapes=images:1x3x544x640 --maxShapes=images:1x3x640x640 --memPoolSize=workspace:2048" > $T/build.log 2>&1 \
  || { echo "trtexec failed: tail $T/build.log"; tail -5 $T/build.log; exit 1; }
docker exec $GPU_CTR chown -R $(id -u):$(id -g) $T      # the container runs as root
[ -f $T/yolov5l_fp16.engine ] && mv $T/yolov5l_fp16.engine $T/yolov5l_fp16.engine.bak
mv $T/yolov5l_fp16.engine.new $T/yolov5l_fp16.engine
echo "OK: $T/yolov5l_fp16.engine ($(du -h $T/yolov5l_fp16.engine | cut -f1)). Start: ~/odin_test/start_yolo_gpu_agx.sh"
