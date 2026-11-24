#!/bin/bash
# ORIN. FP16 TensorRT engines for the RF-DETR ONNX files (fixed shape). ~5-10 min each, robot idle.
cd ~/odin_test/detbench/models
for v in ${@:-nano small medium}; do
  [ -f rfdetr_${v}_fp16.engine ] && { echo "skip $v (engine exists)"; continue; }
  echo "== $v $(date +%T)"
  /usr/src/tensorrt/bin/trtexec --onnx=rfdetr_${v}.onnx --saveEngine=rfdetr_${v}_fp16.engine --fp16 \
    --memPoolSize=workspace:2048 > build_${v}.log 2>&1 && echo "OK $v $(date +%T) $(grep -m1 "GPU Compute Time" build_${v}.log)" || { echo "FAILED $v"; tail -5 build_${v}.log; }
done
df -h / | tail -1
