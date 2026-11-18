#!/bin/bash
# AGX ORIN. Stops the YOLOv5l GPU detector (container odin-yolo-gpu + GPU service: TensorRT or PyTorch).
G=$HOME/odin_test/yolo_gpu
docker rm -f odin-yolo-gpu >/dev/null 2>&1
pkill -f "odin_test/yolo_(gpu/yolo_gpu|trt/yolo_trt)_service[.]py"; rm -f $G/service.pid
echo "Stopped. (Back to the Go2 CPU YOLO: on the Go2 start ONS2 with --with-yolo instead of --with-yolo-agx)"
