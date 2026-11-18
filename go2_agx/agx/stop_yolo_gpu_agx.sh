#!/bin/bash
# AGX ORIN. Stops the YOLOv5l GPU detector (container odin-yolo-gpu + yolo_gpu_service.py).
G=$HOME/odin_test/yolo_gpu
docker rm -f odin-yolo-gpu >/dev/null 2>&1
pkill -f "odin_test/yolo_gpu/yolo_gpu_service[.]py"; rm -f $G/service.pid
echo "Stopped. (Back to the Go2 CPU YOLO: on the Go2 start ONS2 with --with-yolo instead of --with-yolo-agx)"
