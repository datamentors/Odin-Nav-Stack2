#!/bin/bash
# AGX ORIN. Stops the YOLOv5l GPU detector (process yolo in the container odin-agx + GPU service: TensorRT or PyTorch).
D=$HOME/odin_test; G=$D/yolo_gpu; . $D/agx_lib.sh
agx_stop_proc yolo
pkill -f "odin_test/yolo_(gpu/yolo_gpu|trt/yolo_trt)_service[.]py"; rm -f $G/service.pid
echo "Stopped. (Back to the Go2 CPU YOLO: on the Go2 start ONS2 with --with-yolo instead of --with-yolo-agx)"
