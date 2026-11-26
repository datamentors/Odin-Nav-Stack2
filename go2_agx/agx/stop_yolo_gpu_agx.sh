#!/bin/bash
# AGX ORIN. Stops the YOLOv5l GPU detector (process yolo in the container odin-agx + GPU service: TensorRT or PyTorch,
# in the container odin-gpu or on the host). The container odin-gpu keeps running (stop_odin_agx.sh all removes it).
D=$HOME/odin_test; G=$D/yolo_gpu; . $D/agx_lib.sh
agx_stop_proc yolo
gpu_stop_proc yolosvc   # in the container odin-gpu (if it exists)
pkill -f "odin_test/yolo_(gpu/yolo_gpu|trt/yolo_trt)_service[.]py"; rm -f $G/service.pid   # on the host (host mode)
echo "Stopped. (Back to the Go2 CPU YOLO: on the Go2 start ONS2 with --with-yolo instead of --with-yolo-agx)"
