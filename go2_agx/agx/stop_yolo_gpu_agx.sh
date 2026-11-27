#!/bin/bash
# AGX ORIN. Stops the GPU detector (process yolo in the container odin-agx + the detector service, in the container
# odin-gpu or on the host). The container odin-gpu keeps running (stop_odin_agx.sh all removes it).
D=$HOME/odin_test; . $D/agx_lib.sh
agx_stop_proc yolo
gpu_stop_proc yolosvc   # in the container odin-gpu (if it exists)
pkill -f "odin_test/(detector/detector_service|yolo_gpu/yolo_gpu_service|yolo_trt/yolo_trt_service)[.]py"   # on the host (host mode)
echo "Stopped. (Back to the Go2 CPU YOLO: on the Go2 start ONS2 with --with-yolo instead of --with-yolo-agx)"
