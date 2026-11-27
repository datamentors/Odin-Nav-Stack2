#!/bin/bash
# AGX ORIN. YOLOv5l on the AGX GPU for the ONS2 (replaces their CPU YOLOv5s detector on the Go2).
#   1. GPU service (Odin-perception CUDA venv, 127.0.0.1:8750): YOLOv5l forward pass + NMS. Since 18 Nov the TensorRT
#      FP16 engine (yolo_trt/yolo_trt_service.py, ~50 ms/image, same detections); fallback PyTorch yolo_gpu_service.py
#   2. process "yolo" in the container odin-agx (ROS 1): their yolo_detector with the model part sent to (1);
#      publishes the same topics as on the Go2: /yolo_detections(_3d), /yolo_debug_image/compressed, /yolo_class_names
# On the Go2 start ONS2 with:  ~/ons2_docker/ons2_start.sh --with-yolo-agx   (keeps object_query + tf_relay there).
# Needs start_odin_agx.sh go2 and start_depth_agx.sh first.     Stop: ~/odin_test/stop_yolo_gpu_agx.sh
D=$HOME/odin_test; G=$D/yolo_gpu; mkdir -p $D/logs; . $D/agx_lib.sh
agx_alive driver || { echo "First run: ~/odin_test/start_odin_agx.sh go2"; exit 1; }
agx_alive depth || echo "WARNING: no depth (start_depth_agx.sh): detections will have no 3D position"
# 1. GPU service on 127.0.0.1:8750 (restarted so a new model/code is always picked up). 18 Nov: TensorRT FP16 engine
#    (~50 ms/image instead of ~300-430 ms with PyTorch, same detections: ~/odin_test/yolo_trt/compare.py) when the engine
#    exists; if it is missing or does not start, the PyTorch service as before. Force PyTorch: YOLO_BACKEND=pytorch
T=$D/yolo_trt; SVC_PAT="odin_test/yolo_(gpu/yolo_gpu|trt/yolo_trt)_service[.]py"
# The service runs in the container odin-gpu when its image exists (see agx_lib.sh), otherwise on the host
if gpu_in_container; then SVC_LOG=$(gpu_log yolosvc); gpu_ensure || { echo "Container $GPU_CTR failed to start"; exit 1; }
else SVC_LOG=$D/logs/yolo_gpu_service.log; fi
svc_stop() {
  if gpu_in_container; then gpu_stop_proc yolosvc; pkill -u $(id -u) -f "$SVC_PAT"   # a copy left on the host would keep the port
  else pkill -f "$SVC_PAT"; fi
  for i in $(seq 1 10); do pgrep -f "$SVC_PAT" >/dev/null || break; sleep 1; done
}
svc_up() { if gpu_in_container; then gpu_alive yolosvc; else pgrep -f "$SVC_PAT" >/dev/null; fi; }
svc_stop
start_svc() {   # $1 folder  $2 script. Fully detached (own session, no fd shared with this script) so this script can always exit
  if gpu_in_container; then
    gpu_start yolosvc "cd $1 && exec env YOLO_GPU_PORT=8750 PIP_NO_INDEX=1 $(gpu_perc)/.venv/bin/python $1/$2" || return 1
  else
    (cd $1 && exec env YOLO_GPU_PORT=8750 PIP_NO_INDEX=1 setsid $HOME/odin-perception/.venv/bin/python $1/$2 > $SVC_LOG 2>&1 < /dev/null) &
    disown; sleep 2; pgrep -f "$SVC_PAT" | head -1 > $G/service.pid
  fi
  for i in $(seq 1 30); do sleep 1; curl -sf -m 2 http://127.0.0.1:8750/health >/dev/null && return 0; svc_up || return 1; done
  return 1
}
if [ "${YOLO_BACKEND:-tensorrt}" = tensorrt ] && [ -f $T/yolov5l_fp16.engine ] && start_svc $T yolo_trt_service.py; then
  :
else
  if [ "${YOLO_BACKEND:-tensorrt}" = tensorrt ]; then
    echo "TensorRT service not available, using PyTorch. Its log:"; tail -3 $SVC_LOG
    svc_stop; sleep 2
  fi
  start_svc $G yolo_gpu_service.py || { echo "GPU service failed: tail $SVC_LOG"; tail -5 $SVC_LOG; exit 1; }
fi
echo "OK: $(grep -m1 -E "YOLO (GPU|TensorRT) service" $SVC_LOG)$(gpu_in_container && echo "  [container $GPU_CTR]")"
# 2. ROS detector (same params as their yolo_detection.launch)
agx_cp $G/yolo_detector_gpu.py /yolo_detector_gpu.py
agx_start yolo "python3 /yolo_detector_gpu.py __name:=yolo_detector \\
  _rgb_topic:=/odin1/image/undistorted _depth_topic:=/odin1/depth_img_competetion _auto_nav_mode:=false _camera_frame:=camera_link \\
  _target_frame:=map _use_latest_tf:=true _depth_mode:=z _conf_thres:=0.25"
for i in $(seq 1 30); do sleep 1; grep -q "initialized with" $(agx_log yolo) 2>/dev/null && break; done
if grep -q "initialized with" $(agx_log yolo) 2>/dev/null; then
  echo "OK: YOLO detector running on the AGX GPU (process yolo in $AGX_CTR). Go2: ons2_start.sh --with-yolo-agx"
else
  echo "Detector not ready: tail $(agx_log yolo)"; tail -8 $(agx_log yolo) 2>&1; exit 1
fi
