#!/bin/bash
# AGX ORIN. YOLOv5l on the AGX GPU for the ONS2 (replaces their CPU YOLOv5s detector on the Go2).
#   1. GPU service (Odin-perception CUDA venv, 127.0.0.1:8750): YOLOv5l forward pass + NMS. Since 18 Nov the TensorRT
#      FP16 engine (yolo_trt/yolo_trt_service.py, ~50 ms/image, same detections); fallback PyTorch yolo_gpu_service.py
#   2. container "odin-yolo-gpu" (image odin-yolo:0144, ROS 1): their yolo_detector with the model part sent to (1);
#      publishes the same topics as on the Go2: /yolo_detections(_3d), /yolo_debug_image/compressed, /yolo_class_names
# On the Go2 start ONS2 with:  ~/ons2_docker/ons2_start.sh --with-yolo-agx   (keeps object_query + tf_relay there).
# Needs start_odin_agx.sh go2 and start_depth_agx.sh first.     Stop: ~/odin_test/stop_yolo_gpu_agx.sh
CONF="${GO2_AGX_CONF:-$HOME/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
: "${AGX_IP:?set AGX_IP (AGX address on the cable to the Go2) in $CONF}"
D=$HOME/odin_test; G=$D/yolo_gpu; mkdir -p $D/logs
docker ps --format "{{.Names}}" | grep -qx odin-agx || { echo "First run: ~/odin_test/start_odin_agx.sh go2"; exit 1; }
docker ps --format "{{.Names}}" | grep -qx odin-depth || echo "WARNING: no depth (start_depth_agx.sh): detections will have no 3D position"
# 1. GPU service on 127.0.0.1:8750 (restarted so a new model/code is always picked up). 18 Nov: TensorRT FP16 engine
#    (~50 ms/image instead of ~300-430 ms with PyTorch, same detections: ~/odin_test/yolo_trt/compare.py) when the engine
#    exists; if it is missing or does not start, the PyTorch service as before. Force PyTorch: YOLO_BACKEND=pytorch
T=$D/yolo_trt; SVC_PAT="odin_test/yolo_(gpu/yolo_gpu|trt/yolo_trt)_service[.]py"
pkill -f "$SVC_PAT"; for i in $(seq 1 10); do pgrep -f "$SVC_PAT" >/dev/null || break; sleep 1; done
start_svc() {   # $1 folder  $2 script. Fully detached (own session, no fd shared with this script) so this script can always exit
  (cd $1 && exec env YOLO_GPU_PORT=8750 PIP_NO_INDEX=1 setsid $HOME/odin-perception/.venv/bin/python $1/$2 > $D/logs/yolo_gpu_service.log 2>&1 < /dev/null) &
  disown; sleep 2; pgrep -f "$SVC_PAT" | head -1 > $G/service.pid
  for i in $(seq 1 30); do sleep 1; curl -sf -m 2 http://127.0.0.1:8750/health >/dev/null && return 0; pgrep -f "$SVC_PAT" >/dev/null || return 1; done
  return 1
}
if [ "${YOLO_BACKEND:-tensorrt}" = tensorrt ] && [ -f $T/yolov5l_fp16.engine ] && start_svc $T yolo_trt_service.py; then
  :
else
  if [ "${YOLO_BACKEND:-tensorrt}" = tensorrt ]; then
    echo "TensorRT service not available, using PyTorch. Its log:"; tail -3 $D/logs/yolo_gpu_service.log
    pkill -f "$SVC_PAT"; sleep 2
  fi
  start_svc $G yolo_gpu_service.py || { echo "GPU service failed: tail $D/logs/yolo_gpu_service.log"; tail -5 $D/logs/yolo_gpu_service.log; exit 1; }
fi
echo "OK: $(grep -m1 -E "YOLO (GPU|TensorRT) service" $D/logs/yolo_gpu_service.log)"
# 2. ROS detector (same params as their yolo_detection.launch)
docker rm -f odin-yolo-gpu >/dev/null 2>&1
docker run -d --name odin-yolo-gpu --network host -e ROS_MASTER_URI=http://${GO2_IP}:11311 -e ROS_IP=${AGX_IP} \
  -e ROS_HOSTNAME=${AGX_IP} -e PYTHONUNBUFFERED=1 -v $G/yolo_detector_gpu.py:/yolo_detector_gpu.py:ro -v $D/yolo/calib.yaml:/calib.yaml:ro --entrypoint bash \
  odin-yolo:0144 -c "source /opt/ros/noetic/setup.bash; exec python3 /yolo_detector_gpu.py __name:=yolo_detector \
  _rgb_topic:=/odin1/image/undistorted _depth_topic:=/odin1/depth_img_competetion _auto_nav_mode:=false _camera_frame:=camera_link \
  _target_frame:=map _use_latest_tf:=true _depth_mode:=z _conf_thres:=0.25" >/dev/null
for i in $(seq 1 30); do sleep 1; [ "$(docker logs odin-yolo-gpu 2>&1 | grep -c "initialized with")" -gt 0 ] && break; done
if [ "$(docker logs odin-yolo-gpu 2>&1 | grep -c "initialized with")" -gt 0 ]; then
  echo "OK: YOLO detector running on the AGX GPU (container odin-yolo-gpu). Go2: ons2_start.sh --with-yolo-agx"
else
  echo "Detector not ready: docker logs odin-yolo-gpu"; docker logs --tail 8 odin-yolo-gpu 2>&1; exit 1
fi
