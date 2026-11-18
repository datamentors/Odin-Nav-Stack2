#!/bin/bash
# AGX ORIN. YOLOv5l on the AGX GPU for the ONS2 (replaces their CPU YOLOv5s detector on the Go2).
#   1. yolo_gpu_service.py  (Odin-perception CUDA venv, 127.0.0.1:8750): YOLOv5l forward pass + NMS, ~60 ms/image
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
# 1. GPU service (restart it so a new model/code is always picked up)
pkill -f "odin_test/yolo_gpu/yolo_gpu_service[.]py"; for i in $(seq 1 10); do pgrep -f "yolo_gpu_service[.]py" >/dev/null || break; sleep 1; done
# fully detached (own session, no fd shared with this script) so this script can always exit
(cd $G && exec setsid $HOME/odin-perception/.venv/bin/python $G/yolo_gpu_service.py > $D/logs/yolo_gpu_service.log 2>&1 < /dev/null) &
disown; sleep 2; pgrep -f "yolo_gpu_service[.]py" | head -1 > $G/service.pid
for i in $(seq 1 30); do sleep 1; curl -sf -m 2 http://127.0.0.1:8750/health >/dev/null && break; done
curl -sf -m 2 http://127.0.0.1:8750/health >/dev/null || { echo "GPU service failed: tail $D/logs/yolo_gpu_service.log"; tail -5 $D/logs/yolo_gpu_service.log; exit 1; }
echo "OK: $(grep -m1 'YOLO GPU service' $D/logs/yolo_gpu_service.log)"
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
