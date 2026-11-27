#!/bin/bash
# AGX ORIN. GPU object detector for ONS2 (replaces their CPU YOLOv5s detector on the Go2): a light detector + an optional
# tracker, selectable. Two parts:
#   1. GPU service detector/detector_service.py (127.0.0.1:8750, Odin-perception CUDA venv; container odin-gpu or host),
#      process "yolosvc": the detector (YOLOv5l TensorRT/PyTorch, RF-DETR TensorRT or YOLOE open vocabulary) and the tracker
#   2. ROS node detector/detector_node.py, process "yolo" in the container odin-agx (node /yolo_detector): publishes the
#      upstream yolo_ros topics (/yolo_detections(_3d), /yolo_class_names, /yolo_debug_image) + /perception/objects (JSON, all
#      results with track IDs and 3D) and /perception/target_3d
#
#   ~/odin_test/start_yolo_gpu_agx.sh [--detector yolov5|rfdetr|yoloe] [--tracker none|iou|botsort|botsort_reid|sam2]
#                                     [--weights PATH[,FALLBACK...]] [--conf 0.25]
# Defaults come from ~/.config/go2_agx/detector.env (DETECTOR, TRACKER, DETECTOR_WEIGHTS, DETECTOR_CONF; see
# go2_agx/detector.env.example), else YOLOv5l (TensorRT engine, PyTorch fallback) without tracker. Command-line options
# are for this start only: the watchdog and the autostart start it without options, so put a lasting choice in detector.env.
# On the Go2 start ONS2 with:  ~/ons2_docker/ons2_start.sh --with-yolo-agx   (keeps object_query + tf_relay there).
# Needs start_odin_agx.sh go2 and start_depth_agx.sh first.     Stop: ~/odin_test/stop_yolo_gpu_agx.sh
D=$HOME/odin_test; DET=$D/detector; mkdir -p $D/logs; . $D/agx_lib.sh
CONF_FILE=${DETECTOR_ENV:-$HOME/.config/go2_agx/detector.env}; [ -f "$CONF_FILE" ] && . "$CONF_FILE"
while [ $# -gt 0 ]; do
  case $1 in
    --detector) DETECTOR=$2; shift 2 ;;
    --tracker)  TRACKER=$2; shift 2 ;;
    --weights)  DETECTOR_WEIGHTS=$2; shift 2 ;;
    --conf)     DETECTOR_CONF=$2; shift 2 ;;
    *) sed -n 10,11p "$0"; exit 1 ;;
  esac
done
DETECTOR=${DETECTOR:-yolov5}; TRACKER=${TRACKER:-none}; DETECTOR_CONF=${DETECTOR_CONF:-0.25}
# Older settings still work for YOLOv5: YOLO_TRT_ENGINE / YOLO_WEIGHTS, YOLO_BACKEND=pytorch (skip the engine)
if [ "$DETECTOR" = yolov5 ] && [ -z "${DETECTOR_WEIGHTS:-}" ]; then
  PT=${YOLO_WEIGHTS:-$D/yolo/ros_ws/src/yolo_ros/scripts/models/yolov5l.pt}
  if [ "${YOLO_BACKEND:-tensorrt}" = pytorch ]; then DETECTOR_WEIGHTS=$PT
  else DETECTOR_WEIGHTS=${YOLO_TRT_ENGINE:-$D/yolo_trt/yolov5l_fp16.engine},$PT; fi
fi
agx_alive driver || { echo "First run: ~/odin_test/start_odin_agx.sh go2"; exit 1; }
agx_alive depth || echo "WARNING: no depth (start_depth_agx.sh): detections will have no 3D position"

# 1. GPU service (restarted so a new model/code/selection is always picked up)
SVC_PAT="odin_test/(detector/detector_service|yolo_gpu/yolo_gpu_service|yolo_trt/yolo_trt_service)[.]py"   # + the pre-detector services
if gpu_in_container; then SVC_LOG=$(gpu_log yolosvc); gpu_ensure || { echo "Container $GPU_CTR failed to start"; exit 1; }
else SVC_LOG=$D/logs/yolo_gpu_service.log; fi
svc_stop() {
  if gpu_in_container; then gpu_stop_proc yolosvc; pkill -u $(id -u) -f "$SVC_PAT"   # a copy left on the host would keep the port
  else pkill -f "$SVC_PAT"; fi
  for i in $(seq 1 10); do pgrep -f "$SVC_PAT" >/dev/null || break; sleep 1; done
}
svc_up() { if gpu_in_container; then gpu_alive yolosvc; else pgrep -f "$SVC_PAT" >/dev/null; fi; }
svc_stop
ENVS="DETECTOR=$DETECTOR TRACKER=$TRACKER DETECTOR_WEIGHTS=${DETECTOR_WEIGHTS:-} DETECTOR_PORT=8750 PIP_NO_INDEX=1"
export YOLOE_CLASSES=${YOLOE_CLASSES:-}   # may contain spaces: inherited in host mode, quoted for the container
echo "Detector $DETECTOR, tracker $TRACKER${DETECTOR_WEIGHTS:+, weights $DETECTOR_WEIGHTS}"
if gpu_in_container; then
  gpu_start yolosvc "cd $DET && exec env $ENVS YOLOE_CLASSES='$YOLOE_CLASSES' $(gpu_perc)/.venv/bin/python $DET/detector_service.py" \
    || { echo "GPU service failed: tail $SVC_LOG"; exit 1; }
else   # fully detached (own session, no fd shared with this script) so this script can always exit
  (cd $DET && exec env $ENVS setsid $HOME/odin-perception/.venv/bin/python $DET/detector_service.py > $SVC_LOG 2>&1 < /dev/null) &
  disown
fi
ok=0; for i in $(seq 1 120); do sleep 1; curl -sf -m 2 http://127.0.0.1:8750/health >/dev/null && { ok=1; break; }; svc_up || break; done
[ $ok = 1 ] || { echo "GPU service failed: tail $SVC_LOG"; tail -8 $SVC_LOG; exit 1; }
echo "OK: $(grep -m1 "Detector service" $SVC_LOG)$(gpu_in_container && echo "  [container $GPU_CTR]")"

# 2. ROS node
agx_cp $DET/detector_node.py /detector_node.py && agx_cp $DET/depth3d.py /depth3d.py
agx_start yolo "python3 /detector_node.py __name:=yolo_detector \\
  _rgb_topic:=/odin1/image/undistorted _depth_topic:=/odin1/depth_img_competetion _camera_frame:=camera_link _conf_thres:=$DETECTOR_CONF"
for i in $(seq 1 30); do sleep 1; grep -q "initialized with" $(agx_log yolo) 2>/dev/null && break; done
if grep -q "initialized with" $(agx_log yolo) 2>/dev/null; then
  echo "OK: detector running on the AGX GPU (process yolo in $AGX_CTR). Go2: ons2_start.sh --with-yolo-agx"
else
  echo "Detector node not ready: tail $(agx_log yolo)"; tail -8 $(agx_log yolo) 2>&1; exit 1
fi
