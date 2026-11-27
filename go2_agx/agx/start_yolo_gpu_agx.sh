#!/bin/bash
# AGX ORIN. GPU object detector for ONS2: a selectable light detector + optional tracker. Guide (models, topics):
# ~/odin_test/detector/README.md. Two processes:
#   yolosvc  detector/detector_service.py (Odin-perception CUDA venv; container odin-gpu, or the host), 127.0.0.1:$DET_PORT
#   yolo     detector/detector_node.py in the container odin-agx (ROS node /yolo_detector)
# Selection: ~/.config/go2_agx/detector.env (template go2_agx/detector.env.example); the options below override it for this
# start only (the watchdog and autostart start without options). Older YOLO_WEIGHTS / YOLO_TRT_ENGINE / YOLO_BACKEND=pytorch
# still pick the YOLOv5 weights. Needs start_odin_agx.sh and start_depth_agx.sh first; Go2: ons2_start.sh --with-yolo-agx.
# Stop: ~/odin_test/stop_yolo_gpu_agx.sh
usage() {
  echo "usage: $0 [--detector yolov5|rfdetr|yoloe] [--tracker none|iou|botsort|botsort_reid|sam2]"
  echo "          [--weights PATH[,FALLBACK...]] [--conf 0.25] [--scale PERCENT]"
  exit 1
}
D=$HOME/odin_test; DET=$D/detector; mkdir -p $D/logs; . $D/agx_lib.sh
CONF_FILE=${DETECTOR_ENV:-$HOME/.config/go2_agx/detector.env}; [ -f "$CONF_FILE" ] && . "$CONF_FILE"
while [ $# -gt 0 ]; do
  case $1 in
    --detector) DETECTOR=$2; shift 2 ;;
    --tracker)  TRACKER=$2; shift 2 ;;
    --weights)  DETECTOR_WEIGHTS=$2; shift 2 ;;
    --conf)     DETECTOR_CONF=$2; shift 2 ;;
    --scale)    DETECTOR_SCALE=$2; shift 2 ;;
    *) usage ;;
  esac
done
DETECTOR=${DETECTOR:-yolov5}; TRACKER=${TRACKER:-none}; DETECTOR_CONF=${DETECTOR_CONF:-0.25}; DETECTOR_SCALE=${DETECTOR_SCALE:-0}
# Older YOLOv5 settings, only when one is set (otherwise detectors.py picks its default weights)
if [ "$DETECTOR" = yolov5 ] && [ -z "${DETECTOR_WEIGHTS:-}" ] && [ -n "${YOLO_WEIGHTS:-}${YOLO_TRT_ENGINE:-}${YOLO_BACKEND:-}" ]; then
  PT=${YOLO_WEIGHTS:-$D/yolo/ros_ws/src/yolo_ros/scripts/models/yolov5l.pt}
  if [ "${YOLO_BACKEND:-}" = pytorch ]; then DETECTOR_WEIGHTS=$PT
  else DETECTOR_WEIGHTS=${YOLO_TRT_ENGINE:-$D/yolo_trt/yolov5l_fp16.engine},$PT; fi
fi
agx_alive driver || { echo "First run: ~/odin_test/start_odin_agx.sh go2"; exit 1; }
agx_alive depth || echo "WARNING: no depth (start_depth_agx.sh): detections will have no 3D position"

# 1. GPU service (restarted so a new model/code/selection is always picked up)
if gpu_in_container; then SVC_LOG=$(gpu_log yolosvc); gpu_ensure || { echo "Container $GPU_CTR failed to start"; exit 1; }
else SVC_LOG=$D/logs/yolo_gpu_service.log; fi
svc_stop() {
  if gpu_in_container; then gpu_stop_proc yolosvc; pkill -u $(id -u) -f "$DET_SVC_PAT"   # a copy left on the host would keep the port
  else pkill -f "$DET_SVC_PAT"; fi
  for i in $(seq 1 10); do pgrep -f "$DET_SVC_PAT" >/dev/null || break; sleep 1; done
}
svc_up() { if gpu_in_container; then gpu_alive yolosvc; else pgrep -f "$DET_SVC_PAT" >/dev/null; fi; }
svc_stop
# Settings for the service; optional ones only when set (the Python code has the defaults). Values may contain spaces
# (YOLOE_CLASSES): exported for the host, single-quoted for the container command.
ENVS="DETECTOR=$DETECTOR TRACKER=$TRACKER DETECTOR_PORT=$DET_PORT PIP_NO_INDEX=1"
for v in DETECTOR_WEIGHTS YOLOE_CLASSES YOLOE_IMGSZ SAM2_IMAGE_SIZE SAM2_MODEL SAM2_MODE FOCUS_CONF ACQUIRE_CONF \
         REACQUIRE_FRAMES BOTSORT_HIGH BOTSORT_NEW; do
  [ -n "${!v:-}" ] && { export "$v"; ENVS="$ENVS $v='${!v}'"; }
done
echo "Detector $DETECTOR, tracker $TRACKER${DETECTOR_WEIGHTS:+, weights $DETECTOR_WEIGHTS}, model image $([ "$DETECTOR_SCALE" = 0 ] && echo auto || echo "$DETECTOR_SCALE %")"
if gpu_in_container; then
  gpu_start yolosvc "cd $DET && exec env $ENVS $(gpu_perc)/.venv/bin/python $DET/detector_service.py" \
    || { echo "GPU service failed: tail $SVC_LOG"; exit 1; }
else   # fully detached (own session, no fd shared with this script) so this script can always exit
  (cd $DET && export DETECTOR TRACKER DETECTOR_PORT=$DET_PORT PIP_NO_INDEX=1 \
     && exec setsid $HOME/odin-perception/.venv/bin/python $DET/detector_service.py > $SVC_LOG 2>&1 < /dev/null) &
  disown
fi
ok=0; for i in $(seq 1 120); do sleep 1; curl -sf -m 2 http://127.0.0.1:$DET_PORT/health >/dev/null && { ok=1; break; }; svc_up || break; done
[ $ok = 1 ] || { echo "GPU service failed: tail $SVC_LOG"; tail -8 $SVC_LOG; exit 1; }
echo "OK: $(grep -m1 "Detector service" $SVC_LOG)$(gpu_in_container && echo "  [container $GPU_CTR]")"

# 2. ROS node (topics and frames: the node's defaults)
agx_cp $DET/detector_node.py /detector_node.py && agx_cp $DET/depth3d.py /depth3d.py
agx_start yolo "python3 /detector_node.py __name:=yolo_detector _service_url:=http://127.0.0.1:$DET_PORT \\
  _conf_thres:=$DETECTOR_CONF _scale_percent:=$DETECTOR_SCALE"
for i in $(seq 1 30); do sleep 1; grep -q "initialized with" $(agx_log yolo) 2>/dev/null && break; done
if grep -q "initialized with" $(agx_log yolo) 2>/dev/null; then
  echo "OK: detector running on the AGX GPU (process yolo in $AGX_CTR). Go2: ons2_start.sh --with-yolo-agx"
else
  echo "Detector node not ready: tail $(agx_log yolo)"; tail -8 $(agx_log yolo) 2>&1; exit 1
fi
