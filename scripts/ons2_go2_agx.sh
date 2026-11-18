#!/usr/bin/env bash
# ONS2 Go2 launcher for OUR setup (runs INSIDE the ons2-go2 container; host ROS2 Humble untouched).
# Same as the official scripts/ons2_go2.sh (same tmux windows, gates, mission_manager args, console), except:
#   1. The Odin1 driver runs on the AGX Orin (start_odin_agx.sh go2), not here: the "odin" window only checks
#      its data. Gate 1 = /odin1/odometry streaming, Gate 2 = relocalized (TF map -> imu).
#   2. roscore: reuses the ROS master already running on the Go2 (odin-master) that the AGX is connected to.
#   3. Profile robots/go2_agx (copy of robots/go2). Chassis bridge only if ONS2_BRIDGE=1 (ons2_start.sh --with-bridge):
#      then THEIR bridge unitree_control (vel_to_sdk) runs as shipped: it STANDS THE ROBOT UP at start, FreeWalk,
#      no arm/disarm. Without it /cmd_vel is published but nothing moves the legs.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ONS2_ROBOT="go2_agx"
SESSION="ons2_go2"
LOG_DIR="${GO2_LOG_DIR:-${HOME}/go2_logs}"
mkdir -p "${LOG_DIR}"
NEUPAN_CONFIG="${REPO_ROOT}/robots/go2_agx/neupan/config.yaml"
ROBOT_PROFILE="${REPO_ROOT}/robots/go2_agx/robot.yaml"
source "${SCRIPT_DIR}/lib/ons2_lib.sh"
ons2_parse_common_args "$@"
ons2_validate_common
BRINGUP_ARGS="$(ons2_ros1_bringup_args "${ROBOT_PROFILE}")"
if [[ "${ONS2_BRIDGE:-0}" = "1" ]]; then
  BRIDGE_ARGS="bridge_pkg:=unitree_control bridge_launch:=unitree_control.launch"
  BRIDGE_TXT="WITH their chassis bridge unitree_control (the robot stands up and WALKS)"
  HEALTH_NODES="/map_server /pc_to_scan /sliding_map_node /vel_to_sdk /mission_manager"
else
  BRIDGE_ARGS="bridge_pkg:=ons2_bringup bridge_launch:=no_bridge.launch"
  BRIDGE_TXT="WITHOUT chassis bridge (the legs do not move)"
  HEALTH_NODES="/map_server /pc_to_scan /sliding_map_node /mission_manager"
fi
MAP_FILE_NAME="${MAP_NAME:-map_opt}"
if [[ -n "${MAP_NAME}" ]]; then
  [[ -f "${REPO_ROOT}/ros_ws/src/map_planner/maps/${MAP_NAME}.yaml" ]] || ons2_fail "--map=${MAP_NAME} missing navigation map"
  [[ -f "${REPO_ROOT}/ros_ws/src/map_planner/maps/${MAP_NAME}.pgm" ]] || ons2_fail "--map=${MAP_NAME} missing map image"
fi
ons2_source_ros1
ons2_require_tmux
ons2_new_session "${SESSION}" roscore
ENV_PREFIX="source /opt/ros/noetic/setup.bash && source ${REPO_ROOT}/ros_ws/devel/setup.bash && export GO2_ROOT=${REPO_ROOT} && export ROS_MASTER_URI=${ROS_MASTER_URI}"

ons2_log "w0 ROS master (shared with the AGX): ${ROS_MASTER_URI}"
if timeout 3 rosparam get /run_id >/dev/null 2>&1; then
  tmux send-keys -t "${SESSION}:roscore" "echo Using the existing ROS master ${ROS_MASTER_URI} (odin-master container)" C-m
else
  tmux send-keys -t "${SESSION}:roscore" "source /opt/ros/noetic/setup.bash && roscore 2>&1 | tee ${LOG_DIR}/roscore.log" C-m
  timeout 30 bash -c "until timeout 3 rosparam get /run_id >/dev/null 2>&1; do sleep 1; done" || ons2_fail "ROS Master not ready"
fi
rosparam set /global_planner "${GLOBAL_PLANNER}"
rosparam set /local_planner "${LOCAL_PLANNER}"
rosparam set /navigation_mode "${NAVIGATION_MODE}"
ons2_log "  [OK] ROS master ready"

tmux new-window -t "${SESSION}" -n odin "bash --norc"
tmux send-keys -t "${SESSION}:odin" "${ENV_PREFIX} && echo Odin1 driver runs on the AGX Orin. Odometry: && rostopic hz /odin1/odometry" C-m
ons2_log "w1 Odin1 on the AGX (start_odin_agx.sh go2)"
ons2_wait_for "python3 ${REPO_ROOT}/ros_ws/scripts/launch_gate.py --wait-topic /odin1/odometry --timeout 5" "ready" 60 "Gate 1: Odin1 data from the AGX (/odin1/odometry)"
ons2_wait_for "python3 ${REPO_ROOT}/ros_ws/scripts/launch_gate.py --wait-tf map imu --timeout 5" "tf-ready" 60 "Gate 2: Relocalized (TF map -> imu from the Odin1 driver)"

tmux new-window -t "${SESSION}" -n go2 "bash --norc"
tmux send-keys -t "${SESSION}:go2" \
  "${ENV_PREFIX} && roslaunch ons2_bringup go2.launch map_file_name:=${MAP_FILE_NAME} navigation_mode:=${NAVIGATION_MODE} global_planner:=${GLOBAL_PLANNER} local_planner:=${LOCAL_PLANNER} neupan_config:=${NEUPAN_CONFIG} ${BRIDGE_ARGS} ${BRINGUP_ARGS} 2>&1 | tee ${LOG_DIR}/go2.log" C-m
ons2_log "w2 starting Go2 navigation stack ${BRIDGE_TXT} (global_planner=${GLOBAL_PLANNER}, local_planner=${LOCAL_PLANNER}, mode=${LOCAL_PLANNER_MODE})..."
sleep 25
[[ "${ONS2_BRIDGE:-0}" = "1" ]] && ons2_wait_for "rosnode list" "/vel_to_sdk$" 30 "unitree chassis bridge vel_to_sdk"
ons2_wait_for "rostopic list" "/scan$" 30 "pc_to_scan"
ons2_wait_for "rostopic list" "/scan_360$" 30 "sliding_map"
[[ "${GLOBAL_PLANNER}" = "map_planner" ]] && ons2_wait_for "rosservice list" "/map_planner/plan$" 30 "map_planner plan service"
if [[ "${LOCAL_PLANNER}" = "neupan" ]]; then
  MISSION_WAIT="until rostopic list 2>/dev/null | grep -qx /neupan/nav_path/status; do sleep 1; done; "
else
  MISSION_WAIT=""
fi
tmux new-window -t "${SESSION}" -n mission "bash --norc"
tmux send-keys -t "${SESSION}:mission" \
  "${ENV_PREFIX} && ${MISSION_WAIT}roslaunch mission_manager mission_manager.launch enable_dynamic_mode:=true enable_follow_search:=true enable_atomic_move:=true enable_alignment:=true route_segment_enable:=true route_segment_length_m:=10.0 route_segment_overlap_m:=3.0 navigation_mode:=${NAVIGATION_MODE} periodic_replan_sec:=${PERIODIC_REPLAN_SEC} local_planner:=${LOCAL_PLANNER} local_planner_mode:=${LOCAL_PLANNER_MODE} ${MISSION_IFACE_ARGS} 2>&1 | tee ${LOG_DIR}/mission.log" C-m
ons2_log "w4 starting mission_manager..."
sleep 12
ons2_start_window "${SESSION}" waypoint "${ENV_PREFIX} && python3 ${REPO_ROOT}/ros_ws/scripts/waypoint_goal_bridge.py 2>&1 | tee ${LOG_DIR}/waypoint.log"
ons2_start_window "${SESSION}" rosbridge "${ENV_PREFIX} && roslaunch rosbridge_server rosbridge_websocket.launch port:=9090 max_message_size:=100000000 2>&1 | tee ${LOG_DIR}/rosbridge.log"
# foxglove_bridge (ROS1) aborts on a broken browser connection (websocketpp::exception, seen 12 Nov):
# keep it running by restarting it automatically.
ons2_start_window "${SESSION}" foxglove "${ENV_PREFIX} && while true; do roslaunch foxglove_bridge foxglove_bridge.launch port:=8765 send_buffer_limit:=200000000 2>&1 | tee -a ${LOG_DIR}/foxglove.log; echo foxglove_bridge exited, restarting in 3 s; sleep 3; done"
ons2_start_window "${SESSION}" health "${ENV_PREFIX} && export REPO_ROOT=${REPO_ROOT} ONS2_ROBOT=go2_agx && source ${SCRIPT_DIR}/lib/ons2_lib.sh && ons2_run_ros1_health_check \"Go2 (AGX) Navigation Stack\" 1 ${HEALTH_NODES} 2>&1 | tee ${LOG_DIR}/health.log"
ons2_start_window "${SESSION}" cloudview "${ENV_PREFIX} && python3 ${REPO_ROOT}/scripts/cloud_view.py"
ons2_start_window "${SESSION}" bodymarker "${ENV_PREFIX} && python3 ${REPO_ROOT}/scripts/go2_body_marker.py"
if [[ "${ONS2_YOLO:-0}" = "2" ]]; then
  # OUR: --with-yolo-agx = detector on the AGX GPU (YOLOv5l, start_yolo_gpu_agx.sh); here only tf_relay + object_query
  ons2_start_window "${SESSION}" yolo "${ENV_PREFIX} && roslaunch yolo_ros yolo_query_only.launch 2>&1 | tee ${LOG_DIR}/yolo.log"
  ons2_log "w5 yolo_ros query side (detector: AGX GPU YOLOv5l; window yolo: type list / chair ...)"
  ons2_start_window "${SESSION}" yolocmd "${ENV_PREFIX} && python3 ${REPO_ROOT}/scripts/yolo_cmd_bridge.py"
elif [[ "${ONS2_YOLO:-0}" = "1" ]]; then
  ons2_start_window "${SESSION}" yolo "${ENV_PREFIX} && roslaunch yolo_ros yolo_detection.launch 2>&1 | tee ${LOG_DIR}/yolo.log"
  ons2_log "w5 their yolo_ros (window yolo: type list / chair / move to the front of the chair)"
  ons2_start_window "${SESSION}" yolocmd "${ENV_PREFIX} && python3 ${REPO_ROOT}/scripts/yolo_cmd_bridge.py"
  ons2_start_window "${SESSION}" yoloview "${ENV_PREFIX} && python3 ${REPO_ROOT}/scripts/yolo_debug_compress.py"
fi
ons2_start_window "${SESSION}" control "${ENV_PREFIX} && export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\${PYTHONPATH:-} && /usr/bin/python3 ${REPO_ROOT}/ros_ws/scripts/test_nav_control_rospy.py --allow-motion"
ons2_log "=================================================="
ons2_log "Go2 stack started (Odin1 on AGX) ${BRIDGE_TXT}"
ons2_log "  Local planner : ${LOCAL_PLANNER}"
ons2_log "  Navigation    : ${NAVIGATION_MODE}"
ons2_log "  Map           : ${MAP_FILE_NAME}"
ons2_log "  Attach        : docker exec -it ons2-go2 tmux attach -t ${SESSION}"
ons2_log "=================================================="
