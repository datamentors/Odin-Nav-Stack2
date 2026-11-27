#!/bin/bash
# GO2. Starts Odin-Nav-Stack2 (ONS2) inside the container "ons2-go2" (ROS1 inside; host ROS2 Humble untouched), the same
# container that runs the ROS master. Needs: start_odin_remote.sh (creates the container + master) here + start_odin_agx.sh go2
# on the AGX (Odin1, relocalized). Restarting ONS2 does not touch the container or the master.
#   ~/ons2_docker/ons2_start.sh --map=<map_name> [other ons2 options]      stop: ~/ons2_docker/ons2_stop.sh
#   Console:  docker exec -it ons2-go2 tmux attach -t ons2_go2      (Ctrl+b then n/p = next/prev window, Ctrl+b d = leave)
# --with-bridge : THEIR Go2 bridge (unitree_control): the robot STANDS UP at start and walks on /cmd_vel. Remote in hand!
# --with-yolo   : THEIR yolo_ros (yolo_detection.launch) in tmux window "yolo" (needs ~/odin_test/start_depth_agx.sh on the AGX)
# --with-yolo-agx : same, but the DETECTOR runs on the AGX GPU (YOLOv5l, ~60 ms/image; needs ~/odin_test/start_yolo_gpu_agx.sh
#                   on the AGX); here only tf_relay + object_query + our yolo_cmd_bridge. Recommended (16 Nov).
# --with-cloud : OUR cloud_publisher in tmux window "cloud" (JSON for the cloud side on /cloud_gateway/{telemetry,status,events}; starts the
#                ROS 1 -> ROS 2 bridge odin-bridge if it is not running). The cloud side itself is in ardia-robotics. No motion command.
# --with-semantic-map : OUR semantic map MVP in tmux window "semmap" (go2_agx/semantic_map: objects seen by the GPU detector /
#                Florence-2, kept in the map frame; /semantic_map/objects, /semantic_map/bev). Needs --with-yolo-agx data.
CONF="${GO2_AGX_CONF:-$HOME/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
BRIDGE=0; YOLO=0; CLOUD=0; SEMMAP=0; ARGS=()
for a in "$@"; do
  case "$a" in --with-bridge) BRIDGE=1 ;; --with-yolo) YOLO=1 ;; --with-yolo-agx) YOLO=2 ;; --with-cloud) CLOUD=1 ;; --with-semantic-map) SEMMAP=1 ;; *) ARGS+=("$a") ;; esac
done
printf '%s\n' "${ARGS[@]}" | grep -q '^--map=' || { echo "Usage: $0 --map=<map_name> [--with-yolo-agx] [--with-cloud] [--with-semantic-map] [--with-bridge]"; exit 1; }
if [ $BRIDGE = 1 ]; then
  echo "WITH THEIR BRIDGE: the Go2 will STAND UP in ~30 s and walk when a mission is sent. Remote in hand. Ctrl+C to abort (5 s)"
  sleep 5
fi
docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; timeout 3 rosnode list" >/dev/null 2>&1 \
  || { echo "First run: ~/odin_nav_stack/drv0144/start_odin_remote.sh (container ons2-go2 + ROS master)"; exit 1; }
if [ $CLOUD = 1 ] && [ "$(docker inspect -f '{{.State.Running}}' odin-bridge 2>/dev/null)" != true ]; then
  # the cloud side (ardia-robotics, ROS 2) reads our topics through the ROS 1 -> ROS 2 bridge
  "$HOME/odin_nav_stack/drv0144/start_bridge.sh" || echo "WARNING: the ROS 1 -> ROS 2 bridge did not start: the cloud side gets no data"
fi
"$(dirname "$0")/ons2_stop.sh" >/dev/null   # a previous ONS2 run, if any (the container and the master stay)
docker exec -e ONS2_BRIDGE=$BRIDGE -e ONS2_YOLO=$YOLO -e ONS2_CLOUD=$CLOUD -e ONS2_SEMMAP=$SEMMAP ons2-go2 bash $HOME/ons2/scripts/ons2_go2_agx.sh "${ARGS[@]}"
