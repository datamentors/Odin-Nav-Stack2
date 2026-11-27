#!/bin/bash
# GO2 (Jetson). Starts the cloud side of the Odin1 stream, in this order:
#   1. cloud_publisher inside ons2-go2 (ROS 1)          log: ~/ons2/logs/cloud_node.log
#   2. the ROS 1 -> ROS 2 bridge (container odin-bridge, domain 42), only if it is not running
#   3. the Pulse runtime with our profile (container odin-pulse-cloud), comms only
# Needs the stack running (ons2-go2 + ROS master). It never starts the chassis bridge and sends no motion command:
# the profile has no control adapter and the gateway has command dispatch off.
#   ./start_cloud.sh        stop: ./stop_cloud.sh        check: ./test_phase1.sh
# Settings (environment): PULSE_IMAGE (default ardia/pulse-runtime:dev), ARDIA_IOT_DIR (default /etc/ardia/iot, aws_iot only)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${PULSE_IMAGE:-ardia/pulse-runtime:dev}
NAME=odin-pulse-cloud
PROFILE=$HERE/profile
ROS1="source /opt/ros/noetic/setup.bash; source \$HOME/ons2/ros_ws/devel/setup.bash"

BROKER=$(awk '/^cloud_gateway:/{g=1} g && /broker_type:/{print $2; exit}' "$PROFILE/pulse.yaml")
echo "profile: $PROFILE/pulse.yaml (broker_type: $BROKER)"

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

running ons2-go2 || { echo "ERROR: container ons2-go2 is not running. Start the stack first (test_stack.sh start)."; exit 1; }
docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; timeout 3 rosnode list" >/dev/null 2>&1 \
  || { echo "ERROR: the ROS master in ons2-go2 does not answer."; exit 1; }

# The identity is shared with the Go2 runtime: with a real broker only one of the two may be connected.
if [ "$BROKER" != mock ]; then
  for c in ardia-pulse-runtime ardia-pulse-cuda-runtime; do
    if running "$c"; then
      echo "ERROR: $c is running and shares the robot identity with this profile (broker_type: $BROKER). Stop it first."
      exit 1
    fi
  done
fi

echo "== 1. cloud_publisher in ons2-go2"
docker exec ons2-go2 pkill -f cloud_publisher.py >/dev/null 2>&1 || true
docker exec ons2-go2 rm -rf /tmp/cloud
docker cp "$HERE" ons2-go2:/tmp/cloud
COMMIT=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || true)
docker exec -d -e STACK_COMMIT="$COMMIT" -e STACK_IMAGES="ons2=ons2:go2,gpu=odin-gpu:go2" ons2-go2 bash -c \
  "$ROS1; python3 /tmp/cloud/cloud_publisher.py _config:=/tmp/cloud/config/cloud_publisher.yaml >> \$HOME/ons2/logs/cloud_node.log 2>&1"
for i in $(seq 1 20); do
  docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; rosnode list 2>/dev/null | grep -qx /cloud_publisher" && break
  sleep 1
done
docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; rosnode list 2>/dev/null | grep -qx /cloud_publisher" \
  || { echo "ERROR: cloud_publisher did not start: tail ~/ons2/logs/cloud_node.log"; exit 1; }
echo "   running"

echo "== 2. ROS 1 -> ROS 2 bridge"
if running odin-bridge; then
  echo "   already running (odin-bridge)"
else
  "$HERE/../go2/start_bridge.sh"
fi

echo "== 3. Pulse runtime ($NAME, $IMAGE)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
IOT=()
[ "$BROKER" = mock ] || IOT=(-v "${ARDIA_IOT_DIR:-/etc/ardia/iot}:/etc/ardia/iot:ro")
docker run -d --name "$NAME" --init --network host --runtime nvidia --stop-signal SIGINT --stop-timeout 20 \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
  -e ROS_DOMAIN_ID=42 -e ROS_LOCALHOST_ONLY=0 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/ardia/dds/fastdds_udp.xml \
  -e FASTDDS_DEFAULT_PROFILES_FILE=/etc/ardia/dds/fastdds_udp.xml -e CYCLONEDDS_URI= \
  -v "$PROFILE:/etc/ardia:ro" -v odin-pulse-state:/var/lib/ardia/runtime "${IOT[@]}" \
  "$IMAGE" ros2 launch pulse_bringup pulse.launch.py robot_profile:=/etc/ardia/pulse.yaml >/dev/null
sleep 3
running "$NAME" || { echo "ERROR: $NAME stopped: docker logs $NAME"; exit 1; }
echo "   running"
echo "Next: ./test_phase1.sh   (logs: docker logs $NAME)"
