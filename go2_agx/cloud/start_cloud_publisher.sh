#!/bin/bash
# GO2 (Jetson). Starts OUR side of the cloud link: the node cloud_publisher inside ons2-go2 (ROS 1), and the ROS 1 -> ROS 2
# bridge (container odin-bridge, domain 42) if it is not running. Needs the stack running (ons2-go2 + ROS master).
# The cloud side itself (cloud and media gateways, AWS) is NOT here: it is the pulse-runtime profile
# robots/unitree-go2-odin-000 in ardia-robotics (branch go2-demo). The two sides only share ROS topics.
#   ./start_cloud_publisher.sh        stop: ./stop_cloud_publisher.sh        log: ~/ons2/logs/cloud_node.log
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROS1="source /opt/ros/noetic/setup.bash; source \$HOME/ons2/ros_ws/devel/setup.bash"
running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

running ons2-go2 || { echo "ERROR: container ons2-go2 is not running. Start the stack first (test_stack.sh start)."; exit 1; }
docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; timeout 3 rosnode list" >/dev/null 2>&1 \
  || { echo "ERROR: the ROS master in ons2-go2 does not answer."; exit 1; }

echo "== cloud_publisher in ons2-go2"
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

echo "== ROS 1 -> ROS 2 bridge"
if running odin-bridge; then
  echo "   already running (odin-bridge)"
else
  "$HERE/../go2/start_bridge.sh"
fi
echo "Next, in ardia-robotics (branch go2-demo): scripts/go2_odin/start.sh"
