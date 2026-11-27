#!/bin/bash
# GO2 (Jetson). Phase 1 check of the cloud side, after ./start_cloud.sh. Read-only: it starts nothing and sends no command.
# It proves our node -> ROS 1 -> bridge -> ROS 2 -> Pulse cloud_gateway (it does NOT show the MQTT output: the mock broker keeps
# no readable log; that is checked with real AWS in Phase 3).
#   ./test_phase1.sh [seconds to listen, default 25]
HERE=$(cd "$(dirname "$0")" && pwd)
SECS=${1:-25}
FAIL=0
pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

echo "== containers and processes"
running ons2-go2 && pass "ons2-go2" || fail "ons2-go2 is not running (start the stack)"
docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; rosnode list 2>/dev/null | grep -qx /cloud_publisher" \
  && pass "cloud_publisher node in the ROS 1 graph" || fail "cloud_publisher is not running (./start_cloud.sh)"
running odin-bridge && pass "odin-bridge (ROS 1 -> ROS 2)" || fail "odin-bridge is not running"
running odin-pulse-cloud && pass "odin-pulse-cloud (Pulse runtime)" || fail "odin-pulse-cloud is not running"
running ardia-pulse-runtime && fail "ardia-pulse-runtime is running too: same robot identity" || pass "the Go2 runtime is not running (shared identity)"

echo "== safety: nothing here can move the robot"
if running odin-pulse-cloud; then
  NODES=$(docker exec odin-pulse-cloud bash -c "source /opt/ros/humble/setup.bash; ros2 node list 2>/dev/null")
  if echo "$NODES" | grep -qiE "control_adapter|teleop"; then fail "a control or teleop node is running: $(echo "$NODES" | grep -iE 'control_adapter|teleop' | tr '\n' ' ')"
  else pass "no control_adapter or teleop node in ROS 2"; fi
  for n in cloud_gateway communications_lifecycle_manager robot_status_collector; do
    echo "$NODES" | grep -qx "/$n" && pass "node /$n" || fail "node /$n missing"
  done
  state=$(docker exec odin-pulse-cloud bash -c "source /opt/ros/humble/setup.bash; ros2 lifecycle get /cloud_gateway 2>&1" | head -1)
  echo "$state" | grep -qi "active" && pass "cloud_gateway lifecycle: $state" || fail "cloud_gateway lifecycle: $state"
fi

echo "== messages seen by the gateway (listening ${SECS} s)"
if running odin-pulse-cloud; then
  docker exec -i odin-pulse-cloud bash -c "source /opt/ros/humble/setup.bash; python3 - $SECS unitree-go2-000" < "$HERE/tests/ros2_check.py" | sed 's/^/  /'
  [ "${PIPESTATUS[0]}" = 0 ] || FAIL=$((FAIL+1))
fi

echo
echo "== $FAIL failed"
[ "$FAIL" = 0 ]
