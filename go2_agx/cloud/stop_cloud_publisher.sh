#!/bin/bash
# GO2 (Jetson). Stops cloud_publisher in ons2-go2. The ROS 1 -> ROS 2 bridge keeps running (the remote monitor may use it);
# add --with-bridge to stop it too. The stack and the Pulse runtime (ardia-robotics) are not touched.
HERE=$(cd "$(dirname "$0")" && pwd)
docker exec ons2-go2 pkill -f cloud_publisher.py >/dev/null 2>&1 && echo "stopped cloud_publisher" || echo "cloud_publisher was not running"
if [ "${1:-}" = --with-bridge ]; then
  "$HERE/../go2/stop_bridge.sh"
fi
