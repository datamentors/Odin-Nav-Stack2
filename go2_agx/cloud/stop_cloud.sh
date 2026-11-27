#!/bin/bash
# GO2 (Jetson). Stops what start_cloud.sh started, in reverse order: the Pulse runtime, then cloud_publisher.
# The ROS 1 -> ROS 2 bridge keeps running (the remote monitor may use it); add --with-bridge to stop it too.
# The stack (ons2-go2, odin-agx) is not touched.
HERE=$(cd "$(dirname "$0")" && pwd)
docker rm -f odin-pulse-cloud >/dev/null 2>&1 && echo "stopped odin-pulse-cloud" || echo "odin-pulse-cloud was not running"
docker exec ons2-go2 pkill -f cloud_publisher.py >/dev/null 2>&1 && echo "stopped cloud_publisher" || echo "cloud_publisher was not running"
if [ "${1:-}" = --with-bridge ]; then
  "$HERE/../go2/stop_bridge.sh"
fi
