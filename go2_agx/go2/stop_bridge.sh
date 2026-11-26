#!/bin/bash
# JETSON (Go2). Stops the optional ROS 1 -> ROS 2 bridge (start_bridge.sh). Navigation and the ROS master are not affected.
docker rm -f odin-bridge >/dev/null 2>&1 && echo "Bridge stopped." || echo "The bridge was not running."
