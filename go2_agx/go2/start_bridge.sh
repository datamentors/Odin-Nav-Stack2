#!/bin/bash
# JETSON (Go2). OPTIONAL ROS 1 -> ROS 2 bridge (container odin-bridge, image ros1_bridge:go2, domain 42). Only needed by the
# remote web monitor (remote_monitor_start.sh); navigation does not use it. Needs the ROS master (start_odin_remote.sh).
#   ~/odin_nav_stack/drv0144/start_bridge.sh        stop: ~/odin_nav_stack/drv0144/stop_bridge.sh
set -e
CONF="${GO2_AGX_CONF:-$HOME/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
XML=$HOME/odin_nav_stack/drv0144/fastdds_udp.xml
ROSENV="-e ROS_MASTER_URI=http://${GO2_IP}:11311 -e ROS_IP=${GO2_IP} -e ROS_HOSTNAME=${GO2_IP}"
docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; timeout 3 rosnode list" >/dev/null 2>&1 \
  || { echo "First run: ~/odin_nav_stack/drv0144/start_odin_remote.sh"; exit 1; }
docker rm -f odin-bridge >/dev/null 2>&1 || true
docker run -d --name odin-bridge --network host $ROSENV \
  -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds_udp.xml -v $XML:/fastdds_udp.xml:ro \
  --entrypoint /bin/bash ros1_bridge:go2 \
  -c "source /opt/ros/noetic/setup.bash; source /opt/ros/foxy/setup.bash; ros2 run ros1_bridge dynamic_bridge" >/dev/null   # on demand: only topics someone uses in ROS 2 (the AGX publishes 60 MB/s raw images)
echo "ROS1 -> ROS2 bridge running on demand (container odin-bridge, ROS_DOMAIN_ID=42)."
