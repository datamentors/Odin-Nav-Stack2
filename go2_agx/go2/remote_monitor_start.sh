#!/bin/bash
# GO2 host. ONS2 Remote Web Monitor (ros2_ws/scripts/remote_monitor.py, upstream says "ROS 2 only / Go2 not supported").
# Runs UNCHANGED on the Go2 host ROS 2 Humble (nothing installed or modified in Humble), reaching the ONS2 ROS 1 stack
# through our ros1_bridge (odin-bridge, dynamic_bridge, ROS_DOMAIN_ID=42, Fast DDS UDP profile).
#   ~/ons2_docker/remote_monitor_start.sh        ->  http://<GO2_WIFI_IP>:18088  (Go2 Wi-Fi address, same LAN only)
#   needs the optional bridge: ~/odin_nav_stack/drv0144/start_bridge.sh (or start_odin_remote.sh --with-bridge)
#   stop: pkill -f remote_monitor.py
# WARNING (their doc): no authentication; anyone on the LAN who opens the page can pause/cancel missions.
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/odin_nav_stack/drv0144/fastdds_udp.xml
unset CYCLONEDDS_URI
pkill -f "ros2_ws/scripts/remote_monitor.py" 2>/dev/null; sleep 1
cd $HOME/ons2/ros2_ws/scripts
setsid nohup python3 $HOME/ons2/ros2_ws/scripts/remote_monitor.py --host 0.0.0.0 --port 18088 > $HOME/ons2/logs/remote_monitor.log 2>&1 < /dev/null &
sleep 6; tail -5 $HOME/ons2/logs/remote_monitor.log
echo "Open http://<GO2_WIFI_IP>:18088 (or https://<GO2_WIFI_IP>:18088 for the microphone)"
