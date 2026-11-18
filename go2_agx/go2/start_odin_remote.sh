#!/bin/bash
# JETSON (Go2). "Remote Odin1" mode: the Odin1 is on the AGX Orin (AGX_IP, USB 3 + color camera),
# connected to the Go2 with the RJ45 cable. This starts here the ROS1 master (reachable at GO2_IP:11311)
# and the ROS1 -> ROS2 bridge (domain 42), instead of the local Odin1 driver. Order:
#   1. here:    ~/odin_nav_stack/drv0144/start_odin_remote.sh
#   2. AGX:     ~/odin_test/start_odin_agx.sh go2          (Odin1 driver, relocalization)
#   3. here:    ~/ons2_docker/ons2_start.sh --map=<map_name> --with-yolo-agx
# Stop: ons2_stop.sh, then on the AGX stop_odin_agx.sh, then here stop_odin_remote.sh
set -e
CONF="${GO2_AGX_CONF:-$HOME/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
: "${AGX_IP:?set AGX_IP (AGX address on the cable to the Go2) in $CONF}"
XML=$HOME/odin_nav_stack/drv0144/fastdds_udp.xml
ROSENV="-e ROS_MASTER_URI=http://${GO2_IP}:11311 -e ROS_IP=${GO2_IP} -e ROS_HOSTNAME=${GO2_IP}"  # the images set ROS_HOSTNAME=localhost, which wins over ROS_IP
ip -4 addr show enP8p1s0 | grep -q "${GO2_IP}/" || { echo "No ${GO2_IP} on enP8p1s0"; exit 1; }
ping -c1 -W2 ${AGX_IP} >/dev/null && echo "AGX (${AGX_IP}) answers over the cable" \
  || echo "WARNING: the AGX (${AGX_IP}) does not answer: check the RJ45 cable"
docker rm -f odin-master odin-bridge >/dev/null 2>&1 || true
docker run -d --name odin-master --network host $ROSENV --entrypoint /bin/bash odin-nav:drv0144 \
  -c "source /opt/ros/noetic/setup.bash; roscore" >/dev/null
for i in $(seq 1 20); do
  sleep 1
  docker exec odin-master bash -c "source /opt/ros/noetic/setup.bash; rosnode list" >/dev/null 2>&1 && break
done
echo "ROS1 master running at http://${GO2_IP}:11311"
docker run -d --name odin-bridge --network host $ROSENV \
  -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds_udp.xml -v $XML:/fastdds_udp.xml:ro \
  --entrypoint /bin/bash ros1_bridge:go2 \
  -c "source /opt/ros/noetic/setup.bash; source /opt/ros/foxy/setup.bash; ros2 run ros1_bridge dynamic_bridge" >/dev/null   # on demand: only topics someone uses in ROS 2 (the AGX publishes 60 MB/s raw images)
echo "ROS1 -> ROS2 bridge running on demand (ROS_DOMAIN_ID=42)."
echo "Next: the AGX joins this master (autostart, or ~/odin_test/start_odin_agx.sh go2), then here ~/ons2_docker/ons2_start.sh"
