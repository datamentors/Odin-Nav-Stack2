#!/bin/bash
# AGX ORIN. Depth image /odin1/depth_img_competetion for the ONS2 yolo_ros (their driver launch odin1_ros1.launch starts
# pcd2depth_node next to the driver; our AGX driver uses odin1_headless.launch, so it runs here in its own container).
# Stop: docker rm -f odin-depth
CONF="${GO2_AGX_CONF:-$HOME/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
: "${AGX_IP:?set AGX_IP (AGX address on the cable to the Go2) in $CONF}"
docker ps --format "{{.Names}}" | grep -qx odin-agx || { echo "First run: ~/odin_test/start_odin_agx.sh go2"; exit 1; }
D=$HOME/odin_test; mkdir -p $D/yolo
docker exec odin-agx cat /root/.ros/odin_ros_driver/calib.yaml > $D/yolo/calib.yaml 2>/dev/null
docker rm -f odin-depth >/dev/null 2>&1
docker run -d --name odin-depth --network host -e ROS_MASTER_URI=http://${GO2_IP}:11311 -e ROS_IP=${AGX_IP} -e ROS_HOSTNAME=${AGX_IP} \
  -v $D/control_command_agx.yaml:/opt/odin/ros_ws/src/odin_ros_driver/config/control_command.yaml:ro \
  -v $D/yolo/calib.yaml:/root/.ros/odin_ros_driver/calib.yaml:ro -v $D/odin_depth.launch:/tmp/odin_depth.launch:ro --entrypoint bash odin-drv:0144 \
  -c "source /opt/ros/noetic/setup.bash; source /opt/odin/ros_ws/devel_isolated/setup.bash; roslaunch /tmp/odin_depth.launch" >/dev/null
sleep 8
if docker ps --format "{{.Names}}" | grep -qx odin-depth; then
  echo "OK: depth image /odin1/depth_img_competetion running (container odin-depth, ~2-3 Hz)"
else
  echo "FAILED: docker logs odin-depth"; docker logs --tail 5 odin-depth 2>&1
fi
