#!/bin/bash
# AGX ORIN. Depth image /odin1/depth_img_competetion for the ONS2 yolo_ros (their driver launch odin1_ros1.launch starts
# pcd2depth_node next to the driver; our AGX driver uses odin1_headless.launch, so it runs here as its own process "depth"
# in the container odin-agx).
# Stop: ~/odin_test/stop_odin_agx.sh depth   (this also stops the driver; to stop only the depth: see agx_lib.sh agx_stop_proc depth)
D=$HOME/odin_test; . $D/agx_lib.sh; mkdir -p $D/yolo
agx_alive driver || { echo "First run: ~/odin_test/start_odin_agx.sh go2"; exit 1; }
# the Odin1 calibration (the driver wrote it): the host copy is read by florence_service.py
docker exec $AGX_CTR cat /root/.ros/odin_ros_driver/calib.yaml > $D/yolo/calib.yaml 2>/dev/null
agx_cp $D/odin_depth.launch /tmp/odin_depth.launch
agx_start depth "env -u LD_PRELOAD roslaunch /tmp/odin_depth.launch"
sleep 8
if agx_alive depth; then
  echo "OK: depth image /odin1/depth_img_competetion running (process depth in $AGX_CTR, ~2-3 Hz)"
else
  echo "FAILED: tail $(agx_log depth)"; tail -5 $(agx_log depth) 2>&1
fi
