#!/bin/bash
# AGX ORIN. ONS2 Odin-perception (Florence-2), step 3 of its README: the ROS 1 frame server (port 8730) that reads
# /odin1/image/undistorted and republishes annotated frames on /inspection/annotated/compressed (see it in Foxglove).
# ROS 1 only exists inside the container here, so it runs as the process "capture" in the container odin-agx.
# Needs start_odin_agx.sh go2 first. Then send commands on /florence/command (e.g. from Foxglove).
#   stop: ~/odin_test/stop_odin_agx.sh perception   (this also stops the driver)
D=$HOME/odin_test; . $D/agx_lib.sh
agx_alive driver || { echo "First run: ~/odin_test/start_odin_agx.sh go2"; exit 1; }
agx_start capture 'cd /opt/odin-perception; export PYTHONPATH=.:$PYTHONPATH; exec /usr/bin/python3 -m inspection_perception.capture.ros1_odin_node --serve --host 0.0.0.0 --port 8730 --prefer raw --raw-topic /odin1/image/undistorted'
ok=0; for i in $(seq 1 15); do sleep 2; curl -sf -m 2 http://127.0.0.1:8730/meta.json | grep -q "\"has_frame\": true" && { ok=1; break; }; done
[ $ok = 1 ] || { echo "No frames yet: tail $(agx_log capture)"; exit 1; }
echo "OK: frame server ready (http://127.0.0.1:8730, 10 fps)"
# Foxglove control (our helpers): Florence-2 kept loaded on the GPU (127.0.0.1:8740) + ROS relay /florence/command -> /florence/result
# The model service runs in the container odin-gpu when its image exists (see agx_lib.sh), otherwise on the host
mkdir -p $HOME/perception_test
if gpu_in_container; then
  gpu_ensure || { echo "Container $GPU_CTR failed to start"; exit 1; }
  pkill -u $(id -u) -f "[f]lorence_service.py" 2>/dev/null; sleep 1   # a copy left on the host would keep the port (the container runs as root: not matched)
  FLOG=$(gpu_log florence); gpu_start florence "exec $HOME/odin-perception/.venv/bin/python $HOME/odin_test/florence_service.py" || { echo "Florence-2 service did not start: tail $FLOG"; exit 1; }
else
  FLOG=$HOME/perception_test/florence_service.log
  pkill -f "[f]lorence_service.py" 2>/dev/null; sleep 1
  setsid nohup $HOME/odin-perception/.venv/bin/python $HOME/odin_test/florence_service.py > $FLOG 2>&1 < /dev/null &
fi
agx_cp $D/florence_cmd_relay.py /relay.py
agx_start relay "python3 /relay.py"
for i in $(seq 1 30); do sleep 2; grep -q "listening on" $FLOG && { echo "OK: Florence-2 loaded; Foxglove can use /florence/command -> /florence/result$(gpu_in_container && echo "  [container $GPU_CTR]")"; exit 0; }; done
echo "Florence-2 service not ready: tail $FLOG"; exit 1
