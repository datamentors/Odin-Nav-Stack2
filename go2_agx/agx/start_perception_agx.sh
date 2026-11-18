#!/bin/bash
# AGX ORIN. ONS2 Odin-perception (Florence-2), step 3 of its README: the ROS 1 frame server (port 8730) that reads
# /odin1/image/undistorted and republishes annotated frames on /inspection/annotated/compressed (see it in Foxglove).
# ROS 1 only exists inside containers here, so it runs in container "odin-perception-cap" (image odin-yolo:0144).
# Needs start_odin_agx.sh go2 first.  Then run checks with ~/odin_test/inspect.sh [tasks.yaml]
#   stop: docker rm -f odin-perception-cap
CONF="${GO2_AGX_CONF:-$HOME/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
: "${AGX_IP:?set AGX_IP (AGX address on the cable to the Go2) in $CONF}"
docker ps --format "{{.Names}}" | grep -qx odin-agx || { echo "First run: ~/odin_test/start_odin_agx.sh go2"; exit 1; }
docker rm -f odin-perception-cap >/dev/null 2>&1
docker run -d --name odin-perception-cap --network host \
  -e ROS_MASTER_URI=http://${GO2_IP}:11311 -e ROS_IP=${AGX_IP} -e ROS_HOSTNAME=${AGX_IP} -e PYTHONUNBUFFERED=1 \
  -v $HOME/odin-perception:/opt/odin-perception --entrypoint bash odin-yolo:0144 -c "source /opt/ros/noetic/setup.bash; cd /opt/odin-perception; export PYTHONPATH=.:\$PYTHONPATH; exec /usr/bin/python3 -m inspection_perception.capture.ros1_odin_node --serve --host 0.0.0.0 --port 8730 --prefer raw --raw-topic /odin1/image/undistorted" >/dev/null
ok=0; for i in $(seq 1 15); do sleep 2; curl -sf -m 2 http://127.0.0.1:8730/meta.json | grep -q "\"has_frame\": true" && { ok=1; break; }; done
[ $ok = 1 ] || { echo "No frames yet: docker logs odin-perception-cap"; exit 1; }
echo "OK: frame server ready (http://127.0.0.1:8730, 10 fps)"
# Foxglove control (our helpers): Florence-2 kept loaded on the GPU (127.0.0.1:8740) + ROS relay /florence/command -> /florence/result
pkill -f "[f]lorence_service.py" 2>/dev/null; sleep 1
setsid nohup $HOME/odin-perception/.venv/bin/python $HOME/odin_test/florence_service.py > $HOME/perception_test/florence_service.log 2>&1 < /dev/null &
docker rm -f odin-perception-cmd >/dev/null 2>&1
docker run -d --name odin-perception-cmd --network host -e ROS_MASTER_URI=http://${GO2_IP}:11311 -e ROS_IP=${AGX_IP} \
  -e ROS_HOSTNAME=${AGX_IP} -e PYTHONUNBUFFERED=1 -v $HOME/odin_test/florence_cmd_relay.py:/relay.py:ro -v $HOME/odin_test/yolo/calib.yaml:/calib.yaml:ro \
  --entrypoint bash odin-yolo:0144 -c "source /opt/ros/noetic/setup.bash; exec python3 /relay.py" >/dev/null
for i in $(seq 1 30); do sleep 2; grep -q "listening on" $HOME/perception_test/florence_service.log && { echo "OK: Florence-2 loaded; Foxglove can use /florence/command -> /florence/result"; exit 0; }; done
echo "Florence-2 service not ready: tail ~/perception_test/florence_service.log"; exit 1
