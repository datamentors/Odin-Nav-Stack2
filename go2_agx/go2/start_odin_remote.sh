#!/bin/bash
# JETSON (Go2). "Remote Odin1" mode: the Odin1 is on the AGX Orin (AGX_IP, USB 3 + color camera),
# connected to the Go2 with the RJ45 cable. This starts here the container "ons2-go2" (image ons2:go2, ROS 1 Noetic)
# with the ROS 1 master as its process "master" (reachable at GO2_IP:11311). ONS2 itself runs later in the SAME
# container (ons2_start.sh), so the master survives ONS2 restarts and the AGX keeps its connection.
#   ~/odin_nav_stack/drv0144/start_odin_remote.sh                 container + master
#   ~/odin_nav_stack/drv0144/start_odin_remote.sh --with-bridge   also the optional ROS 1 -> ROS 2 bridge (start_bridge.sh),
#                                                                 only needed for the remote web monitor
# Order:
#   1. here:    ~/odin_nav_stack/drv0144/start_odin_remote.sh
#   2. AGX:     ~/odin_test/start_odin_agx.sh go2          (Odin1 driver, relocalization)
#   3. here:    ~/ons2_docker/ons2_start.sh --map=<map_name> --with-yolo-agx
# Stop: ons2_stop.sh (only ONS2), then on the AGX stop_odin_agx.sh, then here stop_odin_remote.sh (master + container)
# Running it again while the container and master are up does nothing: stop_odin_remote.sh first to restart them.
set -e
CONF="${GO2_AGX_CONF:-$HOME/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
AGX_IP="${AGX_IP:-}"   # optional: not needed when everything runs on this Go2 (no AGX)
HERE="$(cd "$(dirname "$0")" && pwd)"
WITH_BRIDGE=0; [ "${1:-}" = "--with-bridge" ] && WITH_BRIDGE=1
ROSENV="-e ROS_MASTER_URI=http://${GO2_IP}:11311 -e ROS_IP=${GO2_IP} -e ROS_HOSTNAME=${GO2_IP}"  # the images set ROS_HOSTNAME=localhost, which wins over ROS_IP
ip -4 addr show enP8p1s0 | grep -q "${GO2_IP}/" || { echo "No ${GO2_IP} on enP8p1s0"; exit 1; }
if [ -n "$AGX_IP" ]; then
  ping -c1 -W2 ${AGX_IP} >/dev/null && echo "AGX (${AGX_IP}) answers over the cable" \
    || echo "WARNING: the AGX (${AGX_IP}) does not answer: check the RJ45 cable (ignore this if everything runs on the Go2)"
fi
master_alive() { docker exec ons2-go2 bash -c 'p=$(cat /run/ons2_master.pid 2>/dev/null); [ -n "$p" ] && kill -0 -- -$p 2>/dev/null' 2>/dev/null; }
master_ready() { docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; timeout 3 rosnode list" >/dev/null 2>&1; }
if [ "$(docker inspect -f '{{.State.Running}}' ons2-go2 2>/dev/null)" = true ] \
   && docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' ons2-go2 | grep -qx "ROS_MASTER_URI=http://${GO2_IP}:11311" \
   && master_alive; then
  echo "ons2-go2 and its ROS master are already running: nothing to do (to restart them: stop_odin_remote.sh first)"
else
  docker rm -f ons2-go2 odin-master >/dev/null 2>&1 || true   # odin-master = the separate master container of the old setup
  PCDENV=""; [ -n "${ONS2_MAP_PCD:-}" ] && PCDENV="-e ONS2_MAP_PCD=${ONS2_MAP_PCD}"   # point cloud shown by cloud_view.py (network.env)
  docker run -d --init --name ons2-go2 --network host $ROSENV $PCDENV \
    -e PATH=/opt/conda/envs/neupan/bin:/opt/ros/noetic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e HOME=$HOME -e GO2_LOG_DIR=$HOME/ons2/logs -e TERM=xterm-256color \
    -v $HOME/ons2:$HOME/ons2 -v $HOME/ons2/missions:$HOME/.go2_waypoints --entrypoint sleep ons2:go2 infinity >/dev/null
  # the master gets a clean PATH: the container PATH has the conda Python 3.10 first, which is for the ONS2 nodes, not for roscore
  docker exec -d ons2-go2 setsid bash -c 'mkdir -p "$GO2_LOG_DIR"; echo $$ > /run/ons2_master.pid
    source /opt/ros/noetic/setup.bash; export PATH=/opt/ros/noetic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    exec roscore >> "$GO2_LOG_DIR/master.log" 2>&1'
fi
for i in $(seq 1 20); do master_ready && break; sleep 1; done
master_ready || { echo "ERROR: the ROS master did not start: tail ~/ons2/logs/master.log"; exit 1; }
echo "ROS1 master running at http://${GO2_IP}:11311 (process master in the container ons2-go2)"
if [ $WITH_BRIDGE = 1 ]; then "$HERE/start_bridge.sh"
else echo "ROS1 -> ROS2 bridge not started (optional, only for remote_monitor_start.sh): start_bridge.sh"; fi
echo "Next: the AGX joins this master (autostart, or ~/odin_test/start_odin_agx.sh go2), then here ~/ons2_docker/ons2_start.sh"
