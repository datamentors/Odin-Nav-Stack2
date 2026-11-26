#!/bin/bash
# AGX ORIN. Starts the Odin1 driver (USB 3, color camera) in relocalization mode with the site map (maps/relocalization).
#   ~/odin_test/start_odin_agx.sh local   own ROS master on the AGX (bench test, no Go2 needed)
#   ~/odin_test/start_odin_agx.sh go2     joins the ROS master on the Go2 Jetson (GO2_IP) over the
#                                         Ethernet cable; the AGX must have AGX_IP on eno1
# Stop with: ~/odin_test/stop_odin_agx.sh
# The driver runs as the process "driver" of the single AGX container odin-agx (see agx_lib.sh). If the container already
# runs with the same ROS master it is kept, so only the driver (and the camera stream) restart.
CONF="${GO2_AGX_CONF:-$HOME/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
: "${AGX_IP:?set AGX_IP (AGX address on the cable to the Go2) in $CONF}"
MODE=${1:-local}
D=$HOME/odin_test
. $D/agx_lib.sh
case $MODE in
  local) ENV="-e ROS_MASTER_URI=http://localhost:11311 -e ROS_HOSTNAME=localhost"; WAIT="" ;;
  go2)   # Wait (up to 90 s) for the cable link, the IP and the Go2, instead of failing at the first try:
         # the link drops for a few seconds when a cable is moved and the Go2 takes ~1 min to boot.
         for i in $(seq 1 45); do
           if [ "$(cat /sys/class/net/eno1/carrier 2>/dev/null)" != 1 ]; then MSG="no link on eno1 (Ethernet cable AGX <-> Go2 unplugged, or the Go2 is off)"
           elif ! ip -4 addr show eno1 | grep -q "${AGX_IP}/"; then MSG="eno1 has link but no ${AGX_IP} yet"; nmcli con up go2-net >/dev/null 2>&1
           elif ! ping -c1 -W2 ${GO2_IP} >/dev/null; then MSG="cable OK but the Go2 Jetson (${GO2_IP}) does not answer yet (still booting?)"
           else MSG=""; break; fi
           [ $i = 1 ] && echo "Waiting for the Go2 network: $MSG"
           sleep 2
         done
         [ -n "$MSG" ] && { echo "ERROR: $MSG"; exit 1; }
         echo "Go2 network OK (eno1 $(cat /sys/class/net/eno1/speed) Mb/s, Go2 answers)"
         ENV="-e ROS_MASTER_URI=http://${GO2_IP}:11311 -e ROS_IP=${AGX_IP} -e ROS_HOSTNAME=${AGX_IP}"; WAIT="--wait" ;;
  *) echo "usage: $0 local|go2"; exit 1 ;;
esac
# The Odin1 must be on USB (vendor 2207) before starting the driver
odin_present() { for d in /sys/bus/usb/devices/*; do [ "$(cat $d/idVendor 2>/dev/null)" = 2207 ] && return 0; done; return 1; }
if ! odin_present; then
  echo "Waiting for the Odin1 on USB..."; for i in $(seq 1 10); do sleep 2; odin_present && break; done
fi
if ! odin_present; then
  echo "ERROR: the Odin1 is NOT connected on USB (the network is fine)."
  journalctl -k -n 50 --no-pager 2>/dev/null | grep -q "Cannot enable" && echo "  The kernel says 'Cannot enable. Maybe the USB cable is bad?': the Odin1 is hung or its USB cable/connector is failing."
  echo "  Fix: run ~/odin_test/fix_odin_usb.sh (resets the USB hub). If that fails: unplug the Odin1 power for 10 s and plug it back."
  exit 1
fi
agx_ensure "$ENV" || { echo "ERROR: could not start the container $AGX_CTR"; exit 1; }
start_driver() { agx_start driver "env -u LD_PRELOAD roslaunch $WAIT odin_ros_driver odin1_headless.launch"; }
echo "Mode: $MODE. Odin1 USB: $(for d in /sys/bus/usb/devices/*; do [ -f $d/idVendor ] && grep -qi hawk $d/product && echo "speed=$(cat $d/speed)"; done)"
start_driver
echo "Loading the map into the Odin1 (USB 3: fast if it is already stored in the Odin1)..."
for i in $(seq 1 40); do
  sleep 5
  LOG=$(cat $(agx_log driver) 2>&1)
  if echo "$LOG" | grep -q "relocalization success"; then echo "OK: relocalization success"; break; fi
  if echo "$LOG" | grep -q "Open device failed" && [ -z "$RETRIED" ]; then
    echo "The Odin1 is not answering yet: retrying in 10 s"; RETRIED=1
    agx_stop_proc driver; sleep 10; start_driver; continue
  fi
  if echo "$LOG" | grep -q -E "file start fail|map path set fail|Open device failed|FATAL"; then
    echo "DRIVER ERROR:"; echo "$LOG" | grep -E "fail|FATAL" | tail -5; exit 1
  fi
done
# Color camera for the laptop over Wi-Fi (MJPEG over HTTP, full resolution): http://<AGX Wi-Fi IP>:8080/
agx_cp $D/cam_stream.py /cam_stream.py && agx_start cam "/usr/bin/python3 /cam_stream.py"
echo "Camera stream: http://$(ip -4 -br addr show wlP1p1s0 | awk "{print \$3}" | cut -d/ -f1):8080/  (open it in a browser)"
echo "$LOG" | grep -q "relocalization success" || echo "WARNING: not relocalized yet (the Odin1 must be inside the mapped area; moving it gently helps)"
