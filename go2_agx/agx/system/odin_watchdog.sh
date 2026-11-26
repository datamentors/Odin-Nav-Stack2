#!/bin/bash
# AGX ORIN. Odin1 watchdog (systemd service "odin-watchdog", runs as root).
# Every ~10 s checks that the Odin1 driver (process "driver" in the container odin-agx) is really publishing /odin1/odometry.
# If it stops for 2 checks in a row (the Odin1 fell off the USB, or the driver got stuck after an
# "Auto USB reset"), it recovers by itself:
#   1. Safety: cancels the ONS2 mission and the YOLO go/follow, and sends zero velocity, so the robot
#      does not walk off towards the old goal when the odometry comes back.
#   2. If the Odin1 is missing from the USB: resets the internal USB hub 2-3 (like fix_odin_usb.sh).
#   3. Restarts the driver with start_odin_agx.sh (same mode as before: go2 or local) -> relocalization.
# It does nothing when the driver is not wanted (you stopped it on purpose with stop_odin_agx.sh),
# while start_odin_agx.sh is running, or while the Go2 network is down (restarting would not help).
# 17 Nov: ALSO checks the AGX "companions" while the driver is fine: if the depth (/pcd2depth_node), YOLO
# (/yolo_detector) or Florence relay (/florence_cmd_relay) node is missing from the Go2 ROS master twice in a row
# (the Go2 master was restarted, e.g. start_odin_remote.sh run again: they stay registered in the OLD master and
# Foxglove says "Image topic does not exist"), it relaunches that start script. Only components that are "wanted"
# (started on purpose), never while a start script runs, at most once per 2 min each.
# 18 Nov: ALSO relaunches a companion whose process DIED (it crashed, e.g. it started while the Go2 master was
# restarting: "unable to contact ROS master"). The stop scripts clear the "wanted" mark, so a component stopped on
# purpose is left alone. Also watches the Florence frame server (/inspection_capture_odin_*, process capture).
# Node names are matched as prefixes.
# The AGX components are processes of ONE container (odin-agx), see agx_lib.sh.
# Log: ~/odin_test/logs/watchdog.log  (also: journalctl -u odin-watchdog)
# Turn off: sudo systemctl disable --now odin-watchdog     Turn on: sudo systemctl enable --now odin-watchdog
U="${ODIN_USER:-$(id -nu 1000)}"; H=$(getent passwd "$U" | cut -d: -f6); D=$H/odin_test; LOG=$D/logs/watchdog.log
CONF="${GO2_AGX_CONF:-$H/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
mkdir -p $D/logs; chown $U:$U $D/logs
. $D/agx_lib.sh
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
ros() { docker exec odin-agx bash -c "source /opt/ros/noetic/setup.bash; $1" 2>/dev/null; }
odin_present() { for d in /sys/bus/usb/devices/*; do [ "$(cat $d/idVendor 2>/dev/null)" = 2207 ] && return 0; done; return 1; }
running() { agx_wanted driver; }          # container up and the driver not stopped on purpose
age() { agx_age driver; }
healthy() { ros "timeout 6 rostopic echo -n1 /odin1/odometry/header/seq" | grep -qE '^[0-9]+'; }
safety_stop() {   # best effort, only meaningful in go2 mode
  ros "timeout 5 rosservice call /waypoint_manager/cancel" >/dev/null
  ros "timeout 4 rostopic pub -1 /yolo/command std_msgs/String 'data: stop'" >/dev/null
  ros "timeout 4 rostopic pub -r 5 /cmd_vel geometry_msgs/Twist '{}'" >/dev/null
}
notify() { ros "timeout 4 rostopic pub -1 /yolo/console std_msgs/String \"data: '[Odin1 watchdog] $1'\"" >/dev/null; }
hub_reset() {
  log "Odin1 missing from USB: resetting hub 2-3"
  echo 0 > /sys/bus/usb/devices/2-3/authorized; sleep 4; echo 1 > /sys/bus/usb/devices/2-3/authorized
  for i in $(seq 1 10); do sleep 2; odin_present && break; done
  /usr/local/bin/odin_usb_tuning.sh >/dev/null 2>&1
}
COMP="depth:/pcd2depth_node:start_depth_agx.sh yolo:/yolo_detector:start_yolo_gpu_agx.sh relay:/florence_cmd_relay:start_perception_agx.sh capture:/inspection_capture:start_perception_agx.sh"
declare -A MISS LAST
check_companions() {
  pgrep -f "start_(depth|yolo_gpu|perception)_agx[.]sh|odin_autostart[.]sh" >/dev/null && return
  local nodes c ctr node script cage why
  nodes=$(ros "timeout 6 rosnode list")
  [ -z "$nodes" ] && return                                   # master not answering: nothing to decide
  for c in $COMP; do
    IFS=: read -r ctr node script <<< "$c"
    agx_wanted $ctr || { MISS[$ctr]=0; continue; }            # not wanted: never started, or stopped on purpose
    if agx_alive $ctr; then
      cage=$(agx_age $ctr)
      [ "${cage:-0}" -lt 90 ] && continue                     # just started
      if grep -q "^$node" <<< "$nodes"; then MISS[$ctr]=0; continue; fi
      why="not registered in the ROS master (Go2 master restarted?)"
    else
      why="process died (it crashed); log: $(agx_log $ctr)"
    fi
    MISS[$ctr]=$(( ${MISS[$ctr]:-0} + 1 )); [ ${MISS[$ctr]} -lt 2 ] && continue
    [ $(( $(date +%s) - ${LAST[$script]:-0} )) -lt 120 ] && continue
    LAST[$script]=$(date +%s); MISS[$ctr]=0
    log "$node ($ctr) $why: relaunching $script"
    su - $U -c "$D/$script" >> $LOG 2>&1 && log "$script OK" || log "$script FAILED (will retry)"
    notify "$node was down ($why) and was relaunched automatically."
  done
}
FAILS=0; ATTEMPTS=0
log "watchdog started"
while true; do
  sleep 8
  running || { FAILS=0; ATTEMPTS=0; continue; }                 # driver stopped on purpose
  pgrep -f start_odin_agx.sh >/dev/null && { FAILS=0; continue; } # someone is (re)starting it
  [ "$(age)" -lt 120 ] && continue                                 # still loading/relocalizing
  MODE_GO2=0; docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' odin-agx | grep -q "ROS_MASTER_URI=http://${GO2_IP}" && MODE_GO2=1
  if healthy; then
    [ $ATTEMPTS -gt 0 ] && { log "OK again: odometry flowing"; notify "Odin1 recovered, odometry OK. Send the goal again."; }
    FAILS=0; ATTEMPTS=0
    [ $MODE_GO2 = 1 ] && check_companions
    continue
  fi
  FAILS=$((FAILS+1)); [ $FAILS -lt 2 ] && continue
  MODE=local; docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' odin-agx | grep -q "ROS_MASTER_URI=http://${GO2_IP}" && MODE=go2
  if [ $MODE = go2 ] && ! ping -c1 -W2 ${GO2_IP} >/dev/null; then
    log "no odometry, but the Go2 (${GO2_IP}) does not answer: network/Go2 problem, not restarting the driver"; FAILS=0; sleep 20; continue
  fi
  ATTEMPTS=$((ATTEMPTS+1))
  log "NO ODOMETRY for ~20 s (attempt $ATTEMPTS, mode $MODE, Odin1 on USB: $(odin_present && echo yes || echo NO)). Recovering..."
  [ $MODE = go2 ] && safety_stop
  tail -n 300 $(agx_log driver) 2>/dev/null | grep -E 'ERROR|reset|unrespons' | tail -5 | sed 's/^/    driver: /' >> $LOG
  # hub reset on the first 3 attempts, then only every 5th (if the Odin1 is unpowered/hung, resetting
  # the hub every minute does not help: just wait for it to come back after you power-cycle it)
  if ! odin_present && { [ $ATTEMPTS -le 3 ] || [ $((ATTEMPTS % 5)) = 0 ]; }; then hub_reset; fi
  if ! odin_present; then
    log "Odin1 not on USB: it is hung or has no power -> unplug its power 10 s and plug it back (watching for it)"
    FAILS=1; for i in $(seq 1 15); do sleep 4; odin_present && break; done; continue
  fi
  cp $(agx_log driver) $D/logs/driver_$(date +%Y%m%d_%H%M%S)_watchdog.log 2>/dev/null
  su - $U -c "$D/start_odin_agx.sh $MODE" >> $LOG 2>&1 && log "driver restarted" || log "driver restart FAILED (will retry)"
  [ $MODE = go2 ] && notify "Odin1 dropped and was restarted automatically (relocalized). Mission was cancelled: send it again."
  FAILS=0
  [ $ATTEMPTS -ge 3 ] && sleep 60
done
