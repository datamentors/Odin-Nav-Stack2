#!/bin/bash
# AGX ORIN. Boot autostart (systemd service "odin-autostart", runs as root, once per boot).
# After the AGX is switched on (or reboots) it brings everything back without anyone typing anything:
#   1. waits for the Odin1 on USB and for the Go2 on the Ethernet cable (they may still be booting)
#   2. start_odin_agx.sh go2       creates the single container odin-agx; Odin1 driver + relocalization (joins the
#                                  ROS master on the Go2)
#   3. start_depth_agx.sh          depth image for the ONS2 yolo_ros
#   3b. start_yolo_gpu_agx.sh      YOLOv5l detector on the AGX GPU (Go2: ons2_start.sh --with-yolo-agx)
#   4. start_perception_agx.sh     Florence-2 (numbered detection) + frame server + Foxglove relay
# Then the odin-watchdog service keeps the driver alive. Log: ~/odin_test/logs/autostart.log
# Turn off: sudo systemctl disable odin-autostart        Run again by hand: sudo systemctl restart odin-autostart
U="${ODIN_USER:-$(id -nu 1000)}"; H=$(getent passwd "$U" | cut -d: -f6); D=$H/odin_test; LOG=$D/logs/autostart.log
CONF="${GO2_AGX_CONF:-$H/.config/go2_agx/network.env}"; [ -f "$CONF" ] && . "$CONF"   # see go2_agx/network.env.example
: "${GO2_IP:?set GO2_IP (Go2 wired address, runs the ROS master) in $CONF}"
mkdir -p $D/logs; chown $U:$U $D/logs
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
as_user() { su - $U -c "$1" >> $LOG 2>&1; }
odin_present() { for d in /sys/bus/usb/devices/*; do [ "$(cat $d/idVendor 2>/dev/null)" = 2207 ] && return 0; done; return 1; }
log "=== boot autostart"
for i in $(seq 1 90); do odin_present && break; sleep 2; done
odin_present && log "Odin1 on USB" || log "Odin1 not on USB yet (the watchdog will handle it later)"
for i in $(seq 1 150); do ping -c1 -W2 ${GO2_IP} >/dev/null 2>&1 && break; sleep 2; done   # up to ~5-10 min
ping -c1 -W2 ${GO2_IP} >/dev/null 2>&1 || { log "Go2 (${GO2_IP}) not reachable: nothing started. Run by hand later: sudo systemctl restart odin-autostart"; exit 0; }
# the ROS master on the Go2 (process master in the container ons2-go2) may come up later than the network: wait for port 11311
for i in $(seq 1 120); do timeout 2 bash -c "exec 3<>/dev/tcp/${GO2_IP}/11311" 2>/dev/null && break; sleep 5; done
log "Go2 network OK, starting the Odin1 driver"
MODE=go2; ip -4 addr show | grep -q "inet ${GO2_IP}/" && MODE=same   # same: everything runs on the Go2 Jetson
as_user "$D/start_odin_agx.sh $MODE" && log "driver OK" || log "driver start reported a problem (the watchdog will retry)"
# 18 Nov: each companion waits for the Go2 ROS master again and is retried up to 3 times (the master may be restarted
# while we start them: they then crash with "unable to contact ROS master"). The watchdog also relaunches them later.
master_up() { timeout 2 bash -c "exec 3<>/dev/tcp/${GO2_IP}/11311" 2>/dev/null; }
start_companion() {   # $1 script  $2 label
  local n i
  for n in 1 2 3; do
    for i in $(seq 1 60); do master_up && break; sleep 5; done
    as_user "$D/$1" && { log "$2 OK"; return 0; }
    log "$2 start failed (attempt $n/3), retrying in 20 s"; sleep 20
  done
  log "$2 start FAILED 3 times: the watchdog keeps retrying"
}
start_companion start_depth_agx.sh "depth"
if [ -x $H/odin-perception/.venv/bin/python ]; then   # GPU environment (CUDA venv): only there
  start_companion start_yolo_gpu_agx.sh "YOLO GPU (YOLOv5l)"
  start_companion start_perception_agx.sh "Florence-2"
else
  log "no GPU environment (~/odin-perception/.venv): YOLO GPU and Florence-2 not started (use ons2_start.sh --with-yolo)"
fi
log "=== autostart done"
