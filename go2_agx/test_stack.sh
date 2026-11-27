#!/bin/bash
# Start and test the whole go2_agx stack from ONE machine that can SSH to the robot(s) (a PC), or ON the Go2 itself (GO2_SSH=local).
# It never starts the chassis bridge: nothing it does makes the robot's legs move.
#
# Layouts (STACK=...):
#   single  (default)  EVERYTHING on the Go2 Jetson: containers ons2-go2 (ROS master + ONS2) and odin-agx (Odin1 driver, depth,
#                      and the YOLO / Florence-2 nodes when the GPU environment exists). No AGX Orin, no cable.
#   two                the AGX Orin runs odin-agx, the Go2 runs ons2-go2 (the Ethernet cable between them).
# YOLO (YOLO_MODE=auto|gpu|cpu):
#   gpu  YOLOv5l TensorRT + Florence-2 on the GPU (needs ~/odin-perception/.venv, TensorRT, the weights and the engine)
#   cpu  ONS2's own YOLOv5s on the CPU (ons2_start.sh --with-yolo), no Florence-2
#   auto = gpu when the GPU environment exists on the machine that runs odin-agx, else cpu
#
#   ./go2_agx/test_stack.sh check      read-only preflight (SSH, network.env, Odin1 on USB, images, deployed scripts)
#   ./go2_agx/test_stack.sh deploy     copy this repo's scripts to the machine(s) (replaced files are kept as *.bak_<time>)
#   ./go2_agx/test_stack.sh start      start everything in order (ONS2 without the chassis bridge)
#   ./go2_agx/test_stack.sh verify     containers, processes, ROS nodes, topics, TF, YOLO (and Florence) answers
#   ./go2_agx/test_stack.sh verify --resilience   also restarts ONS2 and checks that the ROS master and odin-agx survive
#   ./go2_agx/test_stack.sh stop       stop everything (reverse order)
#   ./go2_agx/test_stack.sh all        check + start + verify        (add --stop-after to stop at the end)
#
# Settings (environment). SSH keys are best; for passwords pass them only through the environment, never in a file:
#   GO2_SSH=user@host  GO2_SSHPASS=...                      (always; GO2_SSH=local runs everything on this machine, STACK=single only)
#   AGX_SSH=user@host  AGX_SSHPASS=...                      (STACK=two only)
#   NO_MAP=1   test away from the mapped site: the Odin1 runs in odometry mode (custom_map_mode 0, the map frame equals the odom
#              frame, no relocalization) and ONS2 uses the empty placeholder map (--map=empty). Not the real map: only a functional test.
#   MAP=<ONS2 navigation map on the Go2>   GO2_IP (and AGX_IP for STACK=two)   (the IPs: only for `deploy`, to create a missing network.env)
GO2_SSH="${GO2_SSH:?set GO2_SSH, e.g. GO2_SSH=user@192.168.x.x}"
STACK="${STACK:-single}"
case "$STACK" in
  single) AGX_SSH="$GO2_SSH"; AGX_SSHPASS="${GO2_SSHPASS:-}"; AGX_MODE=same; WHO_AGX="the Go2" ;;
  two)    [ "$GO2_SSH" = local ] && { echo "GO2_SSH=local works only with STACK=single"; exit 1; }
          AGX_SSH="${AGX_SSH:?STACK=two needs AGX_SSH}"; AGX_MODE=go2; WHO_AGX="the AGX" ;;
  *) echo "STACK must be single or two"; exit 1 ;;
esac
MAP="${MAP:-restaurant_walls}"; NO_MAP="${NO_MAP:-0}"; [ "$NO_MAP" = 1 ] && MAP=empty
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SSHOPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
CMD="${1:-check}"; shift || true

rsh() { local host=$1 pass=$2; shift 2
  if [ "$host" = local ]; then bash -c "$*"
  elif [ -n "$pass" ]; then SSHPASS="$pass" sshpass -e ssh $SSHOPTS "$host" "$@"; else ssh $SSHOPTS "$host" "$@"; fi; }
g() { rsh "$GO2_SSH" "${GO2_SSHPASS:-}" "$@"; }       # command on the Go2 computer
a() { rsh "$AGX_SSH" "${AGX_SSHPASS:-}" "$@"; }       # command on the machine that runs odin-agx (the Go2 when STACK=single)
gros() { g 'docker exec -i ons2-go2 bash -s' <<< "source /opt/ros/noetic/setup.bash; source \$HOME/ons2/ros_ws/devel/setup.bash 2>/dev/null; $1"; }
rsync_to() { local host=$1 pass=$2 src=$3 dest=$4; shift 4   # rsync_to HOST PASS SRC DEST_DIR [rsync options]
  if [ "$host" = local ]; then rsync -a "$@" "$src" "${dest/#\~/$HOME}"
  elif [ -n "$pass" ]; then SSHPASS="$pass" sshpass -e rsync -a "$@" -e "ssh $SSHOPTS" "$src" "$host:$dest"; else rsync -a "$@" -e "ssh $SSHOPTS" "$src" "$host:$dest"; fi; }

PASS=0; FAIL=0
ok()  { echo "  PASS  $*"; PASS=$((PASS+1)); }
bad() { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
chk() { local name=$1; shift; if "$@" >/dev/null 2>&1; then ok "$name"; else bad "$name"; fi; }
summary() { echo; echo "== $PASS passed, $FAIL failed"; [ $FAIL -eq 0 ]; }

# YOLO mode: gpu or cpu (decided on the machine that runs odin-agx)
yolo_mode() {
  case "${YOLO_MODE:-auto}" in
    gpu|cpu) echo "$YOLO_MODE" ;;
    # gpu when the GPU services can run: the odin-gpu image (Odin-perception inside) or the host venv, plus the service code
    *) if a '(docker image inspect odin-gpu:go2 >/dev/null 2>&1 || test -x ~/odin-perception/.venv/bin/python) && test -d ~/odin_test/yolo_gpu'; then echo gpu; else echo cpu; fi ;;
  esac
}

# ---------------------------------------------------------------- check
do_check() {
  echo "== preflight (STACK=$STACK, YOLO: $(yolo_mode))"
  chk "SSH to the Go2"  g true
  [ "$STACK" = two ] && chk "SSH to the AGX" a true
  chk "network.env on the Go2" g 'test -f ~/.config/go2_agx/network.env'
  chk "Go2 has GO2_IP on its wired port" g '. ~/.config/go2_agx/network.env && [ -n "$GO2_IP" ] && ip -4 addr show enP8p1s0 | grep -q "$GO2_IP/"'
  chk "Go2 image ons2:go2" g 'docker image inspect ons2:go2'
  chk "Go2 has the ONS2 go2_agx profile (~/ons2)" g 'test -f ~/ons2/robots/go2_agx/robot.yaml'
  if [ "$STACK" = two ]; then
    chk "network.env on the AGX" a 'test -f ~/.config/go2_agx/network.env'
    chk "AGX link to the Go2 (eno1 carrier)" a '[ "$(cat /sys/class/net/eno1/carrier)" = 1 ]'
    chk "Go2 reaches the AGX over the cable" g '. ~/.config/go2_agx/network.env && [ -n "$AGX_IP" ] && ping -c1 -W2 $AGX_IP'
  fi
  chk "Odin1 on the USB of $WHO_AGX" a 'for d in /sys/bus/usb/devices/*; do [ "$(cat $d/idVendor 2>/dev/null)" = 2207 ] && exit 0; done; exit 1'
  chk "image odin-yolo:0144 on $WHO_AGX" a 'docker image inspect odin-yolo:0144'
  chk "the relocalization map named in control_command_agx.yaml exists on $WHO_AGX" a 'f=$(grep -o "maps/relocalization/[^\"]*" ~/odin_test/control_command_agx.yaml | head -1); [ -n "$f" ] && test -f ~/odin_test/$f'
  chk "odin-agx scripts are the single-container version" a 'test -f ~/odin_test/agx_lib.sh && grep -q "same" ~/odin_test/start_odin_agx.sh'
  chk "Go2 scripts are the single-container version" g 'grep -q "process master" ~/odin_nav_stack/drv0144/start_odin_remote.sh && grep -q ons2_stop ~/ons2_docker/ons2_start.sh'
  summary
}

# ---------------------------------------------------------------- deploy
do_deploy() {
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  echo "== deploy (replaced files are kept as *.bak_$ts)"
  local hosts="g"; [ "$STACK" = two ] && hosts="g a"
  for h in $hosts; do
    # network.env is required by the scripts: create it only if it is missing and the address was given
    if ! $h 'test -f ~/.config/go2_agx/network.env'; then
      : "${GO2_IP:?network.env is missing: set GO2_IP (and AGX_IP for STACK=two) to create it}"
      $h "mkdir -p ~/.config/go2_agx && printf 'GO2_IP=$GO2_IP\nAGX_IP=${AGX_IP:-}\n' > ~/.config/go2_agx/network.env" && echo "  created network.env"
    fi
  done
  # odin-agx scripts, except the model folders (yolo_gpu, yolo_trt), the benchmark and the system files
  a 'mkdir -p ~/odin_test'
  rsync_to "$AGX_SSH" "${AGX_SSHPASS:-}" "$REPO/go2_agx/agx/" "~/odin_test/" --backup --suffix=".bak_$ts" \
    --exclude detbench --exclude system --exclude docker --exclude yolo_gpu --exclude yolo_trt --exclude control_command_agx.yaml \
    && echo "  scripts -> ~/odin_test/ on $WHO_AGX"
  # control_command_agx.yaml names the relocalization map (site specific): never overwrite an existing one
  if a 'test -f ~/odin_test/control_command_agx.yaml'; then
    echo "  kept the existing ~/odin_test/control_command_agx.yaml (it names the relocalization map)"
  else
    local rm="${RELOC_MAP:-restaurante_prueba.bin}"
    rsync_to "$AGX_SSH" "${AGX_SSHPASS:-}" "$REPO/go2_agx/agx/control_command_agx.yaml" "~/odin_test/control_command_agx.yaml" \
      && a "sed -i 's|relocalization_map_abs_path: \"[^\"]*\"|relocalization_map_abs_path: \"/opt/odin/maps/relocalization/$rm\"|' ~/odin_test/control_command_agx.yaml" \
      && echo "  created ~/odin_test/control_command_agx.yaml with the relocalization map $rm (RELOC_MAP=... to change)"
  fi
  if [ "$STACK" = single ]; then   # the Go2 already has its maps under ~/odin_nav_stack/docker/maps
    a '[ -e ~/odin_test/maps ] || ln -s ~/odin_nav_stack/docker/maps ~/odin_test/maps' && echo "  ~/odin_test/maps -> ~/odin_nav_stack/docker/maps (if it was missing)"
  fi
  rsync_to "$GO2_SSH" "${GO2_SSHPASS:-}" "$REPO/go2_agx/go2/" "~/ons2_docker/" --backup --suffix=".bak_$ts" \
    --include 'ons2_*.sh' --include remote_monitor_start.sh --include Dockerfile --exclude '*' && echo "  Go2: ons2_*.sh, remote_monitor_start.sh, Dockerfile -> ~/ons2_docker/"
  rsync_to "$GO2_SSH" "${GO2_SSHPASS:-}" "$REPO/go2_agx/go2/" "~/odin_nav_stack/drv0144/" --backup --suffix=".bak_$ts" \
    --include 'start_odin_remote.sh' --include 'stop_odin_remote.sh' --include 'start_bridge.sh' --include 'stop_bridge.sh' --include fastdds_udp.xml --exclude '*' \
    && echo "  Go2: master/bridge scripts -> ~/odin_nav_stack/drv0144/"
  echo "  NOT done here (needs sudo): go2_agx/agx/system/*.sh -> /usr/local/bin/ on $WHO_AGX, then: sudo systemctl restart odin-watchdog"
}

# ---------------------------------------------------------------- start / stop
step() { echo "== $1"; shift; "$@" || { echo "  FAILED: $*"; exit 1; }; }
do_start() {
  local ym; ym=$(yolo_mode)
  step "Go2: container ons2-go2 + ROS master"        g '~/odin_nav_stack/drv0144/start_odin_remote.sh'
  local cfg=""
  if [ "$NO_MAP" = 1 ]; then   # a copy of the driver config in odometry mode (the site config is not touched)
    a "[ -f ~/odin_test/control_command_odometry.yaml ] || sed 's|custom_map_mode: 2|custom_map_mode: 0|' ~/odin_test/control_command_agx.yaml > ~/odin_test/control_command_odometry.yaml"
    a "grep -qE '^ *custom_map_mode: 0' ~/odin_test/control_command_odometry.yaml" || { echo "could not make the odometry-mode config"; exit 1; }
    cfg='AGX_CONFIG=$HOME/odin_test/control_command_odometry.yaml '
    echo "== NO_MAP=1: Odin1 in odometry mode, ONS2 map: empty"
  fi
  step "odin-agx on $WHO_AGX: container + Odin1 driver"  a "${cfg}~/odin_test/start_odin_agx.sh $AGX_MODE"
  if [ "$NO_MAP" = 1 ]; then   # odometry mode publishes no map frame: a fixed identity map frame on odom (the driver publishes odom -> map)
    step "odin-agx: fixed map frame (test only)" a '. ~/odin_test/agx_lib.sh 2>/dev/null; D=$HOME/odin_test; . $D/agx_lib.sh; agx_start maptf "rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 odom map"'
  fi
  step "odin-agx: depth"                              a '~/odin_test/start_depth_agx.sh'
  if [ "$ym" = gpu ]; then
    step "odin-agx: YOLO (GPU service + detector)"    a '~/odin_test/start_yolo_gpu_agx.sh'
    step "odin-agx: Florence-2 (frame server + relay)" a '~/odin_test/start_perception_agx.sh'
    step "Go2: ONS2 (no chassis bridge, map $MAP, YOLO on the GPU)" g "~/ons2_docker/ons2_start.sh --map=$MAP --with-yolo-agx"
  else
    step "Go2: ONS2 (no chassis bridge, map $MAP, YOLO on the CPU)" g "~/ons2_docker/ons2_start.sh --map=$MAP --with-yolo"
  fi
}
do_stop() {
  echo "== stop"
  g '~/ons2_docker/ons2_stop.sh'
  [ "$(yolo_mode)" = gpu ] && a '~/odin_test/stop_yolo_gpu_agx.sh'
  a '~/odin_test/stop_odin_agx.sh all'; g '~/odin_nav_stack/drv0144/stop_odin_remote.sh'
}

# ---------------------------------------------------------------- verify
do_verify() {
  local ym; ym=$(yolo_mode)
  echo "== containers (STACK=$STACK, YOLO: $ym)"
  local gnames gpuctr=no; gnames=$(g 'docker ps --format "{{.Names}}"')
  # GPU path with the image odin-gpu:go2 on the machine: the model services run in a third container
  [ "$ym" = gpu ] && [ "$(a 'docker image inspect odin-gpu:go2 >/dev/null 2>&1 && echo yes')" = yes ] && gpuctr=yes
  if [ "$STACK" = single ] && [ "$gpuctr" = yes ]; then
    n=$(grep -cE '^(odin-agx|ons2-go2|odin-gpu)$' <<< "$gnames"); [ "$n" = 3 ] && ok "Go2: exactly the 3 required containers (odin-agx, ons2-go2, odin-gpu)" || bad "Go2: $n of the 3 required containers running"
  elif [ "$STACK" = single ]; then
    n=$(grep -cE '^(odin-agx|ons2-go2)$' <<< "$gnames"); [ "$n" = 2 ] && ok "Go2: exactly the 2 required containers (odin-agx, ons2-go2)" || bad "Go2: $n of the 2 required containers running"
  else
    grep -qx ons2-go2 <<< "$gnames" && ok "Go2: ons2-go2 running" || bad "Go2: ons2-go2 not running"
    n=$(a 'docker ps --format "{{.Names}}" | grep -c "^odin-"'); [ "$n" = 1 ] && ok "AGX: exactly 1 odin container" || bad "AGX: $n odin containers (expected 1)"
  fi
  grep -qx odin-master <<< "$gnames" && bad "old odin-master container still running" || ok "no separate master container"
  grep -qx odin-bridge <<< "$gnames" && echo "  (the optional odin-bridge is running)"

  echo "== odin-agx processes"
  local procs="driver cam depth"; [ "$ym" = gpu ] && procs="$procs yolo capture relay"
  for p in $procs; do
    chk "process $p alive and wanted" a "docker exec odin-agx bash -c 'kill -0 -- -\$(cat /run/odin/$p.pid) && test -f /run/odin/$p.on'"
  done
  if [ "$gpuctr" = yes ]; then
    echo "== odin-gpu processes"
    for p in yolosvc florence; do
      chk "process $p alive and wanted" a "docker exec odin-gpu bash -c 'kill -0 -- -\$(cat /run/odin-gpu/$p.pid) && test -f /run/odin-gpu/$p.on'"
    done
  fi
  if [ "$ym" = gpu ]; then
    chk "YOLO GPU service answers (127.0.0.1:8750)" a 'curl -sf -m 3 http://127.0.0.1:8750/health'
    chk "Florence frame server has frames (127.0.0.1:8730)" a 'curl -sf -m 3 http://127.0.0.1:8730/meta.json | grep -q "\"has_frame\": true"'
  fi

  echo "== ROS graph (seen from the ons2-go2 container)"
  local nodes want t0; want="/pcd2depth_node /yolo_detector /mission_manager /map_server /pc_to_scan /sliding_map_node"
  [ "$ym" = gpu ] && want="$want /florence_cmd_relay /inspection_capture"
  t0=$SECONDS   # the YOLO model loads slowly: wait up to 60 s for all nodes
  while :; do
    nodes=$(gros 'timeout 8 rosnode list'); miss=0
    for n in $want; do grep -q "^$n" <<< "$nodes" || miss=1; done
    [ $miss = 0 ] || [ $((SECONDS-t0)) -ge 60 ] && break; sleep 3
  done
  for n in $want; do grep -q "^$n" <<< "$nodes" && ok "node $n" || bad "node $n missing"; done
  grep -q "^/vel_to_sdk" <<< "$nodes" && bad "SAFETY: chassis bridge /vel_to_sdk is running" || ok "safety: chassis bridge not running"
  chk "odometry /odin1/odometry streams" gros 'timeout 10 rostopic echo -n1 /odin1/odometry/header/seq'
  chk "relocalized (TF map -> imu)" gros 'python3 $HOME/ons2/ros_ws/scripts/launch_gate.py --wait-tf map imu --timeout 8'
  chk "depth image streams" gros 'timeout 15 rostopic echo -n1 /odin1/depth_img_competetion/header/seq'
  chk "YOLO debug image streams" gros 'timeout 20 rostopic echo -n1 /yolo_debug_image/compressed/header/seq'
  chk "/yolo/command list answers on /yolo/console" gros '(timeout 12 rostopic echo -n1 /yolo/console/data > /tmp/yc.txt &) ; sleep 2; rostopic pub -1 /yolo/command std_msgs/String "data: list" >/dev/null; sleep 6; test -s /tmp/yc.txt'

  if [ "${1:-}" = "--resilience" ]; then
    echo "== resilience: restart ONS2, the master and odin-agx must survive"
    local r1 r2 yflag="--with-yolo"; [ "$ym" = gpu ] && yflag="--with-yolo-agx"
    local s1 s2 rc; s1=$(gros 'tmux display-message -p "#{start_time}"' 2>/dev/null)
    r1=$(gros 'rosparam get /run_id'); g '~/ons2_docker/ons2_stop.sh' >/dev/null; sleep 3
    g "~/ons2_docker/ons2_start.sh --map=$MAP $yflag" >/dev/null 2>&1; rc=$?
    s2=$(gros 'tmux display-message -p "#{start_time}"' 2>/dev/null)
    [ $rc = 0 ] && [ -n "$s2" ] && [ "$s1" != "$s2" ] && ok "ONS2 really restarted (new tmux server, start exit 0)" || bad "ONS2 did NOT restart (exit $rc, tmux start $s1 -> $s2)"
    r2=$(gros 'rosparam get /run_id')
    [ -n "$r1" ] && [ "$r1" = "$r2" ] && ok "same ROS master (run_id) after the ONS2 restart" || bad "ROS master changed or lost ($r1 -> $r2)"
    nodes=$(gros 'timeout 8 rosnode list')
    grep -q "^/pcd2depth_node" <<< "$nodes" && ok "odin-agx node /pcd2depth_node still registered" || bad "odin-agx node /pcd2depth_node lost"
    grep -q "^/mission_manager" <<< "$nodes" && ok "ONS2 is back (/mission_manager)" || bad "ONS2 did not come back"
  fi
  summary
}

case "$CMD" in
  check)  do_check ;;
  deploy) do_deploy ;;
  start)  do_start ;;
  verify) do_verify "$@" ;;
  stop)   do_stop ;;
  all)    do_check || { echo "preflight failed: fix it first"; exit 1; }
          do_start && do_verify "$@"; rc=$?
          case " $* " in *" --stop-after "*) do_stop ;; esac; exit $rc ;;
  *) sed -n '2,29p' "$0"; exit 1 ;;
esac
