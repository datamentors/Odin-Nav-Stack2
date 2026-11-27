#!/bin/bash
# Set up the HOST side of the Go2 stack, run ON the Go2 from a clone of this repository. Needs no sudo.
#   ./go2_agx/install.sh --check       only check the host (nothing is changed)
#   ./go2_agx/install.sh               check, then install: submodules, scripts, network.env, YOLO folders, weights
#   ./go2_agx/install.sh --ons2        also copy this repo to ~/ons2 and build the ROS 1 workspace there (in a throwaway ons2:go2 container)
#   ./go2_agx/install.sh --engine      also build the TensorRT engine (about 15 minutes, needs the image odin-gpu:go2)
# Settings (environment): GO2_IP (address of the Go2 on the wired port, default 192.168.123.18)  RELOC_MAP (map file name, default restaurante_prueba.bin)
# NOT done here: the Docker images (./go2_agx/build_images.sh), the map file itself, and the optional watchdog (needs sudo, go2_agx/agx/system/).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODE=install; DO_ONS2=0; DO_ENGINE=0
for a in "$@"; do case $a in --check) MODE=check ;; --ons2) DO_ONS2=1 ;; --engine) DO_ENGINE=1 ;; *) echo "unknown option $a"; exit 1 ;; esac; done
PASS=0; FAIL=0; WARN=0
ok()   { echo "  ok    $*"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "  warn  $*"; WARN=$((WARN+1)); }
chk()  { local what=$1; shift; if "$@" >/dev/null 2>&1; then ok "$what"; else bad "$what"; fi; }

echo "== host check"
chk "aarch64 (Jetson)" test "$(uname -m)" = aarch64
if grep -q "R36" /etc/nv_tegra_release 2>/dev/null; then ok "JetPack 6 ($(head -1 /etc/nv_tegra_release | cut -c1-40))"; else bad "JetPack 6 / L4T R36 (see /etc/nv_tegra_release)"; fi
chk "docker works for this user (no sudo)" docker ps
chk "docker has the nvidia runtime" bash -c "docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'"
root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
free=$(df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "${free:-0}" -ge 60 ]; then ok "free disk for docker: ${free} GB (about 60 GB needed for all images)"; else bad "free disk for docker: ${free:-?} GB (about 60 GB needed)"; fi
chk "internet (github.com)" curl -sI -m 8 https://github.com
if bash -c 'for d in /sys/bus/usb/devices/*; do [ "$(cat $d/idVendor 2>/dev/null)" = 2207 ] && exit 0; done; exit 1'; then ok "Odin1 on USB"; else warn "Odin1 not seen on USB (plug it in before starting the stack)"; fi
[ -f ~/.config/go2_agx/network.env ] && ok "network.env exists" || warn "network.env missing (created by the install)"
for i in odin-nav:drv0144 ons2:go2 odin-yolo:0144 odin-gpu:go2; do
  docker image inspect $i >/dev/null 2>&1 && ok "image $i" || warn "image $i missing (./go2_agx/build_images.sh)"
done
[ -f ~/odin_nav_stack/docker/maps/relocalization/"${RELOC_MAP:-restaurante_prueba.bin}" ] && ok "map ${RELOC_MAP:-restaurante_prueba.bin}" || warn "map ${RELOC_MAP:-restaurante_prueba.bin} missing (copy it to ~/odin_nav_stack/docker/maps/relocalization/)"
echo "== $PASS ok, $WARN warn, $FAIL fail"
[ $FAIL = 0 ] || { echo "Fix the FAIL lines first."; exit 1; }
[ $MODE = check ] && exit 0

ts=$(date +%Y%m%d_%H%M%S)
echo "== submodules"
(cd "$REPO" && git submodule update --init yolov5 ros_ws/src/odin_ros_driver) || { echo "submodules failed"; exit 1; }
[ -f "$REPO/yolov5/hubconf.py" ] || { echo "yolov5 submodule is empty"; exit 1; }

echo "== network.env"
if [ -f ~/.config/go2_agx/network.env ]; then echo "  kept the existing one"
else
  ip=${GO2_IP:-192.168.123.18}
  ip -4 addr 2>/dev/null | grep -q "inet $ip/" || echo "  note: $ip is not an address of this machine; set GO2_IP=... if the wired port uses another one"
  mkdir -p ~/.config/go2_agx && printf 'GO2_IP=%s\nAGX_IP=\n' "$ip" > ~/.config/go2_agx/network.env && echo "  created (GO2_IP=$ip)"
fi

echo "== scripts (replaced files are kept as *.bak_$ts)"
mkdir -p ~/odin_test/yolo_gpu ~/odin_test/yolo_trt ~/ons2_docker ~/odin_nav_stack/drv0144 ~/perception_test ~/odin_nav_stack/docker/maps/relocalization
rsync -a --backup --suffix=".bak_$ts" --exclude detbench --exclude system --exclude docker --exclude yolo_gpu --exclude yolo_trt \
  --exclude control_command_agx.yaml "$REPO/go2_agx/agx/" ~/odin_test/ && echo "  ~/odin_test/"
rsync -a --backup --suffix=".bak_$ts" "$REPO/go2_agx/agx/yolo_gpu/" ~/odin_test/yolo_gpu/ && echo "  ~/odin_test/yolo_gpu/"
rsync -a --backup --suffix=".bak_$ts" "$REPO/go2_agx/agx/yolo_trt/" ~/odin_test/yolo_trt/ && echo "  ~/odin_test/yolo_trt/"
rsync -a --exclude .git "$REPO/yolov5/" ~/odin_test/yolo_gpu/yolov5/ && echo "  ~/odin_test/yolo_gpu/yolov5/ (the YOLOv5 code)"
if [ -f ~/odin_test/control_command_agx.yaml ]; then echo "  kept the existing ~/odin_test/control_command_agx.yaml (it names the relocalization map)"
else
  rm_=${RELOC_MAP:-restaurante_prueba.bin}
  cp "$REPO/go2_agx/agx/control_command_agx.yaml" ~/odin_test/control_command_agx.yaml
  sed -i "s|relocalization_map_abs_path: \"[^\"]*\"|relocalization_map_abs_path: \"/opt/odin/maps/relocalization/$rm_\"|" ~/odin_test/control_command_agx.yaml
  echo "  created ~/odin_test/control_command_agx.yaml (map $rm_)"
fi
[ -e ~/odin_test/maps ] || ln -s ~/odin_nav_stack/docker/maps ~/odin_test/maps
rsync -a --backup --suffix=".bak_$ts" --include 'ons2_*.sh' --include remote_monitor_start.sh --include Dockerfile --exclude '*' "$REPO/go2_agx/go2/" ~/ons2_docker/ && echo "  ~/ons2_docker/"
rsync -a --backup --suffix=".bak_$ts" --include start_odin_remote.sh --include stop_odin_remote.sh --include start_bridge.sh --include stop_bridge.sh \
  --include fastdds_udp.xml --exclude '*' "$REPO/go2_agx/go2/" ~/odin_nav_stack/drv0144/ && echo "  ~/odin_nav_stack/drv0144/"
[ -f ~/.config/go2_agx/detector.env ] || { cp "$REPO/go2_agx/detector.env.example" ~/.config/go2_agx/detector.env && echo "  created ~/.config/go2_agx/detector.env (YOLOv5l, no tracker)"; }
chmod +x ~/odin_test/*.sh ~/odin_test/detector/*.sh ~/odin_test/detector/*.py ~/odin_test/yolo_trt/*.sh ~/ons2_docker/*.sh ~/odin_nav_stack/drv0144/*.sh 2>/dev/null

echo "== YOLO weights"
~/odin_test/yolo_trt/get_weights.sh || echo "  yolov5l weights failed (retry: ~/odin_test/yolo_trt/get_weights.sh)"
# the small CPU weights ONS2 needs for --with-yolo (also used as the fallback when the GPU is not available)
~/odin_test/yolo_trt/get_weights.sh --cpu || echo "  yolov5s weights failed (retry: ~/odin_test/yolo_trt/get_weights.sh --cpu)"

if [ $DO_ONS2 = 1 ]; then
  echo "== ONS2 workspace (~/ons2)"
  docker image inspect ons2:go2 >/dev/null 2>&1 || { echo "  image ons2:go2 missing: ./go2_agx/build_images.sh first"; exit 1; }
  if [ "$REPO" != "$HOME/ons2" ]; then
    rsync -a --exclude .git --exclude ros_ws/build --exclude ros_ws/devel --exclude ros_ws/install --exclude ros2_ws "$REPO/" ~/ons2/ && echo "  repo copied to ~/ons2"
    c=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) && [ -n "$c" ] && echo "$c" > ~/ons2/.stack_commit   # shown by cloud_publisher in odin.status.v1
  fi
  WL='nav_interfaces;ons2_bringup;map_planner;sliding_map;neupan_ros;mission_manager;atomic_move;yolo_ros;odin_vlm_terminal;fish2pinhole;pcd2pgm;pointcloud_saver;unitree_control'
  docker run --rm -u $(id -u):$(id -g) -e HOME=$HOME -v $HOME/ons2:$HOME/ons2 --entrypoint bash ons2:go2 -c \
    "source /opt/ros/noetic/setup.bash && cd $HOME/ons2/ros_ws && catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/opt/conda/envs/neupan/bin/python3 -DCATKIN_WHITELIST_PACKAGES='$WL'" \
    && echo "  workspace built" || { echo "  workspace build FAILED"; exit 1; }
fi

[ $DO_ENGINE = 1 ] && { echo "== TensorRT engine"; ~/odin_test/yolo_trt/build_engine.sh || exit 1; }

echo; echo "== done. Still to do by hand:"
[ -f ~/odin_nav_stack/docker/maps/relocalization/"${RELOC_MAP:-restaurante_prueba.bin}" ] || echo "  - copy the map to ~/odin_nav_stack/docker/maps/relocalization/${RELOC_MAP:-restaurante_prueba.bin}"
[ -f ~/odin_test/yolo_trt/yolov5l_fp16.engine ] || echo "  - TensorRT engine (optional, faster YOLO): ./go2_agx/install.sh --engine   (or ~/odin_test/yolo_trt/build_engine.sh)"
[ -f ~/ons2/ros_ws/devel/setup.bash ] || echo "  - ONS2 workspace: ./go2_agx/install.sh --ons2"
ls ~/odin_test/models/rfdetr_*_fp16.engine >/dev/null 2>&1 || echo "  - optional RF-DETR detector: ~/odin_test/detector/get_models.sh rfdetr   (then DETECTOR=rfdetr in ~/.config/go2_agx/detector.env)"
echo "  - then start: see go2_agx/README.md, or GO2_SSH=local ./go2_agx/test_stack.sh all"
