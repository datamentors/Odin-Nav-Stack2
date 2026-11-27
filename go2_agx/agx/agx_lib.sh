#!/bin/bash
# AGX ORIN. Helpers shared by the start_*/stop_* scripts, the watchdog and the autostart. Source it, do not run it.
# Everything ROS 1 on the AGX runs in ONE container, "odin-agx" (image odin-yolo:0144, which contains the Odin1 driver
# build and the YOLO runtime). Inside it every component is a process with a short name:
#   driver (roslaunch odin1_headless)   cam (MJPEG stream)   depth (pcd2depth)   yolo (detector node)
#   capture (Florence frame server)     relay (Florence <-> Foxglove)
# Per component, inside the container:  /run/odin/NAME.pid  (process group leader)  /run/odin/NAME.on  (wanted: it was
# started on purpose and not stopped on purpose; the watchdog only relaunches wanted components)
# and on the host (mounted):  $D/logs/agx/NAME.log
# Needs $D (= ~/odin_test) set before sourcing.
AGX_CTR=${AGX_CTR:-odin-agx}      # the override is for tests only
AGX_IMAGE=odin-yolo:0144
AGX_CONFIG=${AGX_CONFIG:-$D/control_command_agx.yaml}   # driver config mounted into the container (another file: e.g. odometry mode without a map)
AGX_LOGS=$D/logs/agx
AGX_LEGACY_CTRS="odin-cam odin-depth odin-yolo-gpu odin-perception-cap odin-perception-cmd"   # the pre-merge containers

agx_running() { [ "$(docker inspect -f '{{.State.Running}}' $AGX_CTR 2>/dev/null)" = true ]; }
agx_log() { echo $AGX_LOGS/$1.log; }

# agx_ensure "<ROS env as docker -e args>": makes sure the container runs with that ROS master. An existing one with the
# same master is kept (so restarting the driver does not touch the other components); with another master it is recreated.
agx_ensure() {
  local want_uri have_uri
  want_uri=$(grep -o 'ROS_MASTER_URI=[^ ]*' <<< "$1")
  if agx_running; then
    have_uri=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $AGX_CTR | grep '^ROS_MASTER_URI=')
    [ "$have_uri" = "$want_uri" ] && return 0
    echo "$AGX_CTR runs with another ROS master ($have_uri): recreating it, all AGX components stop"
  fi
  docker rm -f $AGX_CTR $AGX_LEGACY_CTRS >/dev/null 2>&1
  mkdir -p $AGX_LOGS
  # the Odin-perception code is inside the image (label odin.perception=baked); only an older image needs the host folder
  PERC_MOUNT=''; perc_baked $AGX_IMAGE || PERC_MOUNT="-v $HOME/odin-perception:/opt/odin-perception"
  docker run -d --init --name $AGX_CTR --network host --privileged -v /dev/bus/usb:/dev/bus/usb $1 -e PYTHONUNBUFFERED=1 \
    -v $AGX_CONFIG:/opt/odin/ros_ws/src/odin_ros_driver/config/control_command.yaml:ro \
    -v $D/maps/relocalization:/opt/odin/maps/relocalization:ro \
    $PERC_MOUNT \
    -v $AGX_LOGS:/var/log/odin \
    --entrypoint sleep $AGX_IMAGE infinity >/dev/null || return 1
  # /calib.yaml: the driver writes the Odin1 calibration here; depth, YOLO and the Florence relay read it
  docker exec $AGX_CTR bash -c "mkdir -p /run/odin; ln -sf /root/.ros/odin_ros_driver/calib.yaml /calib.yaml"
}

# perc_baked IMAGE: true when the image already contains Odin-perception (no host folder needed)
perc_baked() { [ "$(docker image inspect -f '{{index .Config.Labels "odin.perception"}}' "$1" 2>/dev/null)" = baked ]; }

agx_cp() { docker cp "$1" $AGX_CTR:"$2"; }   # copy a script in (the container keeps its own copy, started later)

# agx_start NAME "command": (re)starts a component. The command runs in bash with ROS Noetic and the workspace sourced.
agx_start() {
  agx_stop_proc "$1"
  docker exec $AGX_CTR bash -c ': > /var/log/odin/$1.log; touch /run/odin/$1.on' _ "$1" || return 1
  docker exec -d $AGX_CTR setsid bash -c '
      echo $$ > /run/odin/$1.pid
      source /opt/ros/noetic/setup.bash
      [ -f /opt/odin/ros_ws/devel_isolated/setup.bash ] && source /opt/odin/ros_ws/devel_isolated/setup.bash
      exec bash -c "$2" >> /var/log/odin/$1.log 2>&1' _ "$1" "$2"
  for _ in 1 2 3 4 5 6; do agx_alive "$1" && return 0; sleep 0.5; done
  return 1
}

# agx_stop_proc NAME: SIGINT to the whole process group (roslaunch / rospy shut down cleanly), SIGKILL after 10 s.
# It also clears the "wanted" mark: this is a stop on purpose.
agx_stop_proc() {
  agx_running || return 0
  docker exec $AGX_CTR bash -c '
      rm -f /run/odin/$1.on
      p=$(cat /run/odin/$1.pid 2>/dev/null); [ -n "$p" ] || exit 0
      kill -INT -- -$p 2>/dev/null
      for i in $(seq 1 10); do kill -0 -- -$p 2>/dev/null || break; sleep 1; done
      kill -KILL -- -$p 2>/dev/null; rm -f /run/odin/$1.pid; exit 0' _ "$1"
}

agx_alive()  { agx_running && docker exec $AGX_CTR bash -c 'p=$(cat /run/odin/$1.pid 2>/dev/null); [ -n "$p" ] && kill -0 -- -$p 2>/dev/null' _ "$1"; }
agx_wanted() { agx_running && docker exec $AGX_CTR test -f /run/odin/$1.on; }
agx_age()    { docker exec $AGX_CTR bash -c 'echo $(( $(date +%s) - $(stat -c %Y /run/odin/$1.pid 2>/dev/null || date +%s) ))' _ "$1"; }

# ---- optional third container "odin-gpu": the GPU model services (YOLOv5l TensorRT/PyTorch on :8750, Florence-2 on :8740) ----
# Used when the image odin-gpu:go2 exists on this machine (Go2: docker build -f docker/Dockerfile.gpu); otherwise the services
# run on the host as before (AGX Orin: venv in ~/odin-perception). Force with GPU_MODE=container|host.
# The container has its own CUDA / cuDNN / TensorRT (JetPack image) and, when the image has the label odin.perception=baked,
# the venv + Odin-perception + Florence-2 model (/opt/odin-perception); the YOLO weights/engine and the service scripts are
# mounted from the host at the same paths (an older image without the label also gets ~/odin-perception mounted). Host network: 127.0.0.1:8750 / :8740 do not change.
# Processes inside (names yolosvc, florence): /run/odin-gpu/NAME.pid + NAME.on; logs on the host: $D/logs/gpu/NAME.log
GPU_CTR=${GPU_CTR:-odin-gpu}
GPU_IMAGE=${GPU_IMAGE:-odin-gpu:go2}
GPU_LOGS=$D/logs/gpu
gpu_in_container() {
  case "${GPU_MODE:-auto}" in container) return 0 ;; host) return 1 ;; *) docker image inspect $GPU_IMAGE >/dev/null 2>&1 ;; esac
}
# gpu_perc: where the Odin-perception folder (venv, code, Florence-2 model) is as seen by the GPU services
gpu_perc() { if gpu_in_container && perc_baked $GPU_IMAGE; then echo /opt/odin-perception; else echo $HOME/odin-perception; fi; }
gpu_running() { [ "$(docker inspect -f '{{.State.Running}}' $GPU_CTR 2>/dev/null)" = true ]; }
gpu_log() { echo $GPU_LOGS/$1.log; }
gpu_ensure() {
  gpu_running && return 0
  docker rm -f $GPU_CTR >/dev/null 2>&1
  mkdir -p $GPU_LOGS $HOME/perception_test
  GPU_PERC_MOUNT=''; perc_baked $GPU_IMAGE || GPU_PERC_MOUNT="-v $HOME/odin-perception:$HOME/odin-perception"
  docker run -d --init --name $GPU_CTR --network host --runtime nvidia \
    -e HOME=$HOME -e PYTHONUNBUFFERED=1 -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    $GPU_PERC_MOUNT -v $D:$D -v $HOME/perception_test:$HOME/perception_test \
    --entrypoint sleep $GPU_IMAGE infinity >/dev/null || return 1
  docker exec $GPU_CTR mkdir -p /run/odin-gpu
}
# gpu_start NAME "command": (re)starts a service in odin-gpu (own process group, log on the host)
gpu_start() {
  gpu_stop_proc "$1"
  docker exec $GPU_CTR bash -c ': > '$GPU_LOGS'/$1.log; touch /run/odin-gpu/$1.on' _ "$1" || return 1
  docker exec -d $GPU_CTR setsid bash -c '
      echo $$ > /run/odin-gpu/$1.pid
      exec bash -c "$2" >> '$GPU_LOGS'/$1.log 2>&1' _ "$1" "$2"
  for _ in 1 2 3 4 5 6; do gpu_alive "$1" && return 0; sleep 0.5; done
  return 1
}
gpu_stop_proc() {
  gpu_running || return 0
  docker exec $GPU_CTR bash -c '
      rm -f /run/odin-gpu/$1.on
      p=$(cat /run/odin-gpu/$1.pid 2>/dev/null); [ -n "$p" ] || exit 0
      kill -INT -- -$p 2>/dev/null
      for i in $(seq 1 10); do kill -0 -- -$p 2>/dev/null || break; sleep 1; done
      kill -KILL -- -$p 2>/dev/null; rm -f /run/odin-gpu/$1.pid; exit 0' _ "$1"
}
gpu_alive() { gpu_running && docker exec $GPU_CTR bash -c 'p=$(cat /run/odin-gpu/$1.pid 2>/dev/null); [ -n "$p" ] && kill -0 -- -$p 2>/dev/null' _ "$1"; }
