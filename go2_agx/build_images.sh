#!/usr/bin/env bash
# Build all images of the Go2 stack from this repository, in order. Run on the Go2 (JetPack 6, Docker), from anywhere.
#   ./go2_agx/build_images.sh            build everything missing
#   ./go2_agx/build_images.sh --force    rebuild everything
#   BASE_TAG=odin-nav:drv0144-rebuilt ./go2_agx/build_images.sh   test build under another tag (does not touch the working base)
# Needs about 40 GB of free disk, internet access, and the submodules (initialised here).
set -euo pipefail
cd "$(dirname "$0")/.."
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
BASE_TAG=${BASE_TAG:-odin-nav:drv0144}
ONS2_TAG=${ONS2_TAG:-ons2:go2}
ROS_BASE=${ROS_BASE:-dustynv/ros:noetic-pytorch-l4t-r35.3.1}   # what odin-nav:drv0144 was built on (checked with docker history)

have() { [ "$FORCE" = 0 ] && docker image inspect "$1" >/dev/null 2>&1; }
step() { echo; echo "== $*"; }

git submodule update --init ros_ws/src/odin_ros_driver
df -BG --output=avail . | tail -1 | awk '{ if ($1+0 < 40) print "WARNING: only " $1 " free, the images need about 40 GB" }'

step "1/5 $BASE_TAG  (ROS Noetic + conda env neupan, from docker/Dockerfile.global)"
have "$BASE_TAG" && echo "exists, skipped" || \
  docker build -f docker/Dockerfile.global --build-arg BASE_IMAGE="$ROS_BASE" -t "$BASE_TAG" .

step "2/5 $ONS2_TAG  (ONS2 + NeuPAN + Unitree SDK2)"
if have "$ONS2_TAG"; then echo "exists, skipped"; else
  # go2/Dockerfile starts FROM odin-nav:drv0144; build with the chosen base tag
  sed "s#^FROM odin-nav:drv0144#FROM $BASE_TAG#" go2_agx/go2/Dockerfile | docker build -f - -t "$ONS2_TAG" .
fi

step "3/5 odin-drv:0144  (Odin1 driver only)"
have odin-drv:0144 && echo "exists, skipped" || docker build -f go2_agx/agx/docker/Dockerfile.driver -t odin-drv:0144 .

step "4/5 odin-yolo:0144  (the image of the container odin-agx)"
have odin-yolo:0144 && echo "exists, skipped" || docker build -f go2_agx/agx/docker/Dockerfile.yolo --build-arg DRV_IMAGE=odin-drv:0144 -t odin-yolo:0144 .

step "5/5 odin-gpu:go2  (CUDA/TensorRT + Odin-perception venv + Florence-2 model, all inside)"
have odin-gpu:go2 && echo "exists, skipped" || docker build -f go2_agx/agx/docker/Dockerfile.gpu -t odin-gpu:go2 .

echo; docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -E "^(odin-nav:drv0144|ons2:go2|odin-drv:0144|odin-yolo:0144|odin-gpu:go2)"
