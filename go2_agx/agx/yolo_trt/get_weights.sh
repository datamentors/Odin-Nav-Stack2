#!/bin/bash
# Download YOLOv5 weights (official Ultralytics YOLOv5 v7.0 release) to the place the scripts expect, and check them by sha256.
# The checksums are those of the files this stack was built and tested with. An existing file is never overwritten.
#   ./get_weights.sh            yolov5l.pt (93.6 MB): the GPU YOLO and the TensorRT engine   -> ~/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/
#   ./get_weights.sh --cpu      yolov5s.pt (14.8 MB): ONS2's own CPU YOLO (ons2_start.sh --with-yolo)  -> ~/ons2/ros_ws/src/yolo_ros/scripts/models/
#   ./get_weights.sh --all      both
#   ./get_weights.sh --one NAME DEST   a single file with a custom destination (yolov5l or yolov5s)
set -euo pipefail
BASE=https://github.com/ultralytics/yolov5/releases/download/v7.0
sha_of() { case $1 in yolov5l) echo 2f603b7354c25454d1270663a14d8ddc1eea98e5eebc1d84ce0c6e3150fa155f ;; yolov5s) echo 8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee ;; esac; }
fetch() {   # fetch NAME DEST
  local name=$1 dest=$2 sha; sha=$(sha_of "$name")
  if [ -f "$dest" ]; then
    if echo "$sha  $dest" | sha256sum -c - >/dev/null 2>&1; then echo "weights OK: $dest"
    else echo "WARNING: $dest exists but is not the official v7.0 file (kept as it is)"; fi
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "downloading $BASE/$name.pt"
  curl -fL --retry 3 -o "$dest.part" "$BASE/$name.pt"
  echo "$sha  $dest.part" | sha256sum -c - >/dev/null || { rm -f "$dest.part"; echo "checksum mismatch, download removed"; return 1; }
  mv "$dest.part" "$dest" && echo "weights OK: $dest"
}
L=$HOME/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/yolov5l.pt
S=$HOME/ons2/ros_ws/src/yolo_ros/scripts/models/yolov5s.pt
case "${1:-}" in
  "")      fetch yolov5l "$L" ;;
  --cpu)   fetch yolov5s "$S" ;;
  --all)   fetch yolov5l "$L"; fetch yolov5s "$S" ;;
  --one)   fetch "$2" "$3" ;;
  *)       fetch yolov5l "$1" ;;   # old usage: a destination path for yolov5l
esac
