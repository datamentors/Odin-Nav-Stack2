#!/bin/bash
# Download the YOLOv5l weights (official Ultralytics YOLOv5 v7.0 release, 93.6 MB) to the place the scripts expect, and check them.
# The checksum is the one of the file this stack was built and tested with. An existing file is never overwritten.
#   ./get_weights.sh [DEST]      default DEST: ~/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/yolov5l.pt
set -euo pipefail
URL=https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
SHA=2f603b7354c25454d1270663a14d8ddc1eea98e5eebc1d84ce0c6e3150fa155f
DEST=${1:-$HOME/odin_test/yolo/ros_ws/src/yolo_ros/scripts/models/yolov5l.pt}
if [ -f "$DEST" ]; then
  if echo "$SHA  $DEST" | sha256sum -c - >/dev/null 2>&1; then echo "weights OK: $DEST"
  else echo "WARNING: $DEST exists but is not the official v7.0 file (kept as it is)"; fi
  exit 0
fi
mkdir -p "$(dirname "$DEST")"
echo "downloading $URL"
curl -fL --retry 3 -o "$DEST.part" "$URL"
echo "$SHA  $DEST.part" | sha256sum -c - >/dev/null || { rm -f "$DEST.part"; echo "checksum mismatch, download removed"; exit 1; }
mv "$DEST.part" "$DEST" && echo "weights OK: $DEST"
