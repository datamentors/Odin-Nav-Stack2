#!/bin/bash
# Models for the detector service that are not YOLOv5 (YOLOv5l: ~/odin_test/yolo_trt/get_weights.sh + build_engine.sh).
# Everything goes to ~/odin_test/models/ ($ODIN_MODELS). TensorRT engines are built inside the container odin-gpu (its own
# TensorRT) and only load with that TensorRT version: rebuild them after a JetPack / image update. Robot idle while building.
#   ~/odin_test/detector/get_models.sh rfdetr [medium|large ...]   RF-DETR -> ONNX (throwaway venv, CPU) -> FP16 engine (default: large)
#   ~/odin_test/detector/get_models.sh reid                        OSNet x0.25 Re-ID -> FP16 engine (tracker botsort_reid; image with WITH_TRACKERS=1)
#   ~/odin_test/detector/get_models.sh sam2 [tiny|small]           SAM 2.1 checkpoint (tracker sam2; image with WITH_TRACKERS=1)
#   ~/odin_test/detector/get_models.sh yoloe [11s|11m|11l|11l-pf]  YOLOE weights + its text encoder (DETECTOR=yoloe; image with WITH_YOLOE=1)
set -euo pipefail
D=$HOME/odin_test; . $D/agx_lib.sh
M=${ODIN_MODELS:-$D/models}; mkdir -p $M
HERE=$D/detector
TRTEXEC=/usr/src/tensorrt/bin/trtexec
gpu_ensure || { echo "container $GPU_CTR did not start (image $GPU_IMAGE built?)"; exit 1; }
gx() { docker exec -e HOME=$HOME $GPU_CTR bash -c "$1"; }
own() { docker exec $GPU_CTR chown -R $(id -u):$(id -g) $M; }   # the container runs as root

case "${1:-}" in
  rfdetr)
    shift; variants=${*:-large}
    need=""; for v in $variants; do [ -f $M/rfdetr_$v.onnx ] || need="$need $v"; done
    if [ -n "$need" ]; then
      echo "== RF-DETR -> ONNX ($need) in a throwaway venv (downloads rfdetr + a CPU torch, ~1-2 GB)"
      gx "python3 -m venv /tmp/rfdetr-export && /tmp/rfdetr-export/bin/pip install -q rfdetr==1.12.1 onnx==1.17.0 onnxsim \
          && cd $M && /tmp/rfdetr-export/bin/python $HERE/export_rfdetr.py $M $need; rc=\$?; rm -rf /tmp/rfdetr-export; exit \$rc"
    fi
    for v in $variants; do
      [ -f $M/rfdetr_${v}_fp16.engine ] && { echo "engine exists: $M/rfdetr_${v}_fp16.engine (delete it to rebuild)"; continue; }
      echo "== trtexec rfdetr_$v (5-10 min)"
      gx "$TRTEXEC --onnx=$M/rfdetr_$v.onnx --saveEngine=$M/rfdetr_${v}_fp16.engine --fp16 --memPoolSize=workspace:2048" \
        > $M/build_rfdetr_$v.log 2>&1 || { echo "trtexec failed: tail $M/build_rfdetr_$v.log"; tail -5 $M/build_rfdetr_$v.log; exit 1; }
      echo "OK $M/rfdetr_${v}_fp16.engine"
    done ;;
  reid)
    echo "== OSNet x0.25 MSMT17 -> ONNX -> FP16 engine"
    gx "$(gpu_perc)/.venv/bin/python $HERE/export_reid.py $M/osnet_x0_25_msmt17.pt" ;;
  sam2)
    v=${2:-tiny}; mkdir -p $M/sam2
    f=sam2.1_hiera_$v.pt
    if [ ! -f $M/sam2/$f ]; then
      curl -fL --retry 3 -o $M/sam2/$f.part https://dl.fbaipublicfiles.com/segment_anything_2/092824/$f
      mv $M/sam2/$f.part $M/sam2/$f
    fi
    echo "OK $M/sam2/$f" ;;
  yoloe)
    v=${2:-11l}; A=https://github.com/ultralytics/assets/releases/download/v8.4.0
    case $v in
      11l-pf) files="yoloe-11l-seg-pf.pt" ;;                 # prompt-free: built-in vocabulary, no text encoder
      *)      files="yoloe-$v-seg.pt mobileclip_blt.ts" ;;   # text prompts: the encoder must sit next to the weights
    esac
    for f in $files; do
      if [ ! -f $M/$f ]; then curl -fL --retry 3 -o $M/$f.part $A/$f && mv $M/$f.part $M/$f; fi
      echo "OK $M/$f"
    done ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
own 2>/dev/null || true
