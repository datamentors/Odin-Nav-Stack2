#!/bin/bash
# AGX ORIN. Stops the Odin1 driver (and its camera stream) and keeps the driver log. The other components keep running.
#   ~/odin_test/stop_odin_agx.sh                 driver + camera stream
#   ~/odin_test/stop_odin_agx.sh depth           also: depth image
#   ~/odin_test/stop_odin_agx.sh perception      also: Florence-2 frame server + relay + Florence service
#   ~/odin_test/stop_odin_agx.sh all             everything, and removes the containers odin-agx and odin-gpu
# (the GPU detector has its own: stop_yolo_gpu_agx.sh)
D=$HOME/odin_test; . $D/agx_lib.sh
mkdir -p $D/logs
cp $(agx_log driver) $D/logs/driver_$(date +%Y%m%d_%H%M%S).log 2>/dev/null
case "${1:-driver}" in
  driver)     agx_stop_proc driver; agx_stop_proc cam ;;
  depth)      agx_stop_proc driver; agx_stop_proc cam; agx_stop_proc depth ;;
  perception) agx_stop_proc driver; agx_stop_proc cam; agx_stop_proc capture; agx_stop_proc relay; gpu_stop_proc florence; pkill -f "[f]lorence_service.py" ;;
  all)        docker rm -f $AGX_CTR $GPU_CTR >/dev/null 2>&1; pkill -f "[f]lorence_service.py"
              pkill -f "$DET_SVC_PAT" ;;
  *) echo "usage: $0 [driver|depth|perception|all]"; exit 1 ;;
esac
echo "Stopped."
