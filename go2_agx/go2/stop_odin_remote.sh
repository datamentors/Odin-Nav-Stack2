#!/bin/bash
# JETSON (Go2). Stops the "remote Odin1" ROS1 master and bridge (start_odin_remote.sh). Stop navigation first.
LOGDIR=$HOME/odin_nav_stack/drv0144/logs; mkdir -p $LOGDIR
docker logs odin-master > $LOGDIR/master_$(date +%Y%m%d_%H%M%S).log 2>&1 || true
docker rm -f odin-bridge odin-master >/dev/null 2>&1
echo "Stopped (master + bridge). Logs in $LOGDIR"
