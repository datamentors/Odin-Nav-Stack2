#!/bin/bash
# JETSON (Go2). Stops the ROS 1 master and its container ons2-go2 (this also stops ONS2 if it is still running), and the
# optional bridge (start_odin_remote.sh). Stop the AGX side first if you can: it loses its master.
LOGDIR=$HOME/ons2/logs; mkdir -p $LOGDIR
docker exec ons2-go2 chown -R $(id -u):$(id -g) $HOME/ons2 >/dev/null 2>&1
[ -f $LOGDIR/master.log ] && mv $LOGDIR/master.log $LOGDIR/master_$(date +%Y%m%d_%H%M%S).log
docker rm -f odin-bridge ons2-go2 odin-master >/dev/null 2>&1
echo "Stopped (master + ONS2 container + bridge). Master log in $LOGDIR"
