#!/bin/bash
# GO2. Stops ONS2 (tmux session + container). The ROS master and the AGX keep running.
docker exec ons2-go2 tmux kill-session -t ons2_go2 >/dev/null 2>&1
docker exec ons2-go2 chown -R $(id -u):$(id -g) $HOME/ons2 >/dev/null 2>&1
docker rm -f ons2-go2 >/dev/null 2>&1 && echo "ONS2 stopped. Back to ONS1: ~/odin_nav_stack/drv0144/start_nav_remote.sh"
