#!/bin/bash
# GO2. Stops ONS2 (the tmux session and everything started from it). The container ons2-go2, the ROS master inside it
# and the AGX keep running. To stop the master as well: stop_odin_remote.sh.
docker exec ons2-go2 bash -c '
  tp=$(tmux display-message -p "#{pid}" 2>/dev/null); [ -n "$tp" ] || exit 0
  desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
  list="$tp $(desc $tp)"
  tmux kill-server 2>/dev/null; kill -TERM $list 2>/dev/null; sleep 4; kill -KILL $list 2>/dev/null; exit 0' >/dev/null 2>&1
docker exec ons2-go2 chown -R $(id -u):$(id -g) $HOME/ons2 >/dev/null 2>&1
echo "ONS2 stopped (the ROS master keeps running in ons2-go2)."
