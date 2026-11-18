#!/bin/bash
# AGX ORIN. Stops the Odin1 driver and keeps its log.
mkdir -p $HOME/odin_test/logs
docker logs odin-agx > $HOME/odin_test/logs/driver_$(date +%Y%m%d_%H%M%S).log 2>&1 || true
docker rm -f odin-agx odin-cam >/dev/null 2>&1; echo "Stopped."
