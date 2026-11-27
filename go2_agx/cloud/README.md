# cloud: the nav stack's side of the link to AWS

`cloud_publisher.py` is a ROS 1 node. It reads our own topics (pose, map, mission state, YOLO, Florence-2) and publishes them as JSON strings on three ROS topics. **That is all this folder does.** It has no AWS code, no MQTT and no credentials.

The cloud side is not here. The cloud gateway (MQTT to AWS IoT) and the media gateway (camera over KVS WebRTC) are part of Ardia's Pulse runtime: the profile `robots/unitree-go2-odin-000` in **ardia-robotics, branch `go2-demo`**. The two sides share only ROS topics, carried by the ROS 1 -> ROS 2 bridge (`odin-bridge`, FastRTPS, domain 42):

```
this repo (ROS 1 containers)                      ardia-robotics, go2-demo (pulse-runtime, ROS 2)
  cloud_publisher --/cloud_gateway/{telemetry,status,events}--> cloud_gateway --> AWS IoT (MQTT)
  Odin1 driver    --/odin1/image/compressed------------------> media_gateway --> AWS KVS (WebRTC)
```

| Kind | Output topic | Source | Rate |
|---|---|---|---|
| `odin.pose.v1` | `/cloud_gateway/telemetry` | TF `map` -> `odin1_base_link`, `/odin1/odometry` | `rates_hz.pose` (2) |
| `odin.detections.yolo.v1` | `/cloud_gateway/telemetry` | `/yolo_detections_3d`, `/yolo_class_names` | up to `yolo_max` (2), when the list changes, and every `yolo_refresh_s` (10) |
| `odin.status.v1` | `/cloud_gateway/status` | `/map`, `/waypoint_manager/status`, topic ages, env | on change (at most 1/s), every 2 s while NAVIGATING, else every 10 s |
| `odin.detections.florence.v1` | `/cloud_gateway/events` | `/florence/objects_3d` | one per search |

The topic names are the String inputs of the Pulse `cloud_gateway` (node name `cloud_gateway`, `~/telemetry`, `~/status`, `~/events`). It forwards them to MQTT **unchanged**, so the node adds the envelope. The status topic is retained on the cloud side and MQTT keeps one retained message per topic, so everything a late viewer needs (map in use, mission state, health) is in the single `odin.status.v1`, with parts `map`, `nav` and `stack` (`map` and `nav` are `null` until their topic has been seen).

**Envelope** (`schemas/envelope.schema.json`), around every message: `schema_version`, `message_id`, `timestamp` (UTC, robot clock), `source`, `robot_id`, `env`, `fleet`, `payload`. The payload always has `kind`, `seq` (per kind) and `t_robot` (unix seconds). The identity comes from `identity:` in the config and must match the Pulse profile.

- `schemas/`: the envelope schema and one payload schema per kind. `examples/`: one full message per kind (under 4 KB).
- `config/cloud_publisher.yaml`: identity, rates and topic names. A rate of 0 turns a kind off.
- YOLO: `yolo_detector` publishes nothing when it sees nothing, so the node sends one empty list after 1.5 s without detections. Map x/y use TF `camera_frame` -> `map`, the same way `yolo_detector.py` does. `bearing_deg`: positive = to the right of the camera axis. `distance_m` = horizontal distance in the camera frame.
- Florence: `/florence/objects_3d` is only published while `send on` is active in the Florence relay, so searches done with `send off` do not reach the cloud.
- Map name: read from the ROS parameter `/map_planner/map_file_name`.

## Run on the Go2

`cloud_publisher` is one more tmux window of the ONS2 start, like the other components. It runs from the repo copy in `~/ons2` (`install.sh --ons2` copies the repo there and records the commit in `~/ons2/.stack_commit`), so there is no image rebuild and no `docker cp`:

```
~/ons2_docker/ons2_start.sh --map=<map_name> --with-yolo-agx --with-cloud
docker exec -it ons2-go2 tmux attach -t ons2_go2     # window "cloud"; log: ~/ons2/logs/cloud_node.log
```

`--with-cloud` also starts the ROS 1 -> ROS 2 bridge (`odin-bridge`) if it is not running. `ons2_stop.sh` stops the node with the rest of ONS2; the bridge keeps running (`stop_bridge.sh` stops it). With the test script: `CLOUD=1 GO2_SSH=local ./go2_agx/test_stack.sh all` starts everything with the cloud window and verifies the messages (`tests/verify_topics.py`, run inside `ons2-go2`).

Then, in ardia-robotics on the branch `go2-demo`: `scripts/go2_odin/start.sh` and `scripts/go2_odin/check.sh`.
By hand inside ons2-go2 (development): `python3 cloud_publisher.py _config:=config/cloud_publisher.yaml`.

## Tests (no ROS needed)

`python3 tests/test_logic.py` (helpers in `cloud_logic.py`) and `python3 tests/test_verify_topics.py`; `python3 tests/test_examples.py` and `python3 tests/test_node.py` (need `jsonschema` and `pyyaml`). `test_node.py` runs the node with fake ROS modules and checks every message it publishes against the envelope, its payload schema, its topic and the size.

## Status (27 Nov 2026)

Seen on the Go2: pose, YOLO, map, mission state and Florence from the node; the messages with the envelope through the bridge into the cloud gateway (mock broker), and the camera through the bridge at 10 Hz. Not tested: `NAVIGATING`, the events topic through the gateway, anything with real AWS.
