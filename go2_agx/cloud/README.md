# cloud: JSON for the cloud side (Phase 1, in progress)

`cloud_publisher.py` is a ROS 1 node. It reads our own topics and publishes JSON strings. It has no AWS code and no credentials. The Pulse `cloud_gateway` (in ardia-robotics, node name `cloud_gateway`) reads Strings on `~/telemetry`, `~/status` and `~/events` and sends them to MQTT **unchanged**, so the node adds the envelope itself. See `Odin-AWS-Streaming-Proposal.md`.

| Kind | Output topic (MQTT) | Source | Rate |
|---|---|---|---|
| `odin.pose.v1` | `/cloud_gateway/telemetry` (QoS 0) | TF `map` -> `odin1_base_link`, `/odin1/odometry` | `rates_hz.pose` (2) |
| `odin.detections.yolo.v1` | `/cloud_gateway/telemetry` | `/yolo_detections_3d`, `/yolo_class_names` | up to `yolo_max` (2), when the list changes, and every `yolo_refresh_s` (10) |
| `odin.status.v1` | `/cloud_gateway/status` (QoS 1, **retained**) | `/map`, `/waypoint_manager/status`, topic ages, env | on change (at most 1/s), every 2 s while NAVIGATING, else every 10 s |
| `odin.detections.florence.v1` | `/cloud_gateway/events` (QoS 1) | `/florence/objects_3d` | one per search |

**One retained slot.** MQTT keeps one retained message per topic, so everything a late viewer needs (map in use, mission state, health) is in the single `odin.status.v1` message, with parts `map`, `nav` and `stack`. `map` and `nav` are `null` until their topic has been seen.

**Envelope** (`schemas/envelope.schema.json`), around every message:
`schema_version`, `message_id`, `timestamp` (UTC, robot clock), `source`, `robot_id`, `env`, `fleet`, `payload`. The payload always has `kind`, `seq` (per kind) and `t_robot` (unix seconds). The identity comes from `identity:` in the config and must match the Pulse profile.

- `schemas/`: the envelope schema and one payload schema per kind. `examples/`: one full message per kind (under 4 KB).
- `config/cloud_publisher.yaml`: identity, rates and topic names. A rate of 0 turns a kind off.
- YOLO: `yolo_detector` publishes nothing when it sees nothing, so the node sends one empty list after 1.5 s without detections. Map x/y use TF `camera_frame` -> `map`, the same way `yolo_detector.py` does. `bearing_deg`: positive = to the right of the camera axis. `distance_m` = horizontal distance in the camera frame.
- Florence: `/florence/objects_3d` is only published while `send on` is active in the Florence relay, so searches done with `send off` do not reach the cloud.
- Map name: read from the ROS parameter `/map_planner/map_file_name`.

Tests (no ROS needed): `python3 tests/test_logic.py` (helpers in `cloud_logic.py`); `python3 tests/test_examples.py` and `python3 tests/test_node.py` (need `jsonschema` and `pyyaml`). `test_node.py` runs the node with fake ROS modules and checks every message it publishes against the envelope, its payload schema, its topic and the size.

## Run it on the Go2 (stack running)

```
./start_cloud.sh      # 1. cloud_publisher in ons2-go2  2. ROS 1 -> ROS 2 bridge (odin-bridge)  3. Pulse runtime (odin-pulse-cloud)
./test_phase1.sh      # read-only check up to the gateway's ROS 2 inputs and its published counter
./stop_cloud.sh       # stops the runtime and the node (the bridge keeps running; --with-bridge stops it too)
```

- `profile/pulse.yaml`: the Pulse profile, comms only (no control adapter, no teleop, command dispatch off), copied from the Go2 profile. `broker_type: mock` for now. Mounted at `/etc/ardia` in the container `odin-pulse-cloud` (image `ardia/pulse-runtime:dev`).
- DDS: FastRTPS, domain 42, UDP only (`profile/dds/fastdds_udp.xml`), the same as the bridge. The Go2's own Pulse runtime uses CycloneDDS on domain 0, so the two do not see each other.
- The identity is shared with the Go2 runtime. With a real broker, `start_cloud.sh` refuses to start while `ardia-pulse-runtime` is running.
- The mock broker keeps no readable log: `test_phase1.sh` checks the messages the gateway receives and its `outbound_published_count`. The real MQTT output is checked in Phase 3.
- **Phase 2 (camera).** The profile also has the `media_gateway` with the stream `odin_camera` (source `ros_compressed_image` on `/odin1/image/compressed`, 1280x1040, 10 fps, 2000 kbps, `nvv4l2h264`). It is `enabled: false`: the media gateway starts KVS signaling before the GStreamer pipeline, so a stream cannot run without real AWS access. The container runs with the NVIDIA runtime for the encoder. The bridge carries the camera into ROS 2 at the full 10 Hz (measured 27 Nov 2026: 0.77 MB frames, about 8 MB/s).
- `./bench_camera_pipeline.sh [seconds]` measures the video pipeline without AWS (same GStreamer chain as `media_gateway`, one real frame repeated): fps and CPU. It starts one throwaway container and removes it.
- The node alone, inside ons2-go2: `python3 cloud_publisher.py _config:=config/cloud_publisher.yaml`

Seen on the robot (27 Nov 2026, before the envelope and merged status): pose, YOLO, map, nav `IDLE`, stack and Florence. Not tested: the new envelope on the robot, the bridge to ROS 2, the gateway, `NAVIGATING`.
