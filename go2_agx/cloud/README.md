# cloud: JSON for the cloud side (Phase 1, in progress)

`cloud_publisher.py` is a ROS 1 node. It reads our own topics and publishes JSON strings. It has no AWS code and no credentials. The Pulse `cloud_gateway` (in ardia-robotics) takes these strings and sends them to MQTT. See `Odin-AWS-Streaming-Proposal.md`.

| Kind | Output topic | Source | Rate |
|---|---|---|---|
| `odin.pose.v1` | `/cloud/telemetry` | TF `map` -> `odin1_base_link`, `/odin1/odometry` | `rates_hz.pose` (2) |
| `odin.detections.yolo.v1` | `/cloud/telemetry` | `/yolo_detections_3d`, `/yolo_class_names` | up to `rates_hz.yolo_max` (2), only when the list changes |
| `odin.map.v1` | `/cloud/status` | `/map` | on change + every 60 s |
| `odin.nav.v1` | `/cloud/status` | `/waypoint_manager/status` | on change + every 10 s |
| `odin.stack.v1` | `/cloud/status` | topic ages, env `STACK_IMAGES`, `STACK_COMMIT` | every 10 s |
| `odin.detections.florence.v1` | `/cloud/events` | `/florence/objects_3d` | one per search |

Every message has `kind`, `seq` (per kind) and `t_robot` (robot clock, unix seconds).

- `schemas/`: one JSON Schema per kind. `examples/`: one example per kind (under 4 KB).
- `config/cloud_publisher.yaml`: rates and topic names. A rate of 0 turns a kind off.
- YOLO: `yolo_detector` publishes nothing when it sees nothing, so the node sends one empty list after 1.5 s without detections. Map x/y use TF `camera_frame` -> `map`, the same way `yolo_detector.py` does.
- Florence: `/florence/objects_3d` is only published while `send on` is active in the Florence relay, so searches done with `send off` do not reach the cloud.
- YOLO `bearing_deg`: positive = to the right of the camera axis. `distance_m` = horizontal distance in the camera frame.

Tests (no ROS needed): `python3 tests/test_logic.py` (helpers in `cloud_logic.py`); `python3 tests/test_examples.py` and `python3 tests/test_node.py` (need `jsonschema` and `pyyaml`). `test_node.py` runs the node with fake ROS modules and checks every message it publishes against its schema, topic and size.

Run the node (inside the ons2-go2 container, ROS sourced, nav_interfaces built):
`python3 cloud_publisher.py _config:=config/cloud_publisher.yaml`

Not tested yet: the node on the robot. To confirm: the exact String input topics of the Pulse `cloud_gateway`, the frame names, and the `/map` topic.
