# Semantic map (MVP)

Keeps every object the detectors saw, in the **map frame**, with its metadata, and publishes it. This is the first
step of a semantic map: detection → storage → updates. Relations ("chair near the table"), text grounding and saving
to disk are not in it yet.

It only **reads** the existing perception topics; the detector nodes are unchanged.

```
GPU detector (AGX) ── /perception/objects ──┐
Florence-2 relay   ── /florence/objects_3d ─┤          ┌─► /semantic_map/objects   (JSON, latched)
TF map <- camera (at the photo's time) ─────┼─► semantic_map_node ─► /semantic_map/markers   (RViz / Foxglove 3D)
/map (map_server, optional) ────────────────┘          └─► /semantic_map/bev(/compressed)  (top view)
```

## Files

| File | What |
|---|---|
| `semantic_map_node.py` | ROS 1 node (Python 3.8, runs in the `ons2-go2` container) |
| `object_store.py` | the objects and the merge rules; plain Python + numpy, no ROS |
| `bev.py` | the top-view image (OpenCV), no ROS |
| `test_object_store.py` | offline tests: `python3 test_object_store.py [out_dir]` (writes the two test images to `out_dir`) |

## Run

With ONS2 on the Go2 and the GPU detector on the AGX:

```bash
~/ons2_docker/ons2_start.sh --map=<map> --with-yolo-agx --with-semantic-map
```

It starts in the tmux window `semmap` (log: `~/ons2/logs/semantic_map.log`). By hand, inside `ons2-go2`:

```bash
python3 ~/ons2/go2_agx/semantic_map/semantic_map_node.py _min_observations:=3 _assoc_dist_m:=0.6
rostopic echo -n1 /semantic_map/objects/data | python3 -m json.tool
rostopic pub -1 /semantic_map/command std_msgs/String "data: clear"
```

In Foxglove: an Image panel on `/semantic_map/bev/compressed`, and `/semantic_map/markers` in the 3D panel.

Inputs it needs: the GPU detector's `/perception/objects` with 3D points (so `start_yolo_gpu_agx.sh` on the AGX and the
lidar depth image `start_depth_agx.sh`), and TF `map` → camera (the Odin1 driver relocalized on the map, plus
`tf_relay` from `--with-yolo-agx`). Any `DETECTOR` / `TRACKER` works; track IDs (`iou`, `botsort`, `sam2`) make the
merging more stable. Offline test, no ROS: `python3 go2_agx/semantic_map/test_object_store.py /tmp/out`.

## How objects are made and updated

For every object in a `/perception/objects` frame with a 3D point (`xyz`), a score ≥ `~min_score` and a distance ≤
`~max_range_m`:

1. The camera point goes to the map with the TF **at the photo's time stamp** (30 s TF buffer). With no TF for that
   time the detection is dropped (`stats.no_tf`), never placed with the current pose.
2. It is matched to a stored object:
   - **same tracker ID** (same source, seen in the last 3 s, not more than 1.5 m away) → that object;
   - otherwise the **closest object within `~assoc_dist_m`** that has seen this label before. When both have an
     embedding, they must look alike instead (cosine ≥ 0.75), so a "cup" can merge into a "mug";
   - otherwise a **new object** (new integer ID).
3. Update: the position is a score-weighted running mean, capped so an object that was moved still follows. Labels are
   summed score votes (the label is the top one), the size is smoothed, and count, last seen, sources and the last
   detection are updated.
4. Objects are **published after `~min_observations` sightings** (fewer = probably a false detection). Dynamic classes
   (`~dynamic_classes`: person, dog, cat) are forgotten `~dynamic_ttl_s` after they were last seen; other objects stay
   (`~ttl_s` = 0) until `clear` or a restart.

Florence-2 results (`/florence/objects_3d`) go in the same way with source `florence` and score 0.9: Florence gives no
score. Times are message stamps everywhere, so a replayed bag behaves like the live robot.

## `/semantic_map/objects` (std_msgs/String JSON, latched, `~publish_rate` Hz)

```json
{"stamp": 1764240000.1, "frame_id": "map", "count": 2, "stored": 3, "edges": [],
 "stats": {"frames": 812, "placed": 2310, "no_tf": 4, "skipped": 977},
 "objects": [
   {"id": 1, "label": "chair", "labels": {"chair": 31.2, "couch": 1.1}, "confidence": 0.84,
    "position": [3.02, 1.48, 0.31], "size": [0.52, 0.91], "observations": 41,
    "first_seen": 1764239950.2, "last_seen": 1764240000.1, "sources": ["florence", "yoloe"],
    "has_embedding": false,
    "last": {"source": "yoloe", "track_id": 4, "box": [812, 540, 1004, 911], "distance": 2.1, "score": 0.86}}]}
```

- `position`: map frame, m (the object's centre ray at its nearest surface: see `depth3d.py`).
- `size`: width and height in m from the box at the object's depth (not a 3D box).
- `count`: published objects. `stored`: all, including those still under `~min_observations`.
- `edges`: always empty for now (room for spatial relations).
- `embedding` (list of floats) is added per object only with `~publish_embeddings:=true` and when the detector sends
  one. Our detectors do not send one yet: matching then falls back to label + distance.

## `/semantic_map/bev` (+ `/compressed`)

A top view at about 1 Hz, drawn only while someone subscribes:
- **Background:** the `/map` occupancy grid when map_server publishes it (free light, walls dark, unknown grey);
  otherwise a plain 1 m grid (5 m lines darker, map origin as a black dot).
- **Objects:** each a circle sized from its width, coloured by label, with `#id label (sightings)`; objects not seen
  for 30 s are faded.
- **Robot:** red, with its heading and the camera wedge.
- **View:** `~bev_window_m` > 0 shows a square of that size around the robot; 0 shows the whole map, or everything
  drawn when there is no map. At most 1000 px, 50 px/m.

## Parameters

| Param | Default | |
|---|---|---|
| `~objects_topic` | `/perception/objects` | detector JSON |
| `~use_florence` | true | also read `/florence/objects_3d` |
| `~map_topic` | `/map` | occupancy grid for the BEV image |
| `~map_frame`, `~robot_frame`, `~camera_frame` | `map`, `base_link`, `camera_link` | `camera_frame` is used when a message has no frame |
| `~min_score` | 0.4 | weaker detections are ignored |
| `~max_range_m` | 8.0 | farther 3D points are ignored (sparse lidar on small boxes) |
| `~assoc_dist_m` | 0.6 | merge radius |
| `~min_observations` | 3 | sightings before an object is published |
| `~ttl_s` | 0 | forget static objects after this long unseen (0 = never) |
| `~dynamic_classes`, `~dynamic_ttl_s` | `person,dog,cat`, 10 | short-lived classes |
| `~publish_rate` | 1.0 | Hz |
| `~publish_embeddings` | false | include embeddings in the JSON |
| `~bev_window_m` | 0 | BEV crop around the robot (0 = all) |

## Where it comes from, and what is next

`object_store.py` is adapted from the scene-graph store of
[go2-semantic-nav](https://github.com/yusufdxb/go2-semantic-nav) (MIT, see its header). Changes:
- embedding optional;
- tracker IDs used first;
- weighted positions;
- dynamic classes expire, others stay;
- message time stamps only.

Its relation, grounding and goal code is not ported: it has known issues (directional relations are inverted and
measured on the map axes) and needs embeddings we do not produce yet.

Next steps, in order:
1. Save / load per site map (the Odin1 relocalization keeps the map frame fixed across runs).
2. Object embeddings from the GPU service (OpenCLIP), so matching and text queries do not depend on exact labels.
3. Spatial relations (`edges`) and `go <text>` through the map in `yolo_cmd_bridge.py`.
4. Lowering objects that are no longer there (the camera looks at the spot and sees nothing).
