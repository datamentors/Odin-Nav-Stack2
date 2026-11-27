# GPU detector: models, launch, topics

A selectable per-frame detector and tracker for the Go2 + Odin1 stack. Florence-2 runs next to it as the
on-demand search. The Foxglove commands are in `go2_agx/README.md` §3; the service's HTTP API is in the header
of `detector_service.py`.

## 1. How the pieces fit

```
text query ──► Florence-2 (on demand, ~1 s, any phrase) ──► box ──┐
            └► YOLOE "find …" (every frame, open vocabulary) ─────┤
                                                                  ▼
camera ─► per-frame DETECTOR (yolov5 | rfdetr | yoloe) ─► TRACKER (none | iou | botsort | botsort_reid | sam2)
                                                                  ▼
                                    depth ─► 3D ─► /perception/* and /yolo_* topics
```

- **First detection:** Florence-2 is the slow, on-demand search for any phrase. YOLOE is the fast option for words you
  give it, and it runs on every frame.
- **Per-frame detector:** the light model that runs on every camera frame.
- **Tracker:** gives every detection an ID across frames, or (SAM 2) follows one object you point at with a box.
- **Target:** the one tracked object to follow, picked with a box, a class name or a track ID.

## 2. Selecting the models

Edit `~/.config/go2_agx/detector.env` on the machine that runs the detector (the AGX, or the Go2 in single-machine
mode). Start from `go2_agx/detector.env.example`. The watchdog and autostart read the same file.

```bash
DETECTOR=yolov5          # yolov5 | rfdetr | yoloe
TRACKER=none             # none | iou | botsort | botsort_reid | sam2
DETECTOR_WEIGHTS=        # empty = defaults below; comma-separated fallbacks allowed
DETECTOR_CONF=0.25       # confidence threshold at start
YOLOE_CLASSES=           # yoloe only: e.g. "person, chair, fire extinguisher" (empty = 80 COCO classes)
DETECTOR_SCALE=          # model image size in % of the camera image (empty = auto, see below)
SAM2_IMAGE_SIZE=512      # sam2 only: the size SAM 2 works at (multiple of 32; 384 / 256 = faster, coarser masks)
SAM2_MODEL=tiny          # sam2 only: tiny | small
YOLOE_IMGSZ=640          # yoloe only: largest input size
```

**Speed vs detail: smaller images for the models.** The detector and tracker can run on a downsampled copy of the
camera image (OpenCV `INTER_AREA`). Everything else stays at the original 1600 × 1296: boxes, masks, 3D points and the
debug image are scaled back to it.

| `DETECTOR_SCALE` | Image the models get |
|---|---|
| empty (default) | auto: longest side = the detector's input size (640 px for YOLOv5 / YOLOE ≈ 40 %; RF-DETR 576 / 704) |
| `30` | 30 % of the camera image (480 × 389); YOLOv5 and YOLOE then run at 480 instead of 640 |
| `100` | the full camera image (the detectors still shrink it to their own maximum) |

YOLOv5 and YOLOE run at the smaller size instead of enlarging it back. RF-DETR's TensorRT engine has a fixed size, so it
gains nothing. SAM 2 always works at `SAM2_IMAGE_SIZE`: lower that value to make `sam2` faster. Tune it live, without a
restart: `rostopic pub -1 /perception/set_scale std_msgs/Float32 "data: 30"` (0 = auto). Put the chosen value in
`detector.env` to keep it.

| `DETECTOR` | What | Default weights | Needs |
|---|---|---|---|
| `yolov5` | YOLOv5l, 80 COCO classes (TensorRT, falls back to PyTorch) | `~/odin_test/yolo_trt/yolov5l_fp16.engine`, then `yolov5l.pt` | `yolo_trt/get_weights.sh`, `yolo_trt/build_engine.sh` |
| `rfdetr` | RF-DETR Large / Medium, COCO, TensorRT | `~/odin_test/models/rfdetr_large_fp16.engine` | `get_models.sh rfdetr large` |
| `yoloe` | YOLOE-11 seg: any words, with masks | `~/odin_test/models/yoloe-11l-seg.pt` (+ text encoder) | image built with `WITH_YOLOE=1`, `get_models.sh yoloe 11l` |

| `TRACKER` | What | Needs |
|---|---|---|
| `none` | detections only | — |
| `iou` | an ID per detection (simple, no extra packages) | — |
| `botsort` | BoT-SORT, compensates for camera motion | image built with `WITH_TRACKERS=1` |
| `botsort_reid` | BoT-SORT + appearance re-identification (keeps IDs through occlusions) | `WITH_TRACKERS=1`, `get_models.sh reid` |
| `sam2` | SAM 2.1: follows one object given by a box, any class, with a mask | `WITH_TRACKERS=1`, `get_models.sh sam2 tiny` |

To try a combination for one start without editing the file:

```bash
~/odin_test/start_yolo_gpu_agx.sh --detector yoloe --tracker botsort --scale 30
```

That choice lasts until the next restart; a watchdog restart goes back to `detector.env`.

Build the GPU image with the optional extras once, on the robot:

```bash
WITH_TRACKERS=1 WITH_YOLOE=1 ./go2_agx/build_images.sh --force
```

Models go to `~/odin_test/models/`. TensorRT engines are built inside the `odin-gpu` container and only load with that
TensorRT version.

```bash
~/odin_test/detector/get_models.sh rfdetr large     # RF-DETR -> ONNX -> FP16 engine
~/odin_test/detector/get_models.sh reid             # Re-ID engine for botsort_reid
~/odin_test/detector/get_models.sh sam2 tiny        # SAM 2.1 checkpoint
~/odin_test/detector/get_models.sh yoloe 11l        # YOLOE weights + text encoder (11s / 11m / 11l / 11l-pf)
```

## 3. Speed and known behaviour

From the detector benchmark (branch `go2_rf_detr_bench`, AGX Orin 30 W, TensorRT FP16), not yet measured through this
service on the robot:

| Setup | Per frame | With Florence-2 busy |
|---|---|---|
| YOLOv5l | 55 ms (18 FPS) | 80 ms |
| RF-DETR Large / Medium | 42 ms (24 FPS) / 32 ms (32 FPS) | 80 / 60 ms |
| RF-DETR L + BoT-SORT / + Re-ID | 57 ms / 85 ms (p95 140 ms) | |
| SAM 2.1 tiny @512, 1 object | ~85 ms | ~145 ms |
| YOLOE-11s seg | not measured on the Orin (6 ms on a laptop RTX GPU; a class change ~1.5 s) | |

- Tiling the image was slower and less accurate.
- SAM 2 does not re-acquire an object reliably after it leaves the view.
- Re-ID keeps IDs through short occlusions (~4 s) but does not recognise a person who comes back somewhere else.
- The official `sam2` package has no streaming video API, so `sam2` prompts the image predictor with the previous box each
  frame (no memory); a sam2 build with `build_sam2_camera_predictor` is used automatically when installed.
- YOLOE runs in PyTorch (a TensorRT export would fix its classes). With `yoloe` the indoor ignore list is not applied.
- A box selection is matched against the tracks of the frame closest to its time stamp (the service keeps 5 s), so a
  Florence-2 box from a few seconds earlier still picks the right track.
- Licences: YOLOv5, Ultralytics (YOLOE) and BoxMOT (BoT-SORT) are AGPL-3.0; RF-DETR and SAM 2 are Apache-2.0.

## 4. Launching

The order matters. Commands are for the AGX layout; in single-machine mode run the AGX ones on the Go2 and use
`same` instead of `go2`.

1. On the Go2, start the ROS master container: `~/odin_nav_stack/drv0144/start_odin_remote.sh`
2. On the AGX, start the Odin1 driver and the depth image (the autostart service does this at boot):
   `~/odin_test/start_odin_agx.sh go2`, then `~/odin_test/start_depth_agx.sh`
3. On the AGX, start the detector (GPU service + ROS node): `~/odin_test/start_yolo_gpu_agx.sh`
4. Optionally, on the AGX, start Florence-2: `~/odin_test/start_perception_agx.sh`
5. On the Go2, start ONS2 with the detector on the AGX: `~/ons2_docker/ons2_start.sh --map=<map> --with-yolo-agx`

| Process | Container | What |
|---|---|---|
| `yolosvc` | `odin-gpu` (or the host) | `detector_service.py`: detector + tracker, HTTP on `127.0.0.1:8750` |
| `yolo` | `odin-agx` | `detector_node.py`, ROS node `/yolo_detector`: publishes the topics below |
| `florence`, `capture`, `relay` | `odin-gpu` / `odin-agx` | Florence-2 service, frame server, Foxglove relay |

Check and stop:

```bash
curl -s http://127.0.0.1:8750/health
tail -f ~/odin_test/logs/agx/yolo.log ~/odin_test/logs/gpu/yolosvc.log
~/odin_test/stop_yolo_gpu_agx.sh
```

## 5. Inputs

| Topic | Type | Description |
|---|---|---|
| `/odin1/image/undistorted` | `sensor_msgs/Image` | camera frames; only the newest is processed |
| `/odin1/depth_img_competetion` | `sensor_msgs/Image` | lidar depth projected into the same image, used for 3D |
| `/yolo/set_conf` | `std_msgs/Float32` | confidence threshold, live |
| `/perception/set_scale` | `std_msgs/Float32` | model image size in % of the camera image, live (0 = auto) |
| `/perception/query` | `std_msgs/String` | YOLOE only: `"chair, extinguisher"` replaces the list, `"+backpack"` adds, `"-chair"` removes |
| `/perception/target/select` | `std_msgs/String` (JSON) | pick the target: `{"box":[x1,y1,x2,y2],"stamp":s,"name":"chair"}`, `{"name":"person"}`, `{"track_id":3}` or `{"clear":true}` |
| `/florence/command` | `std_msgs/String` | Florence-2 search (`chair`, `far chair`…), then `track N` / `untrack` |
| `/yolo/command` | `std_msgs/String` | Foxglove YOLO panel: `find …`, `track N` / `track person` / `untrack`, plus `list`, `go N`, `follow N`, `stop` |

## 6. Outputs

Images, masks and the debug overlay are only computed while someone subscribes.

| Topic | Type | Description |
|---|---|---|
| `/perception/objects` | `std_msgs/String` (JSON) | everything per frame: name, class, score, 2D box, track ID, target flag, source, `mask_index`, 3D point (m), size (m) |
| `/perception/detections` | `vision_msgs/Detection2DArray` | all 2D boxes, full-resolution pixels |
| `/perception/detections_3d` | `vision_msgs/Detection3DArray` | 3D centre in the camera frame plus width and height (m), for objects with depth |
| `/perception/masks` (+ `/compressed`) | `sensor_msgs/Image` mono8 (PNG when compressed) | one mask image for all objects: pixel = `mask_index` of the object there, 0 = none. Masks come from YOLOE and SAM 2 |
| `/perception/debug_image` (+ `/compressed`) | `sensor_msgs/Image` / `CompressedImage` | full-resolution camera frame with masks, boxes, `#ID class score distance`, target in red |
| `/perception/class_names` | `std_msgs/String` (latched) | `{id: name}` of the running detector; re-published when the YOLOE words change |
| `/perception/target_3d` | `geometry_msgs/PointStamped` | the target's 3D point in the camera frame |
| `/perception/target/status` | `std_msgs/String` (latched) | result of the last target selection |
| `/perception/query/status` | `std_msgs/String` (latched) | result of the last YOLOE query |
| `/yolo_detections`, `/yolo_detections_3d`, `/yolo_class_names`, `/yolo_debug_image(/compressed)` | as upstream | the old topics, detector objects only, kept for the Foxglove bridge, `object_query_node` and the layout |
| `/florence/result`, `/florence/objects_3d`, `/inspection/annotated/compressed` | String, String (JSON), CompressedImage | Florence-2 answers, its objects with 3D point and 2D box, and its annotated image |

All 3D outputs are in the camera's optical frame: x right, y down, z forward. Track IDs only appear in the JSON and on
the debug image, because ROS 1 `vision_msgs` has no field for them.

## 7. Files here

| File | What |
|---|---|
| `detector_service.py` | GPU service (Python 3.10, CUDA): HTTP API `/detect`, `/target`, `/classes`, `/info`, `/health` |
| `detectors.py` | YOLOv5, RF-DETR, YOLOE behind one interface |
| `trackers.py` | IoU, BoT-SORT, SAM 2 behind one interface |
| `detector_node.py` | ROS 1 node (Python 3.8, no torch) |
| `depth3d.py` | box → 3D point from the lidar depth image (shared with the Florence relay) |
| `get_models.sh`, `export_rfdetr.py`, `export_reid.py` | model download / export / TensorRT build |
