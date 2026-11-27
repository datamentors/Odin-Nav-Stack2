#!/usr/bin/env python3
"""Runs CloudPublisher with fake ROS modules and checks that every message it publishes
matches its schema (needs jsonschema + pyyaml; no ROS needed)."""
import json
import pathlib
import sys
import time
import types
from types import SimpleNamespace as NS

from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---- fake ROS
published = []


class FakeTime:
    def __init__(self, s=0.0):
        self.s = s

    def to_sec(self):
        return self.s

    @staticmethod
    def now():
        return FakeTime(time.time())


class FakePub:
    def __init__(self, topic, *a, **k):
        self.topic = topic

    def publish(self, msg):
        published.append((self.topic, json.loads(msg.data)))


TF = {}
PARAMS = {}


class FakeBuffer:
    def lookup_transform(self, target, source, t, timeout=None):
        if (target, source) not in TF:
            raise LookupError("no tf")
        return TF[(target, source)]


def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


mod("rospy", Publisher=FakePub, Subscriber=lambda *a, **k: None, Time=FakeTime, Duration=lambda s: s,
    Rate=lambda hz: NS(sleep=lambda: None), is_shutdown=lambda: False, logwarn=print, logwarn_throttle=lambda *a: None,
    init_node=lambda n: None, get_param=lambda n, d=None: PARAMS.get(n, d))
mod("tf2_ros", Buffer=FakeBuffer, TransformListener=lambda b: None)
mod("nav_msgs"), mod("nav_msgs.msg", OccupancyGrid=object, Odometry=object)
mod("std_msgs"), mod("std_msgs.msg", String=lambda data="": NS(data=data))
mod("vision_msgs"), mod("vision_msgs.msg", Detection3DArray=object)
mod("nav_interfaces"), mod("nav_interfaces.msg", MissionStatus=object)

import yaml  # noqa: E402
import cloud_publisher  # noqa: E402


def q(yaw):
    import math
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def stamped_tf(yaw, x, y):
    return NS(header=NS(stamp=FakeTime(12345.0)), transform=NS(rotation=q(yaw), translation=NS(x=x, y=y, z=0.0)))


def mission_status(**kw):
    """A fake MissionStatus with exactly the fields of nav_interfaces/msg/MissionStatus.msg."""
    text = (ROOT.parent.parent / "ros_ws/src/nav_interfaces/msg/MissionStatus.msg").read_text()
    fields = [l.split()[1] for l in text.splitlines() if l.strip() and not l.startswith("#")]
    unknown = set(kw) - set(fields)
    assert not unknown, "not in MissionStatus.msg: %s" % unknown
    base = {f: 0 for f in fields}
    base.update(kw)
    return NS(**base)


def det(cid, score, x, z):
    return NS(results=[NS(id=cid, score=score)], bbox=NS(center=NS(position=NS(x=x, y=0.0, z=z))),
              header=NS(frame_id="camera_link"))


def main():
    cfg = yaml.safe_load((ROOT / "config" / "cloud_publisher.yaml").read_text())
    PARAMS[cfg["map_name_param"]] = "restaurant_walls"
    n = cloud_publisher.CloudPublisher(cfg)
    n.t0 -= 10  # past the Florence start filter

    # pose: no TF yet, then TF present
    n.pose()
    TF[("map", "odom_base")] = None
    TF[("map", cfg["base_frame"])] = stamped_tf(1.0, 3.0, -1.0)
    TF[("map", cfg["camera_frame"])] = stamped_tf(0.0, 3.0, -1.0)
    n.on_odom(NS(twist=NS(twist=NS(linear=NS(x=0.2, y=0.0)))))
    n.pose()
    # map with -1 cells
    n.on_map(NS(data=(-1, 0, 100, 0) * 4, info=NS(resolution=0.05, width=4, height=4,
                                                  origin=NS(position=NS(x=-1.0, y=-2.0), orientation=q(0.0)))))
    n.on_map(NS(data=(-1, 0, 100, 0) * 4, info=NS(resolution=0.05, width=4, height=4,
                                                  origin=NS(position=NS(x=-1.0, y=-2.0), orientation=q(0.0)))))
    # nav
    n.on_nav(mission_status(state="NAVIGATING", mission_id="m1", current_waypoint_idx=0, total_waypoints=1,
                            current_target=NS(position=NS(x=5.1, y=0.4)), distance_to_target_m=1.9,
                            progress_percent=38.0, linear_vel=0.1, angular_vel=0.0))
    n.on_nav(mission_status(state="IDLE", mission_id="", current_target=NS(position=NS(x=0.0, y=0.0)),
                            distance_to_target_m=-1.0, progress_percent=140.0))
    # yolo: names, one frame, same frame again (no new message), a changed list, then silence
    n.on_names(NS(data=json.dumps({"0": "person", "56": "chair"})))
    frame = NS(header=NS(stamp=FakeTime(99.5)), detections=[det(56, 0.87, 0.5, 2.4), det(0, 0.7, -1.0, 3.0)])
    n.on_yolo(frame)
    n.last["yolo"] = time.time() - 1
    n.on_yolo(frame)
    n.last["yolo"] = time.time() - 11  # an unchanged list is sent again after yolo_refresh_s
    n.on_yolo(frame)
    n.last["yolo"] = time.time() - 1
    n.on_yolo(NS(header=NS(stamp=FakeTime(99.6)), detections=[det(56, 0.9, 0.5, 1.0)]))
    n.last_seen["yolo"] -= 5
    # florence: a normal result, and one with missing optional fields
    n.on_florence(NS(data=json.dumps({"phrase": "chair", "seq": 12, "stamp": 1.0, "frame": "camera_link", "objects": [
        {"n": 1, "name": "chair 1", "label": "chair", "x": 0, "y": 0, "z": 2.0, "dist": 2.31, "direction": "ahead"}]})))
    n.on_florence(NS(data=json.dumps({"phrase": "table", "seq": None, "objects": [{"n": 1}]})))
    n.on_florence(NS(data="not json"))
    n.stack()

    # the spin loop's "list went empty" step, run once
    n.yolo_sig = ((("chair", 1.0)),)
    n.last_seen["yolo"] = time.time() - 5
    cloud_publisher.rospy.is_shutdown = iter([False, True]).__next__
    n.due = lambda k, p: False
    n.spin()

    schemas = {p.name.replace(".schema.json", ""): json.loads(p.read_text()) for p in (ROOT / "schemas").glob("*.json")}
    failed = 0
    counts = {}
    for topic, msg in published:
        kind = msg["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        errs = list(Draft202012Validator(schemas[kind]).iter_errors(msg))
        want = {"odin.pose.v1": "/cloud/telemetry", "odin.detections.yolo.v1": "/cloud/telemetry",
                "odin.map.v1": "/cloud/status", "odin.nav.v1": "/cloud/status", "odin.stack.v1": "/cloud/status",
                "odin.detections.florence.v1": "/cloud/events"}[kind]
        if errs or topic != want or len(json.dumps(msg)) > 4096:
            failed += 1
            print("FAIL  %s on %s: %s" % (kind, topic, [e.message for e in errs] or "wrong topic or too big"))
    print("messages:", counts)

    def count(kind):
        return counts.get(kind, 0)

    checks = [
        ("pose sent twice, first without TF", count("odin.pose.v1") == 2 and "x" not in published[0][1]),
        ("pose with TF is relocalized", published[1][1].get("relocalized") is True),
        ("map sent once for two identical maps", count("odin.map.v1") == 1),
        ("nav sent for each state change", count("odin.nav.v1") == 2),
        ("yolo: 1st, refresh, changed list, then empty", count("odin.detections.yolo.v1") == 4),
        ("yolo objects carry class names", any(o["class"] == "chair" for t, m in published
                                               if m["kind"] == "odin.detections.yolo.v1" for o in m["objects"])),
        ("yolo map position from TF", any(o.get("map_x") == 3.5 for t, m in published
                                          if m["kind"] == "odin.detections.yolo.v1" for o in m["objects"])),
        ("florence: two valid searches, bad json ignored", count("odin.detections.florence.v1") == 2),
        ("stack sent", count("odin.stack.v1") == 1),
        ("map name from the ROS parameter", all(m.get("map") == "restaurant_walls" for t, m in published if m["kind"] in ("odin.pose.v1", "odin.map.v1"))),
        ("last yolo message is the empty list", [m for t, m in published if m["kind"] == "odin.detections.yolo.v1"][-1]["objects"] == []),
    ]
    for name, ok in checks:
        print("%s  %s" % ("PASS" if ok else "FAIL", name))
        failed += 0 if ok else 1
    print("== %d messages, %d failed" % (len(published), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
