#!/usr/bin/env python3
"""Tests of the pure helpers (no ROS needed)."""
import math
import pathlib
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from cloud_logic import apply_tf, envelope, iso8601, parse_class_names, yaw_of, yolo_signature  # noqa: E402


def quat_z(yaw):
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def tf(yaw, tx, ty):
    return NS(rotation=quat_z(yaw), translation=NS(x=tx, y=ty, z=0.0))


def close(a, b):
    return abs(a - b) < 0.011


def main():
    checks = []
    # no rotation: only the translation
    checks.append(("identity", apply_tf(tf(0, 1, 2), 3, 4, 5) == (4.0, 6.0)))
    # 90 degrees about z: (1, 0) -> (0, 1)
    x, y = apply_tf(tf(math.pi / 2, 0, 0), 1.0, 0.0, 0.0)
    checks.append(("yaw 90", close(x, 0.0) and close(y, 1.0)))
    # 90 degrees about z, then translation
    x, y = apply_tf(tf(math.pi / 2, 10, 20), 0.0, 2.0, 0.0)
    checks.append(("yaw 90 + shift", close(x, 8.0) and close(y, 20.0)))
    # a tilted camera (roll 90 about x): z of the point lands on -y of the map
    q = NS(x=math.sin(math.pi / 4), y=0.0, z=0.0, w=math.cos(math.pi / 4))
    x, y = apply_tf(NS(rotation=q, translation=NS(x=0.0, y=0.0, z=0.0)), 0.0, 0.0, 1.0)
    checks.append(("roll 90 (z -> -y)", close(x, 0.0) and close(y, -1.0)))
    checks.append(("yaw_of", close(yaw_of(quat_z(1.2)), 1.2)))
    checks.append(("names id->name", parse_class_names('{"0": "person", "56": "chair"}') == {0: "person", 56: "chair"}))
    checks.append(("names name->id", parse_class_names('{"person": 0, "chair": 56}') == {0: "person", 56: "chair"}))
    a = [{"class": "chair", "distance_m": 2.4}, {"class": "person", "distance_m": 3.9}]
    b = [{"class": "person", "distance_m": 3.95}, {"class": "chair", "distance_m": 2.42}]
    c = [{"class": "chair", "distance_m": 3.4}]
    checks.append(("signature: same list, small move", yolo_signature(a) == yolo_signature(b)))
    checks.append(("signature: different list", yolo_signature(a) != yolo_signature(c)))
    checks.append(("iso8601", iso8601(1795789200.25) == "2026-11-27T14:20:00.250Z"))
    ident = {"robot_id": "r1", "env": "dev", "fleet": "f1"}
    e = envelope(ident, {"kind": "odin.x.v1"}, 1795789200.25, "id-1")
    checks.append(("envelope keys", sorted(e) == sorted(["schema_version", "message_id", "timestamp", "source", "robot_id", "env", "fleet", "payload"])))
    checks.append(("envelope values", e["robot_id"] == "r1" and e["source"] == "odin_cloud_publisher" and e["payload"]["kind"] == "odin.x.v1"))
    failed = 0
    for name, ok in checks:
        print("%s  %s" % ("PASS" if ok else "FAIL", name))
        failed += 0 if ok else 1
    print("== %d checks, %d failed" % (len(checks), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
