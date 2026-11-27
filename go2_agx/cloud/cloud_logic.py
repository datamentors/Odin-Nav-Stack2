"""Pure helpers of cloud_publisher (no ROS import, so they can be tested anywhere)."""
import json
import math


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yolo_signature(objects):
    """What counts as 'the list changed': classes and distances rounded to 0.5 m."""
    return tuple(sorted((o["class"], round(o.get("distance_m", 0.0) * 2) / 2) for o in objects))


def parse_class_names(text):
    """/yolo_class_names is JSON: {"0": "person", ...} or {"person": 0, ...}. Returns {id: name}."""
    names = {}
    for k, v in json.loads(text).items():
        if isinstance(v, str):
            names[int(k)] = v
        else:
            names[int(v)] = str(k)
    return names


def apply_tf(t, x, y, z):
    """Point (x, y, z) -> map x, y: rotate by the quaternion, add the translation. t has .rotation, .translation."""
    q = t.rotation
    r00 = 1 - 2 * (q.y * q.y + q.z * q.z)
    r01 = 2 * (q.x * q.y - q.z * q.w)
    r02 = 2 * (q.x * q.z + q.y * q.w)
    r10 = 2 * (q.x * q.y + q.z * q.w)
    r11 = 1 - 2 * (q.x * q.x + q.z * q.z)
    r12 = 2 * (q.y * q.z - q.x * q.w)
    return (round(r00 * x + r01 * y + r02 * z + t.translation.x, 2),
            round(r10 * x + r11 * y + r12 * z + t.translation.y, 2))
