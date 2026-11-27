#!/usr/bin/env python3
"""2D box -> 3D point in the camera frame, from the Odin1 lidar depth image (pcd2depth: the lidar projected into the
undistorted camera image, same size and intrinsics). Shared by detector_node.py and florence_cmd_relay.py (both copy
this file next to them in the container odin-agx). Python 3.8, numpy only.

Method (fixed 16 Nov after the hollow-chair bug): the box shrunk 10 % per side, the valid lidar depths inside it,
their 20th percentile = the nearest surfaces of the object, not what is seen through it (a chair's box centre shows
the wall behind). The box centre pixel is unprojected at that depth with the pinhole model of THIS Odin1.
"""
import re

import numpy as np


def load_intrinsics(path="/calib.yaml"):
    """Pinhole intrinsics from the Odin1 calib.yaml written by the driver (A11 = fx, A22 = fy, u0/v0 = centre, A12 = skew)."""
    txt = open(path).read()
    g = lambda k: float(re.search(r"^\s*%s:\s*([-+0-9.eE]+)" % k, txt, re.M).group(1))
    return {"fx": g("A11"), "fy": g("A22"), "cx": g("u0"), "cy": g("v0"), "skew": g("A12")}


def units_of(encoding):
    """Metres per pixel unit of a sensor_msgs/Image depth encoding (16UC1 / mono16 are millimetres)."""
    return 0.001 if encoding in ("16UC1", "mono16") else 1.0


def nearest(items, stamp):
    """(stamp, msg) items -> the item closest in time to stamp (None if empty; stamp <= 0 = the newest)."""
    items = list(items)
    if not items:
        return None
    return max(items, key=lambda it: it[0]) if stamp <= 0 else min(items, key=lambda it: abs(it[0] - stamp))


def box_depth(depth, box, percentile=20.0, units=1.0, min_pts=20, small_min_pts=5, small_area=4000):
    """Depth (m) of the object in box (x1, y1, x2, y2, pixels of the depth image) or None when the lidar has too few
    points on it (small/far boxes may pass with small_min_pts). depth: the raw image (units = metres per unit); only
    the box is converted."""
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    xa, xb = int(np.clip(x0 + 0.1 * bw, 0, w - 1)), int(np.clip(x1 - 0.1 * bw, 1, w))
    ya, yb = int(np.clip(y0 + 0.1 * bh, 0, h - 1)), int(np.clip(y1 - 0.1 * bh, 1, h))
    v = depth[ya:yb, xa:xb].astype(np.float32) * units
    v = v[np.isfinite(v) & (v > 0.15) & (v < 40.0)]
    if v.size < (min_pts if bw * bh > small_area else small_min_pts):
        return None
    return float(np.percentile(v, percentile))


def unproject(K, u, v, z):
    """Pixel (u, v) at depth z (along the optical axis) -> (x, y, z) in the camera frame (x right, y down, z forward)."""
    my = (v - K["cy"]) / K["fy"]
    mx = (u - K["cx"] - K["skew"] * my) / K["fx"]
    return float(mx * z), float(my * z), float(z)


def box_to_3d(K, depth, box, percentile=20.0, units=1.0):
    """Box -> (x, y, z) of its centre ray at the object's depth, or None."""
    z = box_depth(depth, box, percentile, units)
    if z is None:
        return None
    return unproject(K, (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, z)


def box_size_m(K, box, z):
    """Metric (width, height) of a box at depth z."""
    return (box[2] - box[0]) * z / K["fx"], (box[3] - box[1]) * z / K["fy"]
