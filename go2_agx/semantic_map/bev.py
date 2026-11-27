#!/usr/bin/env python3
"""Bird's-eye-view debug image of the semantic map: objects (and the robot) drawn on the occupancy grid, or on a plain
1 m grid when there is no map. numpy + OpenCV only (Python 3.8), no ROS: semantic_map_node.py calls render().

render(objects, grid=None, robot=None, window_m=0, px_per_m=50, max_px=1000) -> BGR image (longest side <= max_px)
  objects  [{"id", "label", "position": [x, y, z], "size": [w, h] | None, "observations", "last_seen"}] (map frame)
  grid     None, or {"data": int8 H x W (-1 unknown, 0..100), "resolution": m/cell, "origin": (x, y)} (OccupancyGrid)
  robot    None, or (x, y, yaw) in the map frame
  window_m 0 = the whole map (or everything drawn, without a map); > 0 = a square of that size around the robot
"""
import zlib

import cv2
import numpy as np

UNKNOWN, FREE, OCCUPIED = (205, 205, 205), (250, 250, 250), (60, 60, 60)


def label_color(label):
    """Stable, saturated BGR colour per label."""
    h = zlib.crc32(label.encode()) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _bounds(objects, robot, grid, window_m):
    """World rectangle (x0, y0, x1, y1) to draw."""
    if window_m > 0 and robot is not None:
        return robot[0] - window_m / 2, robot[1] - window_m / 2, robot[0] + window_m / 2, robot[1] + window_m / 2
    if grid is not None:
        h, w = grid["data"].shape
        ox, oy = grid["origin"]
        return ox, oy, ox + w * grid["resolution"], oy + h * grid["resolution"]
    pts = [o["position"][:2] for o in objects] + ([robot[:2]] if robot is not None else [])
    if not pts:
        return -5.0, -5.0, 5.0, 5.0
    pts = np.array(pts, np.float64)
    (x0, y0), (x1, y1) = pts.min(0) - 2.0, pts.max(0) + 2.0
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0, 10.0) / 2   # at least 10 m, square
    return cx - half, cy - half, cx + half, cy + half


def render(objects, grid=None, robot=None, window_m=0.0, px_per_m=50.0, max_px=1000):
    x0, y0, x1, y1 = _bounds(objects, robot, grid, window_m)
    s = min(px_per_m, max_px / max(x1 - x0, y1 - y0))           # pixels per metre
    W, H = max(int((x1 - x0) * s), 1), max(int((y1 - y0) * s), 1)
    to_px = lambda x, y: (int(round((x - x0) * s)), int(round((y1 - y) * s)))   # image y points down, map y up

    if grid is not None:   # occupancy grid -> colours, then the visible part resampled to the image
        g = grid["data"]
        rgb = np.empty(g.shape + (3,), np.uint8)
        rgb[:] = UNKNOWN
        rgb[(g >= 0) & (g < 50)] = FREE
        rgb[g >= 50] = OCCUPIED
        rgb = rgb[::-1]   # row 0 = highest y
        r, (ox, oy), gh = grid["resolution"], grid["origin"], g.shape[0]
        # affine: image px -> grid cell (flipped rows), nearest neighbour keeps walls crisp
        m = np.float32([[1 / (s * r), 0, (x0 - ox) / r], [0, 1 / (s * r), gh - (y1 - oy) / r]])
        img = cv2.warpAffine(rgb, m, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=UNKNOWN)
    else:                  # no map: white with a 1 m grid (5 m lines darker)
        img = np.full((H, W, 3), 255, np.uint8)
        for k in range(int(np.floor(x0)), int(np.ceil(x1)) + 1):
            cv2.line(img, to_px(k, y0), to_px(k, y1), (200, 200, 200) if k % 5 else (150, 150, 150), 1)
        for k in range(int(np.floor(y0)), int(np.ceil(y1)) + 1):
            cv2.line(img, to_px(x0, k), to_px(x1, k), (200, 200, 200) if k % 5 else (150, 150, 150), 1)
        cv2.circle(img, to_px(0, 0), 4, (0, 0, 0), -1)   # map origin

    newest = max([o["last_seen"] for o in objects], default=0.0)
    for o in sorted(objects, key=lambda o: o["last_seen"]):      # newest drawn last (on top)
        c = label_color(o["label"])
        if newest - o["last_seen"] > 30:                           # not seen for a while: faded
            c = tuple(int(0.4 * v + 0.6 * 255) for v in c)
        p = to_px(*o["position"][:2])
        rad = max(int(0.5 * (o["size"][0] if o.get("size") else 0.3) * s), 4)
        cv2.circle(img, p, rad, c, -1)
        cv2.circle(img, p, rad, (0, 0, 0), 1)
        txt = "#%d %s (%d)" % (o["id"], o["label"], o["observations"])
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        tx = p[0] + rad + 2 if p[0] + rad + 2 + tw < W else p[0] - rad - 2 - tw   # keep the label inside the image
        cv2.putText(img, txt, (tx, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    if robot is not None:  # robot: dot + heading arrow + 60 deg camera wedge (1.5 m)
        x, y, yaw = robot
        p = to_px(x, y)
        for a in (yaw - 0.52, yaw + 0.52):
            cv2.line(img, p, to_px(x + 1.5 * np.cos(a), y + 1.5 * np.sin(a)), (0, 140, 255), 1, cv2.LINE_AA)
        cv2.arrowedLine(img, p, to_px(x + 0.6 * np.cos(yaw), y + 0.6 * np.sin(yaw)), (0, 0, 255), 2, cv2.LINE_AA,
                        tipLength=0.3)
        cv2.circle(img, p, 5, (0, 0, 255), -1)
    cv2.putText(img, "%d objects  %.0f px/m%s" % (len(objects), s, "" if grid is not None else "  (no map: 1 m grid)"),
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img
