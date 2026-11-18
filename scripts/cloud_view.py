#!/usr/bin/env python3
# OUR helper (not ONS2 upstream): light point clouds for Foxglove over Wi-Fi.
#  1. /odin1/cloud_slam (~0.5 MB at 10 Hz = ~5 MB/s, too heavy for Wi-Fi)
#       -> /odin1/cloud_slam_lite : 10 cm voxels, 2 Hz, colors kept (what the Odin1 sees NOW)
#  2. Site 3D map from a Pocket2 scan (.pcd, env ONS2_MAP_PCD)
#       -> /map_cloud : 12 cm voxels, ceiling removed, frame "map", latched (sent once per viewer)
import os, sys, time
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField

PCD = os.environ.get("ONS2_MAP_PCD", os.path.expanduser("~/ons2/ros_ws/src/map_planner/maps/map.pcd"))   # Pocket2 cloud (ONS2 guide: .pcd in maps/)
FLOOR_Z, CEIL_CUT = -1.42, 1.0          # Pocket2 cloud: floor at z=-1.42, ceiling ~+1.42 -> keep z < +1.0
LIVE_VOXEL, MAP_VOXEL, LIVE_HZ = 0.10, 0.12, 2.0


def voxel(xyz, size, extra=None):
    keys = np.floor(xyz / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], (extra[idx] if extra is not None else None)


def make_cloud(frame, stamp, xyz, rgb=None):
    fields = [PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1),
              PointField("z", 8, PointField.FLOAT32, 1)]
    cols = [xyz.astype(np.float32)]
    if rgb is not None:
        fields.append(PointField("rgb", 12, PointField.FLOAT32, 1))
        cols.append(rgb.astype(np.float32).reshape(-1, 1))
    data = np.hstack([c.view(np.float32) for c in cols]).astype(np.float32)
    m = PointCloud2()
    m.header.frame_id, m.header.stamp = frame, stamp
    m.height, m.width = 1, len(xyz)
    m.fields, m.is_bigendian, m.is_dense = fields, False, True
    m.point_step = 4 * data.shape[1]
    m.row_step = m.point_step * m.width
    m.data = data.tobytes()
    return m


def main():
    rospy.init_node("cloud_view")
    pub_live = rospy.Publisher("/odin1/cloud_slam_lite", PointCloud2, queue_size=1)
    pub_map = rospy.Publisher("/map_cloud", PointCloud2, queue_size=1, latch=True)

    # site map (once, latched)
    try:
        sys.path.insert(0, os.path.expanduser("~/ons2/scripts/map_processing"))
        import pcd_to_pgm
        r = pcd_to_pgm.read_pcd(PCD)
        pts = np.asarray(r[0] if isinstance(r, tuple) else r, dtype=np.float64)[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] < CEIL_CUT) & (pts[:, 2] > FLOOR_Z - 0.3)]
        pts, _ = voxel(pts, MAP_VOXEL)
        pub_map.publish(make_cloud("map", rospy.Time.now(), pts))
        rospy.loginfo("/map_cloud: %d points (Pocket2 map, %.2f m voxels, ceiling removed)", len(pts), MAP_VOXEL)
    except Exception as e:
        rospy.logwarn("map cloud not published: %s", e)

    last = [0.0]

    def cb(m):
        if time.time() - last[0] < 1.0 / LIVE_HZ:
            return
        last[0] = time.time()
        n = m.width * m.height
        if n == 0 or m.point_step < 16:
            return
        a = np.frombuffer(m.data, dtype=np.float32).reshape(n, m.point_step // 4)
        xyz, rgb = a[:, :3], a[:, 3]
        ok = np.isfinite(xyz).all(axis=1)
        xyz, rgb = voxel(xyz[ok].astype(np.float64), LIVE_VOXEL, rgb[ok])
        pub_live.publish(make_cloud(m.header.frame_id, m.header.stamp, xyz, rgb))

    rospy.Subscriber("/odin1/cloud_slam", PointCloud2, cb, queue_size=1, buff_size=2 ** 24)

    # The 2D map (/map) and /map_cloud are sent ONCE (latched); if Foxglove drops that single message (busy link,
    # layout change) the map never shows. Re-send them periodically: /map_view every 3 s, /map_cloud every 15 s.
    from nav_msgs.msg import OccupancyGrid
    pub_mapview = rospy.Publisher("/map_view", OccupancyGrid, queue_size=1, latch=True)
    grid = {}
    def on_map(m):
        # draw the 2D map at FLOOR height: in this map frame the floor is at z=-1.42 (Pocket2 scan origin = scanner
        # height), so at z=0 the opaque grid hid the robot (base_link is at about z=-1.19) when seen from above
        m.info.origin.position.z = FLOOR_Z
        grid["msg"] = m
    rospy.Subscriber("/map", OccupancyGrid, on_map, queue_size=1)
    map_msg = [None]
    try:
        map_msg[0] = make_cloud("map", rospy.Time.now(), pts)
    except Exception:
        pass
    k = 0
    while not rospy.is_shutdown():
        if "msg" in grid:
            pub_mapview.publish(grid["msg"])
        if map_msg[0] is not None and k % 5 == 0:
            pub_map.publish(map_msg[0])
        k += 1
        rospy.sleep(3.0)


if __name__ == "__main__":
    main()
