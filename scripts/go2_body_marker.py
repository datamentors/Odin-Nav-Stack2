#!/usr/bin/env python3
# OUR helper (not ONS2 upstream): draws the Go2 body in Foxglove/RViz so the robot position is easy to see.
# /go2_body_marker = orange box 0.70 x 0.35 x 0.30 m (ONS2 Go2 footprint) + white arrow (heading), frame base_link.
import rospy
from visualization_msgs.msg import Marker, MarkerArray
rospy.init_node("go2_body_marker")
pub = rospy.Publisher("/go2_body_marker", MarkerArray, queue_size=1, latch=True)
def mk(i, typ, sx, sy, sz, rgba, x=0.0):
    m = Marker(); m.header.frame_id = "base_link"; m.ns = "go2"; m.id = i; m.type = typ; m.action = Marker.ADD
    m.pose.position.x = x; m.pose.position.z = 0.15; m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    m.frame_locked = True
    return m
r = rospy.Rate(2)
while not rospy.is_shutdown():
    now = rospy.Time(0)
    body = mk(0, Marker.CUBE, 0.70, 0.35, 0.30, (1.0, 0.45, 0.0, 0.9))
    arrow = mk(1, Marker.ARROW, 0.6, 0.08, 0.08, (1.0, 1.0, 1.0, 1.0), x=0.0); arrow.pose.position.z = 0.35
    for m in (body, arrow): m.header.stamp = now
    pub.publish(MarkerArray(markers=[body, arrow])); r.sleep()
