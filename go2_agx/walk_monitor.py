#!/usr/bin/env python3
"""WALKING TEST of one goal. The robot MOVES if the chassis bridge is running (ons2_start.sh --with-bridge): remote in hand, clear floor.
Runs inside ons2-go2 (ROS 1), after a Florence-2 search has saved the goal as F1 (flo "chair", then `list` shows F1):
    docker exec ons2-go2 bash -c "source /opt/ros/noetic/setup.bash; source ~/ons2/ros_ws/devel/setup.bash; python3 ~/ons2/go2_agx/walk_monitor.py [seconds]"
It sends `go F1`, then prints every 2 s: mission state, distance to the goal, how far the robot moved, the walking command, and what
cloud_publisher sends to the cloud side. It stops the robot (`stop` + cancel) when the walk stalls (no progress for 20 s), when the
distance grows by 0.5 m, or at the time limit (default 100 s). It ends without stopping when the mission ends by itself.
Without the chassis bridge the mission still runs (state NAVIGATING) but the legs do not move, so it ends in a stall stop."""
import json
import math
import sys
import time

import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_interfaces.msg import MissionStatus
from std_msgs.msg import String

LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
rospy.init_node("walk_monitor", anonymous=True)
buf = tf2_ros.Buffer()
tf2_ros.TransformListener(buf)
st = {"m": None, "cmd": 0.0, "cloud": None}
rospy.Subscriber("/waypoint_manager/status", MissionStatus, lambda m: st.update(m=m), queue_size=1)
rospy.Subscriber("/cmd_vel", Twist, lambda m: st.update(cmd=m.linear.x), queue_size=1)
rospy.Subscriber("/cloud_gateway/status", String, lambda m: st.update(cloud=m.data), queue_size=1)
cmd_pub = rospy.Publisher("/yolo/command", String, queue_size=1)
time.sleep(1.5)


def pose():
    try:
        t = buf.lookup_transform("map", "odin1_base_link", rospy.Time(0)).transform.translation
        return t.x, t.y
    except Exception:
        return None


def stop(reason):
    print("STOP: %s" % reason, flush=True)
    for _ in range(3):
        cmd_pub.publish(String(data="stop"))
        time.sleep(0.3)
    try:
        rospy.ServiceProxy("/waypoint_manager/cancel", __import__("std_srvs.srv", fromlist=["Trigger"]).Trigger)()
    except Exception as e:
        print("(cancel service: %s)" % e, flush=True)


p0 = pose()
print("start pose: %s" % (p0,), flush=True)
cmd_pub.publish(String(data="go F1"))
print("sent: go F1", flush=True)
t0 = time.time()
hist = []
best = 99.0
ended = None
while time.time() - t0 < LIMIT and not rospy.is_shutdown():
    time.sleep(2.0)
    m = st["m"]
    p = pose()
    cloud = None
    if st["cloud"]:
        try:
            cloud = json.loads(st["cloud"])["payload"]["nav"]
        except Exception:
            pass
    moved = math.hypot(p[0] - p0[0], p[1] - p0[1]) if (p and p0) else 0.0
    d = m.distance_to_target_m if m else -1
    print("t=%4.0fs state=%-10s dist=%5.2f m  moved=%4.2f m  cmd_vel=%5.2f  cloud: state=%s dist=%s" % (
        time.time() - t0, m.state if m else "?", d, moved, st["cmd"], cloud and cloud["state"], cloud and cloud["distance_to_target_m"]), flush=True)
    if not m:
        continue
    # ALIGNING is the turn toward the object at the end of the walk, DWELLING / PAUSED are part of a mission: not an end
    if m.state not in ("NAVIGATING", "ALIGNING", "DWELLING", "PAUSED") and time.time() - t0 > 6:
        ended = "mission ended with state %s" % m.state
        break
    hist.append((time.time() - t0, d))
    best = min(best, d)
    if m.state == "NAVIGATING" and d > best + 0.5:
        stop("distance grew by more than 0.5 m")
        ended = "stopped: distance grew"
        break
    old = [h for h in hist if h[0] <= hist[-1][0] - 20]
    if m.state == "NAVIGATING" and old and time.time() - t0 > 20 and old[-1][1] - d < 0.05:
        stop("no progress for 20 s (stall)")
        ended = "stopped: stall"
        break
if not ended:
    stop("time limit")
    ended = "stopped: time limit"
print("RESULT: %s" % ended, flush=True)
p = pose()
if p and p0:
    print("moved in total: %.2f m; final pose %s" % (math.hypot(p[0] - p0[0], p[1] - p0[1]), p), flush=True)
