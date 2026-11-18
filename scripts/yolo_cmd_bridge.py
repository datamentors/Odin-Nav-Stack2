#!/usr/bin/env python3
# OUR helper (not ONS2 upstream): Foxglove control of THEIR yolo_ros + "go to" / "follow" an object.
#   /yolo/command (std_msgs/String):
#     list | classes | chair | move to the front of the chair ...  -> typed into THEIR object_query_node (tmux "yolo")
#     go N        -> walk to object #N of the last "list" (stops STANDOFF_GO m in front of it, facing it)
#     follow N    -> follow object #N (e.g. a person) with THEIR mission_manager dynamic mode
#                    (/waypoint_manager/set_dynamic_mode + /waypoint_manager/dynamic_goal + follow_search when lost)
#     stop        -> stop EVERYTHING that moves the robot (follow, follow mode, mission, NeuPAN, turns); keeps lists
#     reset | clear -> stop + wipe what could be reused later: the LATCHED last route (/initial_path, a restarted
#                    NeuPAN would pick it up again), the "list" numbers, Florence F# objects, goal/object markers
#     go F1 1 F3 5 [wait 5] -> ONE mission with several stops, in that order (F# = Florence-2, number = last "list"),
#                    standing STANDOFF_GO m in front of each, facing it, DWELL_SEC s each (or "wait N"), via THEIR
#                    mission_manager (/waypoint_manager/start_mission; STOP cancels, pause/resume work)
#     go F1       -> walk to object F1 found by FLORENCE-2 (any English word, e.g. "fire extinguisher", which YOLO
#                    does not know). Florence objects arrive on /florence/objects_3d (AGX relay, lidar distance),
#                    are put on the map at the photo's time, shown in "list" and as markers on /object_markers,
#                    and forgotten after FLO_TTL s or with "forget".
#   /yolo/console (std_msgs/String): answers. "list" is numbered with distance and direction of each object,
#     computed from THEIR 3D detections (/yolo_detections_3d, camera frame: x right, y down, z forward).
# SAFETY (17 Nov): follow must never walk the robot backwards (nothing watches behind it):
#   - follow only uses FRESH detections (YOLO publishes /yolo_detections_3d only when it sees something, so the last
#     message stayed forever: a "ghost" person glued to the camera that moved with the robot),
#   - the person is put on the map with the TF of the PHOTO's time (no latency feedback when the robot moves),
#   - a goal is sent only if it is IN FRONT of the robot and the person is in front (+-60 deg) and farther than
#     the standoff; otherwise the robot is stopped at once (/neupan/stop) and waits, still in follow mode,
#   - black box ~/ons2/logs/follow_blackbox.log: every follow decision + every non-zero /cmd_vel with its publisher
#     (backward commands always), to know next time who moved the robot.
import json, math, os, re, subprocess, threading, time
import logging, logging.handlers
import rospy, tf2_ros
from std_msgs.msg import String, Empty
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import PoseStamped, Twist
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

SESSION_WIN = "ons2_go2:yolo"
CAMERA_FRAME = "camera_link"          # their tf_relay_node: odin1_base_link -> lidar_link -> camera_link
STANDOFF_GO, STANDOFF_FOLLOW = 0.8, 1.2
FOLLOW_HZ, LOST_SEC, ASSOC_M = 2.0, 4.0, 1.5
FOLLOW_FRESH_SEC = 0.8      # a detection older than this is NOT "seeing" the person
FOLLOW_JUMP_M = 0.8         # once locked, an estimate farther than this from the person is another object
FOLLOW_EMA = 0.6            # smoothing of the person position (weight of the new measurement)
# v3 (17 Nov, after the first tests: it moved in jerks / barely moved). NeuPAN stops 0.45 m BEFORE any goal
# (arrive_threshold), so a goal must be >= 0.6 m ahead to make it walk at all.
FOLLOW_GAP = 0.8            # goal = 0.8 m before the person -> NeuPAN stops ~1.25 m from the person
FOLLOW_START_AHEAD = 0.6    # start walking when the goal is >= 0.6 m ahead (person >= ~1.4 m)...
FOLLOW_KEEP_AHEAD = 0.2     # ...and keep updating it while >= 0.2 m ahead (NeuPAN stops by itself near it)
FOLLOW_STOP_M = 1.0         # person closer than this (robot centre) -> STOP at once
FOLLOW_TURN_DEG = 25.0      # robot standing and person more than this to a side -> turn in place to face them
FOLLOW_MAX_BEARING = 60.0   # walking and person more than this to a side -> stop, then turn
# MANUAL TAKEOVER (17 Nov: follow fought the remote, pulling the robot back to the person). NeuPAN can no longer
# reverse and is diff-drive (no sideways motion), so per 0.5 s loop: backward > 8 cm, sideways > 10 cm, or any motion
# (> 12 cm / 15 deg) while follow commands nothing -> it is the REMOTE (or someone pushing): follow stops itself.
FOLLOW_MANUAL_BACK = 0.08
FOLLOW_MANUAL_SIDE = 0.10
FOLLOW_MANUAL_IDLE_M = 0.12
FOLLOW_MANUAL_IDLE_DEG = 15.0
BLACKBOX = os.path.join(os.environ.get("GO2_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")),
                        "follow_blackbox.log")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
NOISE = ("Ultralytics Settings", "yolo settings", "docs.ultralytics.com")

names, last3d = {}, {"msg": None, "t": 0.0}   # t = arrival time (local clock)
numbered = []                          # last "list": [{n, cls, conf, mx, my, dist, side}]
follow = {"on": False, "cls": None, "x": 0.0, "y": 0.0, "seen": 0.0, "searching": False, "moving": False,
          "locked": False, "turning": False, "turned_at": 0.0, "prev": None, "cmd_at": 0.0}
DWELL_SEC = 3.0                        # default wait at each stop of a multi-stop "go"
FLO_TTL = 600.0                        # Florence objects are remembered 10 min (or until "forget"/a new search)
flo = {"objs": [], "phrase": "", "t": 0.0}   # [{id "F1", name, label, mx, my, dist0}]


def pane_lines():
    out = subprocess.run(["tmux", "capture-pane", "-t", SESSION_WIN, "-p", "-J", "-S", "-400"],
                         capture_output=True, text=True).stdout
    return [ANSI.sub("", l).rstrip() for l in out.splitlines()]


def type_into_their_node(cmd):
    before = len(pane_lines())
    subprocess.run(["tmux", "send-keys", "-t", SESSION_WIN, "-l", cmd])
    subprocess.run(["tmux", "send-keys", "-t", SESSION_WIN, "Enter"])
    time.sleep(2.5)
    lines = pane_lines()
    new = lines[max(0, before - 1):] if len(lines) >= before else lines[-40:]
    return "\n".join(l for l in new[-40:] if l.strip() and not any(n in l for n in NOISE))


def robot_xy(tfb):
    t = tfb.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.5)).transform.translation
    return t.x, t.y


def robot_pose(tfb):
    t = tfb.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.5)).transform
    q = t.rotation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return t.translation.x, t.translation.y, yaw


def to_map(tfb, frame, x, y, z, stamp=None):
    """camera point -> map (manual quaternion rotation; avoids tf2_geometry_msgs/PyKDL on Python 3.10).
    stamp: use the TF of that moment (Florence photo, ~2 s old); falls back to the latest TF."""
    t = None
    if stamp:
        try:
            t = tfb.lookup_transform("map", frame, rospy.Time.from_sec(stamp), rospy.Duration(0.5)).transform
        except Exception:
            t = None
    if t is None:
        t = tfb.lookup_transform("map", frame, rospy.Time(0), rospy.Duration(0.5)).transform
    qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
    # v' = q * v * q^-1
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    return rx + t.translation.x, ry + t.translation.y


def objects(tfb, only=None, max_age=None):
    """Objects of the last 3D frame: class, conf, map x/y, distance and direction from the camera.
    max_age: ignore the frame if it arrived more than max_age s ago (follow)."""
    m = last3d["msg"]
    out = []
    if m is None:
        return out
    if max_age is not None and time.time() - last3d.get("t", 0.0) > max_age:
        return out
    frame = m.header.frame_id or CAMERA_FRAME
    stamp = m.header.stamp.to_sec() if m.header.stamp else None
    for d in m.detections:
        if not d.results:
            continue
        h = d.results[0]
        cls = names.get(str(h.id), str(h.id))
        if only and only not in cls.lower():
            continue
        p = d.bbox.center.position
        if abs(p.x) < 1e-6 and abs(p.y) < 1e-6 and abs(p.z) < 1e-6:
            out.append({"cls": cls, "conf": h.score, "nodepth": True})
            continue
        dist = math.hypot(p.x, p.z)
        ang = math.degrees(math.atan2(p.x, p.z))
        side = "ahead" if abs(ang) < 10 else "%d deg %s" % (abs(ang), "right" if ang > 0 else "left")
        o = {"cls": cls, "conf": h.score, "dist": dist, "side": side, "nodepth": False}
        try:
            o["mx"], o["my"] = to_map(tfb, frame, p.x, p.y, p.z, stamp)   # OUR: TF of the photo's time
        except Exception:
            o["mx"] = o["my"] = None
        out.append(o)
    out.sort(key=lambda o: o.get("dist", 99))
    return out


def table(objs, numbered_list):
    if last3d["msg"] is None:
        return "(no 3D detections received yet)"
    age = time.time() - last3d.get("t", 0.0)
    if age > 2.0:
        return "(YOLO has seen nothing for %.0f s)" % age
    if not objs:
        return "(no matching object in the last frame)"
    rows, n = [], 0
    for o in objs:
        if o["nodepth"]:
            rows.append(f"  -  {o['cls']:<12} {o['conf']:4.2f}   no depth (their detector put it at 0,0,0)")
            continue
        n += 1
        o["n"] = n
        numbered_list.append(o)
        rows.append(f" {n:>2}  {o['cls']:<12} {o['conf']:4.2f}   {o['dist']:5.2f} m   {o['side']}")
    return ("Objects (last frame) - use: go N | follow N | stop\n"
            " #   Class        Conf    Dist      Direction\n" + "\n".join(rows))


def flo_alive():
    if flo["objs"] and time.time() - flo["t"] > FLO_TTL:
        flo["objs"] = []
    return flo["objs"]


def flo_table(tfb):
    objs = flo_alive()
    if not objs:
        return ""
    try:
        rx, ry, ryaw = robot_pose(tfb)
    except Exception:
        rx = ry = ryaw = None
    rows = []
    for o in objs:
        if rx is None:
            rows.append(f" {o['id']:>3}  {o['name']:<16}  (no robot pose)")
            continue
        d = math.hypot(o["mx"] - rx, o["my"] - ry)
        a = math.degrees(math.atan2(o["my"] - ry, o["mx"] - rx) - ryaw)
        a = (a + 180) % 360 - 180
        side = "ahead" if abs(a) < 10 else ("behind" if abs(a) > 150 else "%d deg %s" % (abs(a), "left" if a > 0 else "right"))
        rows.append(f" {o['id']:>3}  {o['name']:<16}  {d:5.2f} m   {side}")
    age = time.time() - flo["t"]
    return ("\n\nFlorence-2 objects ('%s', found %.0f s ago, kept %.0f min) - use: go F1\n" % (flo["phrase"], age, FLO_TTL / 60)
            + "\n".join(rows))


def goal_in_front(ox, oy, rx, ry, standoff):
    dx, dy = rx - ox, ry - oy
    L = math.hypot(dx, dy) or 1.0
    gx, gy = ox + dx / L * standoff, oy + dy / L * standoff
    yaw = math.atan2(oy - gy, ox - gx)          # face the object
    g = PoseStamped(); g.header.frame_id = "map"; g.header.stamp = rospy.Time.now()
    g.pose.position.x, g.pose.position.y = gx, gy
    g.pose.orientation.z, g.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return g, L


def reachable_goal(ox, oy, rx, ry, standoff):
    """Try goals around the object (farther / a bit to the side, always facing it) and return the first one
    THEIR global planner can reach (/map_planner/plan). Avoids GOAL_OCCUPIED when the object is near furniture."""
    from nav_interfaces.srv import PlanPath, PlanPathRequest
    plan = rospy.ServiceProxy("/map_planner/plan", PlanPath)
    base = math.atan2(ry - oy, rx - ox)                     # direction object -> robot
    last = "no candidate"
    for d in (standoff, standoff + 0.3, standoff + 0.6, standoff + 1.0, standoff + 1.5):
        for da in (0, 30, -30, 60, -60):
            a = base + math.radians(da)
            gx, gy = ox + d * math.cos(a), oy + d * math.sin(a)
            g, _ = goal_in_front(ox, oy, gx, gy, d)       # pose at (gx, gy) facing the object
            try:
                r = PlanPathRequest(); r.goal = g
                res = plan(r)
            except Exception as e:
                return None, "planner not available (%s)" % e
            if res.success:
                return g, "%.1f m from it%s" % (d, "" if da == 0 else ", %d deg to the side" % abs(da))
            last = res.message
    return None, last


def main():
    rospy.init_node("yolo_cmd_bridge")
    tfb = tf2_ros.Buffer(); tf2_ros.TransformListener(tfb)
    pub = rospy.Publisher("/yolo/console", String, queue_size=1, latch=True)
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
    dyn_pub = rospy.Publisher("/waypoint_manager/dynamic_goal", PoseStamped, queue_size=1)
    rospy.Subscriber("/yolo_class_names", String, lambda m: names.update(json.loads(m.data)))
    rospy.Subscriber("/yolo_detections_3d", Detection3DArray, lambda m: last3d.update(msg=m, t=time.time()),
                     queue_size=1)
    stop_pub = rospy.Publisher("/neupan/stop", Empty, queue_size=1)
    bb = logging.getLogger("follow_blackbox"); bb.setLevel(logging.INFO); bb.propagate = False
    try:
        h = logging.handlers.RotatingFileHandler(BLACKBOX, maxBytes=5_000_000, backupCount=3)
        h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%Y-%m-%d %H:%M:%S"))
        bb.addHandler(h)
    except Exception as e:
        rospy.logwarn("black box log not available: %s" % e)

    def blackbox(text):
        bb.info(text)

    last_cv = {}

    def on_cmd_vel(msg):
        """Black box: who sends velocity commands to the robot (vel_to_sdk only listens to /cmd_vel)."""
        v, vy, w = msg.linear.x, msg.linear.y, msg.angular.z
        if abs(v) < 0.01 and abs(vy) < 0.01 and abs(w) < 0.01:
            return
        try:
            who = msg._connection_header.get("callerid", "?")
        except Exception:
            who = "?"
        now = time.time()
        if v > -0.01 and now - last_cv.get(who, 0.0) < 0.5:     # forward/turn: max 2 lines/s per publisher
            return
        last_cv[who] = now
        try:
            rx, ry, ryaw = robot_pose(tfb)
        except Exception:
            rx = ry = ryaw = float("nan")
        blackbox("CMD_VEL %s v=%.2f vy=%.2f w=%.2f robot=(%.2f, %.2f, %.0f deg)%s"
                 % (who, v, vy, w, rx, ry, math.degrees(ryaw), "   <<< BACKWARD" if v < -0.01 else ""))

    rospy.Subscriber("/cmd_vel", Twist, on_cmd_vel, queue_size=50)
    lock = threading.Lock()
    say = lambda t: pub.publish(String(data=t))
    mk_pub = rospy.Publisher("/object_markers", MarkerArray, queue_size=1)
    flo_cmd_pub = rospy.Publisher("/florence/command", String, queue_size=1)
    goal_mk_pub = rospy.Publisher("/navigation_goal_marker", Marker, queue_size=1)

    def on_florence(m):
        try:
            d = json.loads(m.data)
            if d.get("cleared"):
                n = len(flo["objs"]); flo.update(objs=[], phrase="", t=time.time())
                say("> Florence-2 'clean': %d Florence object(s) removed from YOLO." % n)
                return
            objs = []
            for o in d.get("objects", []):
                try:
                    mx, my = to_map(tfb, d.get("frame") or CAMERA_FRAME, o["x"], o["y"], o["z"], d.get("stamp"))
                except Exception as e:
                    rospy.logwarn("florence object without map position: %s", e)
                    continue
                objs.append({"id": "F%d" % o["n"], "name": o["name"], "label": o["label"], "mx": mx, "my": my,
                             "dist0": o["dist"]})
            flo.update(objs=objs, phrase=d.get("phrase", ""), t=time.time())
            known = (d.get("phrase", "").lower() in {v.lower() for v in names.values()})
            say("> Florence-2 sent %d '%s'%s:\n" % (len(objs), d.get("phrase", ""),
                "" if known else " (YOLO does NOT know this object)")
                + "\n".join("  %s  %-16s %5.2f m  (map %.2f, %.2f)" % (o["id"], o["name"], o["dist0"], o["mx"], o["my"])
                            for o in objs) + ("\nUse: go F1 ... | list | forget" if objs else ""))
        except Exception as e:
            rospy.logwarn("florence objects: %s", e)

    def markers_loop():
        while not rospy.is_shutdown():
            time.sleep(1.0)
            arr = MarkerArray()
            for i, o in enumerate(flo_alive()):
                for kind in ("body", "text"):
                    mk = Marker()
                    mk.header.frame_id, mk.header.stamp = "map", rospy.Time.now()
                    mk.ns, mk.id = "florence_" + kind, i
                    mk.lifetime = rospy.Duration(2.5)
                    mk.pose.position.x, mk.pose.position.y = o["mx"], o["my"]
                    mk.pose.orientation.w = 1.0
                    mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.75, 0.2, 1.0, 0.85   # purple = Florence
                    if kind == "body":
                        mk.type, mk.action = Marker.CYLINDER, Marker.ADD
                        mk.pose.position.z = -1.0
                        mk.scale.x = mk.scale.y = 0.35; mk.scale.z = 0.8
                    else:
                        mk.type, mk.action = Marker.TEXT_VIEW_FACING, Marker.ADD
                        mk.pose.position.z = -0.4
                        mk.scale.z = 0.3; mk.color.r = mk.color.g = mk.color.b = 1.0
                        mk.text = "%s %s" % (o["id"], o["name"])
                    arr.markers.append(mk)
            if arr.markers:
                mk_pub.publish(arr)

    rospy.Subscriber("/florence/objects_3d", String, on_florence, queue_size=1)
    threading.Thread(target=markers_loop, daemon=True).start()
    say("Ready. /yolo/command: list | chair | go N | follow N | stop | reset | go F1 (Florence-2 objects) | "
        "go F1 1 F3 5 [wait 5] (several stops) | forget")

    def set_dynamic(on):
        try:
            rospy.wait_for_service("/waypoint_manager/set_dynamic_mode", 2.0)
            r = rospy.ServiceProxy("/waypoint_manager/set_dynamic_mode", SetBool)(on)
            return r.success, r.message
        except Exception as e:
            return False, "dynamic mode not available (%s)" % e

    def halt_everything():
        """Stop EVERYTHING that can move the robot, whatever state it is in (also after a bridge restart, when
        follow["on"] is False but THEIR mission_manager may still be in follow/DYNAMIC mode)."""
        was = follow["on"]
        follow.update(on=False, moving=False)
        stop_pub.publish(Empty())                  # NeuPAN clears its path + velocity; atomic_move aborts a turn
        ok, m = set_dynamic(False)                 # leave follow (dynamic) mode
        try:
            rospy.ServiceProxy("/waypoint_manager/cancel", Trigger)()
        except Exception:
            pass
        stop_pub.publish(Empty())
        return was

    def stop_all(reason):
        was = halt_everything()
        blackbox("STOP (%s) follow_was_on=%s" % (reason, was))
        say("> stop\n" + reason)

    def reset_all(cmd):
        halt_everything()
        # THEIR map_planner keeps the last route LATCHED on /initial_path and NeuPAN subscribes to it: a restarted or
        # reconnecting NeuPAN would load the old route again. Replace it with a null route AT the robot (goal = the
        # robot's own pose -> 0 m, no motion possible), then clear NeuPAN once more.
        route = "old route replaced"
        try:
            from nav_interfaces.srv import PlanPath, PlanPathRequest
            rx, ry, ryaw = robot_pose(tfb)
            g = PoseStamped(); g.header.frame_id = "map"; g.header.stamp = rospy.Time.now()
            g.pose.position.x, g.pose.position.y = rx, ry
            g.pose.orientation.z, g.pose.orientation.w = math.sin(ryaw / 2), math.cos(ryaw / 2)
            r = PlanPathRequest(); r.goal = g
            rospy.wait_for_service("/map_planner/plan", 2.0)
            rospy.ServiceProxy("/map_planner/plan", PlanPath)(r)
        except Exception as e:
            route = "route not replaced (%s)" % e
        rospy.sleep(0.3)
        stop_pub.publish(Empty())
        n_list, n_flo = len(numbered), len(flo["objs"])
        numbered.clear()
        flo["objs"] = []; flo["phrase"] = ""
        flo_cmd_pub.publish(String(data="clean"))   # Florence relay: forget what it sent (latched "cleared")
        rospy.sleep(1.0)                            # its answer comes back first; our RESET message must be the last
        mk = MarkerArray(); d = Marker(); d.action = Marker.DELETEALL; d.header.frame_id = "map"; mk.markers.append(d)
        mk_pub.publish(mk)
        gm = Marker(); gm.action = Marker.DELETEALL; gm.header.frame_id = "map"
        goal_mk_pub.publish(gm)
        blackbox("RESET: %s, list %d, Florence %d" % (route, n_list, n_flo))
        say("> %s\nRESET: robot stopped, follow off, mission cancelled, %s, %d list number(s) and %d Florence "
            "object(s) forgotten. Start again with 'list'." % (cmd, route, n_list, n_flo))

    def hold(reason):
        """Stop the robot NOW but stay in follow mode: NeuPAN stops on /neupan/stop and THEIR mission_manager
        waits in DYNAMIC for the next goal. Only if we had sent it walking (no stop spam)."""
        if follow["moving"]:
            stop_pub.publish(Empty())
            follow["moving"] = False
            follow["cmd_at"] = time.time()
            blackbox("FOLLOW HOLD: %s" % reason)

    def turn_async(deg):
        """Turn in place deg degrees (+ = left) with THEIR atomic_move (/atomic_move/turn, relative, 20 deg
        tolerance) in a thread; no goals are sent meanwhile. /neupan/stop (stop/reset) aborts it."""
        def run():
            try:
                from atomic_move.srv import Turn
                rospy.wait_for_service("/atomic_move/turn", 1.0)
                r = rospy.ServiceProxy("/atomic_move/turn", Turn)(float(deg))
                blackbox("FOLLOW TURN %+.0f deg -> %s %s" % (deg, r.success, r.message))
            except Exception as e:
                blackbox("FOLLOW TURN %+.0f deg failed: %s" % (deg, e))
            finally:
                follow["turning"] = False
                follow["turned_at"] = follow["cmd_at"] = time.time()
        follow["turning"] = True
        blackbox("FOLLOW TURN start %+.0f deg" % deg)
        threading.Thread(target=run, daemon=True).start()

    def follow_loop():
        period = 1.0 / FOLLOW_HZ
        while not rospy.is_shutdown() and follow["on"]:
            time.sleep(period)
            if not follow["on"]:
                break
            try:
                rx, ry, ryaw = robot_pose(tfb)
                prev, follow["prev"] = follow["prev"], (rx, ry, ryaw)
                if prev is not None:
                    ddx, ddy = rx - prev[0], ry - prev[1]
                    fwd = ddx * math.cos(prev[2]) + ddy * math.sin(prev[2])
                    side = -ddx * math.sin(prev[2]) + ddy * math.cos(prev[2])
                    dyaw = math.degrees(math.atan2(math.sin(ryaw - prev[2]), math.cos(ryaw - prev[2])))
                    quiet = (not follow["moving"] and not follow["turning"]
                             and time.time() - follow["cmd_at"] > 1.5)
                    if (fwd < -FOLLOW_MANUAL_BACK or abs(side) > FOLLOW_MANUAL_SIDE
                            or (quiet and (abs(fwd) > FOLLOW_MANUAL_IDLE_M or abs(dyaw) > FOLLOW_MANUAL_IDLE_DEG))):
                        blackbox("FOLLOW MANUAL TAKEOVER: fwd=%.2f side=%.2f dyaw=%.0f quiet=%s robot=(%.2f, %.2f)"
                                 % (fwd, side, dyaw, quiet, rx, ry))
                        stop_all("Remote control detected (the robot moved by itself / backwards / sideways): FOLLOW "
                                 "STOPPED so it does not fight you. Send 'follow N' again when you are ready.")
                        break
                cands = [o for o in objects(tfb, follow["cls"].lower(), max_age=FOLLOW_FRESH_SEC)
                         if not o["nodepth"] and o.get("mx") is not None]
                best = min(cands, key=lambda o: math.hypot(o["mx"] - follow["x"], o["my"] - follow["y"]), default=None)
                jump = math.hypot(best["mx"] - follow["x"], best["my"] - follow["y"]) if best else 99.0
                if best and jump < (FOLLOW_JUMP_M if follow["locked"] else ASSOC_M):
                    a = FOLLOW_EMA if follow["locked"] else 1.0
                    follow.update(x=(1 - a) * follow["x"] + a * best["mx"], y=(1 - a) * follow["y"] + a * best["my"],
                                  seen=time.time(), searching=False, locked=True)
                    px, py = follow["x"], follow["y"]
                    L = math.hypot(px - rx, py - ry)
                    ang = math.atan2(py - ry, px - rx) - ryaw
                    bearing = math.degrees(math.atan2(math.sin(ang), math.cos(ang)))
                    g, _ = goal_in_front(px, py, rx, ry, FOLLOW_GAP)
                    gx, gy = g.pose.position.x, g.pose.position.y
                    ahead = (gx - rx) * math.cos(ryaw) + (gy - ry) * math.sin(ryaw)
                    if follow["turning"]:
                        what = "turning to face it"
                    elif L < FOLLOW_STOP_M:
                        hold("person too close (%.2f m)" % L); what = "too close, stopped"
                    elif follow["moving"] and abs(bearing) > FOLLOW_MAX_BEARING:
                        hold("person to the side (%.0f deg)" % bearing); what = "to the side, stopped"
                    elif not follow["moving"] and abs(bearing) > FOLLOW_TURN_DEG:
                        if time.time() - follow["turned_at"] > 1.5:
                            turn_async(max(-90.0, min(90.0, bearing))); what = "turning to face it"
                        else:
                            what = "to the side, waiting"
                    elif ahead >= (FOLLOW_KEEP_AHEAD if follow["moving"] else FOLLOW_START_AHEAD):
                        dyn_pub.publish(g); follow["moving"] = True; follow["cmd_at"] = time.time(); what = "walking"
                    elif ahead < 0.0:
                        hold("goal behind the robot (%.2f m)" % ahead); what = "waiting"
                    else:
                        if follow["moving"]:
                            follow["cmd_at"] = time.time()                     # NeuPAN is still braking
                        follow["moving"] = False; what = "close enough, waiting"   # NeuPAN stops by itself near it
                    blackbox("FOLLOW %s person=(%.2f, %.2f) L=%.2f bearing=%.0f robot=(%.2f, %.2f, %.0f deg) "
                             "goal=(%.2f, %.2f) ahead=%.2f -> %s"
                             % (follow["cls"], px, py, L, bearing, rx, ry, math.degrees(ryaw), gx, gy, ahead, what))
                    say("> follow %s: %.1f m, %+.0f deg - %s" % (follow["cls"], L, bearing, what))
                else:
                    # not seen in a FRESH frame: stop at once instead of walking on to an old goal
                    if time.time() - follow["seen"] > 1.0:
                        hold("person not seen for %.1f s" % (time.time() - follow["seen"]))
                    if time.time() - follow["seen"] > LOST_SEC and not follow["searching"]:
                        follow["searching"] = True
                        blackbox("FOLLOW LOST %s" % follow["cls"])
                        say("> follow %s: target LOST (not seen for %.0f s). Robot stopped; 'stop' or 'follow N' "
                            "again." % (follow["cls"], LOST_SEC))
            except Exception as e:
                rospy.logwarn_throttle(5, "follow: %s" % e)

    def multi_go(cmd, tokens, dwell):
        """go F1 1 F3 5 [wait N] -> one mission through their mission_manager."""
        from nav_interfaces.srv import StartMission, StartMissionRequest
        from nav_interfaces.msg import Waypoint
        targets, missing = [], []
        for t in tokens:
            if re.match(r"^f\d+$", t):
                o = next((x for x in flo_alive() if x["id"].lower() == t), None)
                if o:
                    targets.append({"tag": o["id"], "name": o["name"], "mx": o["mx"], "my": o["my"]})
                else:
                    missing.append(t.upper())
            elif t.isdigit():
                o = next((x for x in numbered if x.get("n") == int(t)), None)
                if o and o.get("mx") is not None:
                    targets.append({"tag": "#" + t, "name": o["cls"], "mx": o["mx"], "my": o["my"]})
                else:
                    missing.append("#" + t)
            else:
                missing.append(t)
        if missing:
            say("> %s\nUnknown: %s. F# come from Florence-2 (search first), numbers from the last YOLO 'list'."
                % (cmd, ", ".join(missing))); return
        rx, ry = robot_xy(tfb)
        px, py = rx, ry
        wps, lines = [], []
        for i, t in enumerate(targets):
            L = math.hypot(t["mx"] - px, t["my"] - py)
            if i == 0 and L < STANDOFF_GO + 0.6:
                lines.append("  -  %s %s: only %.1f m away, skipped (already in front)" % (t["tag"], t["name"], L))
                continue
            g, how = reachable_goal(t["mx"], t["my"], px, py, STANDOFF_GO)
            if g is None:
                lines.append("  -  %s %s: no free spot around it (%s), skipped" % (t["tag"], t["name"], how))
                continue
            yaw = 2 * math.atan2(g.pose.orientation.z, g.pose.orientation.w)
            w = Waypoint()
            w.x, w.y, w.yaw, w.dwell_sec = g.pose.position.x, g.pose.position.y, yaw, dwell
            w.name = "%s %s" % (t["tag"], t["name"])
            wps.append(w)
            lines.append(" %2d  %-18s goal %s" % (len(wps), w.name, how))
            px, py = w.x, w.y
        if not wps:
            say("> %s\nNo reachable stop:\n%s" % (cmd, "\n".join(lines))); return
        if follow["on"]:
            follow["on"] = False
            set_dynamic(False)
        try:
            rospy.ServiceProxy("/waypoint_manager/cancel", Trigger)()   # replace whatever was running
            time.sleep(0.5)
            rospy.wait_for_service("/waypoint_manager/start_mission", 3.0)
            req = StartMissionRequest()
            req.waypoints, req.loop, req.temporary = wps, False, False
            res = rospy.ServiceProxy("/waypoint_manager/start_mission", StartMission)(req)
        except Exception as e:
            say("> %s\nMission manager not available: %s" % (cmd, e)); return
        if not res.success:
            say("> %s\nMission rejected: %s\n%s" % (cmd, res.message, "\n".join(lines))); return
        say("> %s\nMission started (%d stops, %.0f s at each, STOP cancels):\n%s"
            % (cmd, len(wps), dwell, "\n".join(lines)))

    def handle(msg):
        cmd = (msg.data or "").strip().replace("\n", " ")
        low = cmd.lower()
        if not cmd:
            return
        with lock:
            parts = low.split()
            if low == "stop":
                stop_all("Stopped: no follow, follow mode off, mission cancelled, NeuPAN cleared.")
                return
            if low in ("reset", "clear", "reset all", "cancel all", "clear all"):
                reset_all(cmd)
                return
            if low in ("forget", "forget florence", "clear florence"):
                n = len(flo["objs"]); flo["objs"] = []
                say("> %s\nForgot %d Florence-2 object(s)." % (cmd, n))
                return
            mm = re.match(r"^go\s+(.+?)(?:\s+wait\s+(\d+(?:\.\d+)?))?$", low)
            if mm:
                toks = mm.group(1).split()
                if len(toks) > 1 or mm.group(2):
                    multi_go(cmd, toks, float(mm.group(2)) if mm.group(2) else DWELL_SEC)
                    return
            mf = re.match(r"^(go|follow)\s+f(\d+)$", low)
            if mf:
                o = next((x for x in flo_alive() if x["id"].lower() == "f" + mf.group(2)), None)
                if o is None:
                    say("> %s\nNo Florence-2 object F%s (search it first in the Florence-2 panel, or it expired)."
                        % (cmd, mf.group(2))); return
                if mf.group(1) == "follow":
                    say("> %s\nFlorence-2 objects are saved positions (a photo): use 'go %s'. To follow something moving,"
                        " use YOLO 'follow N'." % (cmd, o["id"])); return
                rx, ry = robot_xy(tfb)
                L = math.hypot(o["mx"] - rx, o["my"] - ry)
                if L < STANDOFF_GO + 0.6:
                    say("> %s\n%s %s is only %.1f m away: already in front of it." % (cmd, o["id"], o["name"], L)); return
                g, how = reachable_goal(o["mx"], o["my"], rx, ry, STANDOFF_GO)
                if g is None:
                    say("> %s\nCannot reach %s %s (%.1f m): no free spot around it (%s)." % (cmd, o["id"], o["name"], L, how))
                    return
                goal_pub.publish(g)
                say("> %s\nGoing to %s %s (%.1f m): goal %s, facing it." % (cmd, o["id"], o["name"], L, how))
                return
            if len(parts) == 2 and parts[0] in ("go", "follow") and parts[1].isdigit():
                n = int(parts[1])
                o = next((x for x in numbered if x.get("n") == n), None)
                if o is None or o.get("mx") is None:
                    say("> %s\nNo object #%d with a map position. Send 'list' first." % (cmd, n))
                    return
                rx, ry = robot_xy(tfb)
                if parts[0] == "go":
                    g, L = goal_in_front(o["mx"], o["my"], rx, ry, STANDOFF_GO)
                    if L < STANDOFF_GO + 0.6:
                        say("> %s\n%s #%d is only %.1f m away: already in front of it." % (cmd, o["cls"], n, L))
                        return
                    g, how = reachable_goal(o["mx"], o["my"], rx, ry, STANDOFF_GO)
                    if g is None:
                        say("> %s\nCannot reach %s #%d (%.1f m, %s): no free spot around it (%s)."
                            % (cmd, o["cls"], n, L, o["side"], how))
                        return
                    goal_pub.publish(g)
                    say("> %s\nGoing to %s #%d (%.1f m, %s): goal %s, facing it."
                        % (cmd, o["cls"], n, L, o["side"], how))
                else:
                    ok, m = set_dynamic(True)
                    if not ok:
                        say("> %s\nCannot follow: %s" % (cmd, m)); return
                    follow.update(on=True, cls=o["cls"], x=o["mx"], y=o["my"], seen=time.time(), searching=False,
                                  moving=False, locked=False, turning=False, turned_at=0.0, prev=None,
                                  cmd_at=time.time())
                    blackbox("FOLLOW START %s #%d at map (%.2f, %.2f), %.2f m %s; robot=(%.2f, %.2f)"
                             % (o["cls"], n, o["mx"], o["my"], o["dist"], o["side"], rx, ry))
                    threading.Thread(target=follow_loop, daemon=True).start()
                    say("> %s\nFollowing %s #%d (%.1f m, %s): it walks when the person is > ~1.4 m away and "
                        "stops ~1.2 m from them, turns to face them, never backwards. 'stop' to end."
                        % (cmd, o["cls"], n, o["dist"], o["side"]))
                return
            their = type_into_their_node(cmd)
            text = "> " + cmd + "\n"
            if low == "list" or not low.startswith(("move", "classes", "voice", "mode", "quit", "exit")):
                numbered.clear()
                text += table(objects(tfb, None if low == "list" else low), numbered)
                if low == "list":
                    text += flo_table(tfb)
                text += "\n\n--- their yolo console ---\n"
            pub.publish(String(data=text + their))
            rospy.loginfo("yolo command: %s", cmd)

    rospy.Subscriber("/yolo/command", String, lambda m: threading.Thread(target=handle, args=(m,), daemon=True).start())
    rospy.spin()


if __name__ == "__main__":
    main()
