#!/usr/bin/env python3
# AGX ORIN, ROS 1, process relay in the container odin-agx. OUR helper (not upstream): Foxglove <-> Florence-2 service.
#   /florence/command (std_msgs/String): "chair" (numbered: chair 1, chair 2 ...), "describe 2", "show",
#                                        "check chair", "all" (full checklist) ...
#   /florence/result  (std_msgs/String): the answers; the annotated image is on /inspection/annotated/compressed
#   /florence/objects_3d (std_msgs/String, JSON, latched): after every numbered search, each object with its
#       distance from the LIDAR (depth image /odin1/depth_img_competetion at the photo's time, 20th percentile inside
#       the box = same method as our YOLO detector) and its 3D point in the camera frame (pinhole, this Odin1's
#       calib.yaml). yolo_cmd_bridge.py on the Go2 puts them on the map as F1, F2 ... -> "list", "go F1".
#   Tracking (typed in the same panel, handled here): "track 2" makes object #2 of the last search the target of the GPU
#     detector's tracker (/perception/target/select: its box + the photo's time stamp, so the tracker finds it in the
#     frame of that moment); "untrack" clears it. Needs a tracker (TRACKER in detector.env); any class works with sam2.
#   Sending to YOLO (typed in the same Florence-2 panel, handled here, not by Florence-2):
#     send on    automatic (default): every search is sent to YOLO as F1, F2 ...
#     send off   manual: searches are NOT sent; then "send 2" (only chair 2), "send 2 3", "send all"
#     clean      removes all F objects from YOLO and the purple markers
import collections, json, math, threading, urllib.parse, urllib.request
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

import depth3d   # copied next to this file (/depth3d.py) by start_perception_agx.sh

SERVICE = "http://127.0.0.1:8740"
NOT_A_SEARCH = ("describe", "desc", "show", "list", "last", "check", "help", "all", "track", "untrack")

rospy.init_node("florence_cmd_relay")
pub = rospy.Publisher("/florence/result", String, queue_size=1, latch=True)
obj_pub = rospy.Publisher("/florence/objects_3d", String, queue_size=1, latch=True)
sel_pub = rospy.Publisher("/perception/target/select", String, queue_size=1)
pub.publish(String(data="Ready. Send an English word to /florence/command (chair, table, person, fire extinguisher ...): "
                        "every object found gets a name and number (chair 1, chair 2 ...) on /inspection/annotated/compressed "
                        "and is sent to YOLO as F1, F2 ... ('list', 'go F1'); 'send off' = choose what to send, 'clean' = remove. "
                        "Then: far <word> (looks farther, ~12 s) | describe N | show | check <word> | all | track N (detector tracker) | untrack."))
busy = threading.Lock()
bridge = CvBridge()
state = {"auto": True, "last": None, "sent": {}, "boxes": {}, "stamp": 0.0}
# last = 3D result of the last search (sent or not); boxes = every object of it {n: (name, box)}, stamp = the photo's time
depth_buf = collections.deque(maxlen=25)          # (stamp_sec, msg): ~4 s of lidar depth images
yolo_names = set()

try:
    K = depth3d.load_intrinsics("/calib.yaml")
except Exception as e:
    K = None
    rospy.logwarn("florence relay: /calib.yaml not read (%s): no 3D positions" % e)

rospy.Subscriber("/odin1/depth_img_competetion", Image,
                 lambda m: depth_buf.append((m.header.stamp.to_sec(), m)), queue_size=2, buff_size=2 ** 25)
rospy.Subscriber("/yolo_class_names", String,
                 lambda m: yolo_names.update(v.lower() for v in json.loads(m.data).values()))


def depth_at(stamp, snap=()):
    """Lidar depth image closest in time to the photo (stamp in s; 0 = latest) -> (raw image, metres per unit, its stamp).
    snap = the depth images held when the command arrived (the photo is taken right then; a far search lasts ~12 s,
    longer than depth_buf keeps)."""
    it = depth3d.nearest({id(m): (st, m) for st, m in list(snap) + list(depth_buf)}.values(), stamp)
    if it is None:
        return None, 1.0, None
    st, m = it
    return bridge.imgmsg_to_cv2(m, desired_encoding="passthrough"), depth3d.units_of(m.encoding), st


def box_of(o):
    return [o["x0"], o["y0"], o["x1"], o["y1"]]


def publish_3d(snap=()):
    """After a numbered search: add the lidar distance to each object and publish /florence/objects_3d."""
    if K is None:
        return "\n(3D: no calibration, objects not saved on the map)"
    last = json.loads(urllib.request.urlopen(SERVICE + "/last", timeout=5).read())
    stamp = (last.get("stamp_ns") or 0) / 1e9
    depth, units, dst = depth_at(stamp, snap)
    if depth is None:
        return "\n(3D: no lidar depth image yet: start_depth_agx.sh)"
    if depth.shape[1] != last.get("width") or depth.shape[0] != last.get("height"):
        return "\n(3D: depth %dx%d does not match the photo %sx%s)" % (depth.shape[1], depth.shape[0],
                                                                      last.get("width"), last.get("height"))
    out, lines = [], []
    state["boxes"] = {o["n"]: (o["name"], box_of(o)) for o in last.get("objects", [])}
    state["stamp"] = stamp
    for o in last.get("objects", []):
        p = depth3d.box_to_3d(K, depth, box_of(o), units=units)   # same method as the GPU detector
        if p is None:
            lines.append("  #%d %s: no lidar points on it (too small/far): not saved" % (o["n"], o["name"]))
            continue
        x, y, z = p
        dist = math.hypot(x, z)
        out.append({"n": o["n"], "name": o["name"], "label": o["label"], "x": x, "y": y, "z": z, "dist": dist,
                    "direction": o.get("direction", ""), "box": box_of(o)})
        lines.append("  #%d %-16s %5.2f m  %s" % (o["n"], o["name"], dist, o.get("direction", "")))
    msg = {"phrase": last.get("phrase"), "seq": last.get("seq"), "stamp": stamp if stamp > 0 else dst,
           "frame": "camera_link", "objects": out}
    state["last"], state["sent"] = msg, {}
    if state["auto"]:
        state["sent"] = {o["n"]: o for o in out}
        obj_pub.publish(String(data=json.dumps(msg)))
    label = (last.get("phrase") or "").strip().lower()
    known = label in yolo_names
    head = "\nDistances (lidar, %.1f s from the photo):" % abs(dst - stamp) if stamp > 0 else "\nDistances (lidar):"
    if not out:
        tail = ""
    elif state["auto"]:
        tail = "\nSent to YOLO as %s ('list', 'go F1'). Not wanted? 'clean'." % ", ".join("F%d" % o["n"] for o in out)
    else:
        tail = "\nNOT sent to YOLO (send off). Choose: 'send 2' (only #2), 'send 2 3', 'send all'."
    if out and not known:
        tail += "\nYOLO does NOT know '%s': only Florence-2 can find it." % label
    elif out:
        tail += "\nYOLO also knows '%s' (its own numbers are in 'list' too)." % label
    return head + "\n" + "\n".join(lines) + tail


def send_cmd(q):
    """send on | send off | send all | send N [M ...] | clean  -> what reaches YOLO (/florence/objects_3d)."""
    low = q.strip().lower()
    parts = low.split()
    if low == "clean":
        state["sent"] = {}
        obj_pub.publish(String(data=json.dumps({"cleared": True, "objects": []})))
        return "Cleaned: no Florence-2 objects in YOLO any more (purple markers gone in ~3 s)."
    if low == "send on":
        state["auto"] = True
        return "Automatic: every new search is sent to YOLO as F1, F2 ... ('send off' = manual)."
    if low == "send off":
        state["auto"] = False
        return ("Manual: new searches are NOT sent to YOLO. After a search: 'send 2', 'send 2 3' or 'send all'. "
                "('send on' = automatic again)")
    last = state["last"]
    if not last or not last.get("objects"):
        return "Nothing to send: search first (e.g. 'far chair')."
    objs = {o["n"]: o for o in last["objects"]}
    if parts[1:] == ["all"]:
        chosen = list(objs)
    else:
        try:
            chosen = [int(x.lstrip("#f")) for x in parts[1:]]
        except ValueError:
            return "Use: send all | send 2 | send 2 3"
        missing = [n for n in chosen if n not in objs]
        if missing:
            return "No %s in the last search '%s' (it has %s)." % (", ".join("#%d" % n for n in missing), last["phrase"],
                                                                   ", ".join("#%d" % n for n in sorted(objs)) or "nothing")
    state["sent"].update({n: objs[n] for n in chosen})
    msg = dict(last, objects=[state["sent"][n] for n in sorted(state["sent"])])
    obj_pub.publish(String(data=json.dumps(msg)))
    return "Sent to YOLO: %s (from '%s'). In the YOLO panel: 'list', 'go F%d'." % (
        ", ".join("F%d %s" % (n, objs[n]["name"]) for n in chosen), last["phrase"], chosen[0])


def track_cmd(q):
    """track N | untrack -> /perception/target/select for the GPU detector's tracker (answer on /perception/target/status)."""
    parts = q.strip().lower().split()
    if parts[0] == "untrack":
        sel_pub.publish(String(data=json.dumps({"clear": True})))
        return "Tracking target cleared."
    try:
        n = int(parts[1].lstrip("#f"))
    except (IndexError, ValueError):
        return "Use: track N (an object number of the last search) | untrack"
    if n not in state["boxes"]:
        return "No #%d in the last search (search first, e.g. 'person')." % n
    name, box = state["boxes"][n]
    sel_pub.publish(String(data=json.dumps({"box": box, "stamp": state["stamp"], "name": (state["last"] or {}).get("phrase") or name})))
    return "Asked the detector to track #%d %s (answer on /perception/target/status and /yolo/console)." % (n, name)


def work(q):
    low = q.strip().lower()
    if low.split()[:1] in (["track"], ["untrack"]):
        pub.publish(String(data="> %s\n%s" % (q, track_cmd(q))))
        return
    if low == "clean" or low.split()[:1] == ["send"]:
        pub.publish(String(data="> %s\n%s" % (q, send_cmd(q))))
        return
    if not busy.acquire(blocking=False):
        pub.publish(String(data="Busy with the previous request, try again in a moment."))
        return
    try:
        snap = list(depth_buf)          # keep the depth images of the moment the photo is taken
        pub.publish(String(data="> %s\nworking ..." % q))
        url = SERVICE + "/run?" + urllib.parse.urlencode({"q": q})
        try:
            text = urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
        except Exception as e:
            text = "Florence-2 service not reachable (%s). On the AGX: ~/odin_test/start_perception_agx.sh" % e
            pub.publish(String(data="> %s\n%s" % (q, text)))
            return
        first = q.strip().lower().split()[0] if q.strip() else ""
        if first and first not in NOT_A_SEARCH and not q.strip().lower().endswith((".yaml", ".yml", ".json")) \
                and text.startswith("Found"):
            try:
                text += publish_3d(snap)
            except Exception as e:
                text += "\n(3D failed: %s)" % e
        pub.publish(String(data="> %s\n%s" % (q, text)))
    finally:
        busy.release()


rospy.Subscriber("/florence/command", String,
                 lambda m: threading.Thread(target=work, args=((m.data or "").strip(),), daemon=True).start())
rospy.spin()
