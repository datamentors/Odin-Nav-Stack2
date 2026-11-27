#!/usr/bin/env python3
"""Runs INSIDE ons2-go2 (ROS 1), against the running stack with `--with-cloud`. Used by test_stack.sh verify.
Listens to what cloud_publisher publishes for the cloud side and checks it: every message is JSON with the envelope and the
identity of config/cloud_publisher.yaml, `seq` increases per kind, and the kinds that must appear within the time limit do.
    python3 verify_topics.py [seconds, default 25]
Prints one PASS / FAIL line per check; exit code 1 if anything fails. No jsonschema needed: the full schema checks are in
tests/test_examples.py and tests/test_node.py. Florence events appear only after a search, so they are not required."""
import json
import os
import sys
import time

import rospy
import yaml
from std_msgs.msg import String

HERE = os.path.dirname(os.path.abspath(__file__))
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
KEYS = {"schema_version", "message_id", "timestamp", "source", "robot_id", "env", "fleet", "payload"}
TOPICS = {"telemetry": "/cloud_gateway/telemetry", "status": "/cloud_gateway/status", "events": "/cloud_gateway/events"}
REQUIRED = {"odin.pose.v1", "odin.status.v1"}
STATUS_PARTS = ("map", "nav", "stack")


def kinds_seen(got):
    kinds = set()
    for k in ("telemetry", "status"):
        for raw in got[k]:
            try:
                kinds.add(json.loads(raw)["payload"]["kind"])
            except (ValueError, KeyError, TypeError):
                pass
    return kinds


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "config", "cloud_publisher.yaml")))
    ident = cfg["identity"]
    rospy.init_node("verify_cloud_topics", anonymous=True)
    got = {k: [] for k in TOPICS}
    for k, topic in TOPICS.items():
        rospy.Subscriber(topic, String, lambda m, k=k: got[k].append(m.data), queue_size=100)
    end = time.time() + SECONDS
    while time.time() < end and not rospy.is_shutdown():
        time.sleep(0.2)
        if REQUIRED <= kinds_seen(got):
            break

    failed = 0

    def report(ok, text):
        nonlocal failed
        print(("PASS  " if ok else "FAIL  ") + text)
        failed += 0 if ok else 1

    seen = {}
    bad = 0
    for k in TOPICS:
        for raw in got[k]:
            try:
                d = json.loads(raw)
                ok = (set(d) == KEYS and isinstance(d["payload"], dict) and d["payload"].get("kind")
                      and d["robot_id"] == ident["robot_id"] and d["env"] == ident["env"] and d["fleet"] == ident["fleet"])
            except ValueError:
                ok = False
            if not ok:
                bad += 1
                continue
            seen.setdefault(d["payload"]["kind"], []).append(d["payload"])
    report(sum(len(v) for v in got.values()) > 0, "cloud_publisher publishes on %s" % ", ".join(TOPICS.values()))
    report(bad == 0, "every message has the envelope and the identity %s/%s/%s (%d bad)" % (ident["robot_id"], ident["env"], ident["fleet"], bad))
    for kind in sorted(REQUIRED):
        report(kind in seen, "%s seen (%d)" % (kind, len(seen.get(kind, []))))
    for kind, payloads in sorted(seen.items()):
        seqs = [p["seq"] for p in payloads]
        report(seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "%s: seq increases (%d..%d)" % (kind, seqs[0], seqs[-1]))
    status = seen.get("odin.status.v1")
    if status:
        last = status[-1]
        report(all(p in last for p in STATUS_PARTS), "odin.status.v1 has map, nav and stack")
        report(last["map"] is not None, "odin.status.v1: map in use is %s" % ((last["map"] or {}).get("name")))
    pose = seen.get("odin.pose.v1")
    if pose:
        report(pose[-1].get("relocalized") is True, "odin.pose.v1: relocalized")
    print("NOTE  events (Florence): %d messages (appear after a search)" % len(seen.get("odin.detections.florence.v1", [])))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
