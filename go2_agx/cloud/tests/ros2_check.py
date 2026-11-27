#!/usr/bin/env python3
"""Runs INSIDE the odin-pulse-cloud container (ROS 2 Humble). Listens for a while on what the Pulse cloud_gateway receives from
our node (through the ROS 1 -> ROS 2 bridge) and on the gateway's own counters, then prints one line per check.
    python3 ros2_check.py [seconds] [robot_id]
Lines start with PASS, FAIL or NOTE. Exit code 1 if anything FAILs."""
import json
import sys
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from std_msgs.msg import String

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
ROBOT = sys.argv[2] if len(sys.argv) > 2 else "unitree-go2-000"
KEYS = {"schema_version", "message_id", "timestamp", "source", "robot_id", "env", "fleet", "payload"}
TOPICS = {"telemetry": "/cloud_gateway/telemetry", "status": "/cloud_gateway/status", "events": "/cloud_gateway/events"}
WANT = {"telemetry": {"odin.pose.v1"}, "status": {"odin.status.v1"}}   # events only appear when a Florence search is sent


class Check(Node):
    def __init__(self):
        super().__init__("odin_cloud_check")
        self.msgs = {k: [] for k in TOPICS}
        self.counts = []
        for k, topic in TOPICS.items():
            self.create_subscription(String, topic, lambda m, k=k: self.msgs[k].append(m.data), 50)
        self.create_subscription(DiagnosticArray, "/diagnostics", self.on_diag, 50)

    def on_diag(self, m):
        for st in m.status:
            kv = {x.key: x.value for x in st.values}
            if "outbound_published_count" in kv:
                self.counts.append((time.time(), int(kv["outbound_published_count"]), kv.get("broker_type"),
                                    kv.get("active"), kv.get("robot_id")))


def main():
    rclpy.init()
    node = Check()
    end = time.time() + SECONDS
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    failed = 0

    def report(ok, text):
        nonlocal failed
        print(("PASS  " if ok else "FAIL  ") + text)
        failed += 0 if ok else 1

    for k in TOPICS:
        parsed, bad = [], 0
        for raw in node.msgs[k]:
            try:
                d = json.loads(raw)
                if set(d) == KEYS and isinstance(d["payload"], dict) and d["payload"].get("kind") and d["robot_id"] == ROBOT:
                    parsed.append(d)
                else:
                    bad += 1
            except ValueError:
                bad += 1
        kinds = {d["payload"]["kind"] for d in parsed}
        if k in WANT:
            report(len(parsed) > 0 and WANT[k] <= kinds, "%s: %d messages, kinds %s" % (TOPICS[k], len(parsed), sorted(kinds)))
        else:
            print("NOTE  %s: %d messages (appear when a Florence search is sent)" % (TOPICS[k], len(parsed)))
        if node.msgs[k]:
            report(bad == 0, "%s: every message has the envelope and robot_id %s (%d bad)" % (TOPICS[k], ROBOT, bad))
        seqs = {}
        for d in parsed:
            seqs.setdefault(d["payload"]["kind"], []).append(d["payload"]["seq"])
        for kind, s in seqs.items():
            report(s == sorted(s) and len(set(s)) == len(s), "%s: seq increases (%d..%d)" % (kind, s[0], s[-1]))

    if node.counts:
        first, last = node.counts[0], node.counts[-1]
        report(last[1] > first[1], "gateway outbound_published_count %d -> %d (broker %s, active %s)" % (first[1], last[1], last[2], last[3]))
    else:
        report(False, "no cloud_gateway diagnostics seen on /diagnostics")
    node.destroy_node()
    rclpy.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
