#!/usr/bin/env python3
"""verify_topics.py with fake ROS: good messages pass; a wrong identity, a missing kind and a broken envelope fail."""
import contextlib
import copy
import io
import json
import pathlib
import runpy
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent


def run(messages_by_topic, seconds="1"):
    handlers = {}

    def subscriber(topic, cls, cb, queue_size=1):
        for raw in messages_by_topic.get(topic, []):
            cb(types.SimpleNamespace(data=raw))

    sys.modules["rospy"] = types.SimpleNamespace(init_node=lambda *a, **k: None, Subscriber=subscriber, is_shutdown=lambda: False)
    sys.modules["std_msgs"] = types.ModuleType("std_msgs")
    sys.modules["std_msgs.msg"] = types.SimpleNamespace(String=object)
    out = io.StringIO()
    old = sys.argv
    sys.argv = ["verify_topics.py", seconds]
    code = 0
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(ROOT / "tests" / "verify_topics.py"), run_name="__main__")
    except SystemExit as e:
        code = e.code or 0
    finally:
        sys.argv = old
    return code, out.getvalue()


def example(kind):
    return json.loads((ROOT / "examples" / (kind + ".json")).read_text())


def main():
    good = {"/cloud_gateway/telemetry": [json.dumps(example("odin.pose.v1"))],
            "/cloud_gateway/status": [json.dumps(example("odin.status.v1"))]}
    checks = []
    code, text = run(good)
    checks.append(("good messages pass", code == 0 and "FAIL" not in text))

    wrong = copy.deepcopy(good)
    m = example("odin.pose.v1"); m["robot_id"] = "someone-else"
    wrong["/cloud_gateway/telemetry"] = [json.dumps(m)]
    code, text = run(wrong)
    checks.append(("wrong identity fails", code == 1 and "bad" in text))

    code, text = run({"/cloud_gateway/telemetry": good["/cloud_gateway/telemetry"]})
    checks.append(("missing odin.status.v1 fails", code == 1 and "odin.status.v1 seen (0)" in text))

    broken = {"/cloud_gateway/telemetry": ["not json"], "/cloud_gateway/status": good["/cloud_gateway/status"]}
    code, text = run(broken)
    checks.append(("broken message fails", code == 1))

    nomap = example("odin.status.v1"); nomap["payload"]["map"] = None
    code, text = run({"/cloud_gateway/telemetry": good["/cloud_gateway/telemetry"], "/cloud_gateway/status": [json.dumps(nomap)]})
    checks.append(("status without a map fails", code == 1 and "map in use" in text))

    failed = 0
    for name, ok in checks:
        print("%s  %s" % ("PASS" if ok else "FAIL", name))
        failed += 0 if ok else 1
    print("== %d checks, %d failed" % (len(checks), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
