#!/usr/bin/env python3
"""Pick evenly spaced frames from ROS 1 bags (sensor_msgs/CompressedImage or Image) as JPEGs for labelling.

Run where ROS Noetic python is available, e.g. on the Go2 inside the ONS2 container:
  docker exec -it ons2-go2 bash -lc "source /opt/ros/noetic/setup.bash && \
    python3 ~/ons2/detbench/extract_frames.py --out ~/bench_frames ~/bags/*.bag"
Defaults: topic /odin1/image/compressed, one frame every 1.5 s, frames darker than --min-mean skipped,
near-duplicates (robot standing still) skipped when the small grey thumbnail barely changed.
"""
import argparse, os, sys

import cv2
import numpy as np
import rosbag


def to_bgr(msg):
    if hasattr(msg, "format"):  # CompressedImage
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    enc, a = msg.encoding.lower(), np.frombuffer(msg.data, np.uint8)
    img = a.reshape(msg.height, msg.step)[:, :msg.width * (3 if "8uc3" in enc or enc in ("rgb8", "bgr8") else 1)]
    img = img.reshape(msg.height, msg.width, -1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bags", nargs="+")
    p.add_argument("--topic", default="/odin1/image/compressed")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=float, default=1.5, help="seconds between kept frames")
    p.add_argument("--max", type=int, default=0, help="stop after N frames in total")
    p.add_argument("--min-mean", type=float, default=15, help="skip nearly black frames")
    p.add_argument("--min-change", type=float, default=4.0, help="mean abs diff of 64x48 thumbnails to keep a frame")
    p.add_argument("--quality", type=int, default=95)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for bag in a.bags:
        tag = os.path.splitext(os.path.basename(bag))[0]
        last_t, last_th, kept = None, None, 0
        with rosbag.Bag(bag) as b:
            if a.topic not in b.get_type_and_topic_info().topics:
                print("%s: no topic %s (has: %s)" % (bag, a.topic, ", ".join(sorted(b.get_type_and_topic_info().topics))))
                continue
            for _, msg, t in b.read_messages(topics=[a.topic]):
                st = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else t.to_sec()
                if last_t is not None and st - last_t < a.every:
                    continue
                img = to_bgr(msg)
                if img is None or img.mean() < a.min_mean:
                    continue
                th = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32)
                if last_th is not None and np.abs(th - last_th).mean() < a.min_change:
                    continue
                last_t, last_th = st, th
                cv2.imwrite(os.path.join(a.out, "%s_%.3f.jpg" % (tag, st)), img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                kept += 1
                n += 1
                if a.max and n >= a.max:
                    break
        print("%s: %d frames" % (bag, kept))
        if a.max and n >= a.max:
            break
    print("total %d frames in %s" % (n, a.out))


if __name__ == "__main__":
    sys.exit(main())
