#!/usr/bin/env python3
"""GPU load generator: Florence-2 exactly as florence_service.py runs it (florence2-base-ft, float32, CUDA,
<OPEN_VOCABULARY_DETECTION> "person"), back to back on the frames in a folder, until killed.
This is the WORST case: on the robot Florence-2 only runs when a command is sent (one search, or ~12 s for 'far').

  cd ~/odin-perception && .venv/bin/python ~/odin_test/detbench/florence_load.py ~/odin_test/detbench/data/frames_clip
"""
import glob, os, sys, time

import numpy as np
from PIL import Image

FRAMES = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.expanduser("~/odin-perception"))
os.chdir(os.path.expanduser("~/odin-perception"))  # model_dir is relative to the Odin-perception checkout
from inspection_perception.cli import make_backend  # noqa: E402

backend = make_backend("florence2", model_dir="models/florence2-base-ft", dtype="float32", device="cuda")
files = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
if not files:
    sys.exit("no .jpg frames in %s" % FRAMES)
task = "<OPEN_VOCABULARY_DETECTION>"
print("Florence-2 loaded, looping on %d frames" % len(files), flush=True)
n, t0 = 0, time.time()
while True:
    rgb = np.array(Image.open(files[n % len(files)]).convert("RGB"))
    t = time.time()
    backend._run_task(rgb, task, text_input="person", max_new_tokens=1024)
    n += 1
    if n % 10 == 0:
        print("florence calls %d, last %.0f ms, avg %.0f ms" % (n, 1000 * (time.time() - t), 1000 * (time.time() - t0) / n), flush=True)
