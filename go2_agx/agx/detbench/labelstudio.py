#!/usr/bin/env python3
"""Label Studio <-> COCO helpers for the detector benchmark (Python 3, standard library only).

  to-tasks   COCO pre-labels (prelabel.py) -> Label Studio tasks with the boxes as PREDICTIONS, images served by
             Label Studio local-files storage, plus the labelling config (one RectangleLabels per category).
  to-coco    Label Studio JSON export (Export -> JSON) -> COCO ground truth for bench.py. Every task must be
             SUBMITTED (an empty submission = "nobody here"); unsubmitted or skipped tasks stop the conversion.

Workflow (frames in ./frames_label, pre-labels in ./prelabels_person.json):
  python3 labelstudio.py to-tasks prelabels_person.json --image-dir frames_label \
      --out label_studio_tasks.json --config label_studio_config.xml
  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$PWD label-studio start
    project: paste the config; Cloud Storage -> Local files -> absolute path of frames_label (do NOT sync);
    Import label_studio_tasks.json; review and Submit every task; Export -> JSON
  python3 labelstudio.py to-coco project-1-at-....json --image-dir frames_label --out labels_person.json
"""
import argparse, json, os, sys


def image_name(task):
    return os.path.basename(task["data"]["image"].split("?d=")[-1])


def to_tasks(a):
    coco = json.load(open(a.coco))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_img = {}
    for an in coco["annotations"]:
        by_img.setdefault(an["image_id"], []).append(an)
    tasks = []
    for im in coco["images"]:
        W, H = im["width"], im["height"]
        res = []
        for an in by_img.get(im["id"], []):
            x, y, w, h = an["bbox"]
            res.append({"id": "p%d" % an["id"], "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                        "original_width": W, "original_height": H, "image_rotation": 0,
                        "value": {"x": 100 * x / W, "y": 100 * y / H, "width": 100 * w / W, "height": 100 * h / H,
                                  "rotation": 0, "rectanglelabels": [cats[an["category_id"]]]}})
        tasks.append({"data": {"image": "/data/local-files/?d=%s/%s" % (a.image_dir.strip("/"), im["file_name"])},
                      "predictions": [{"model_version": a.model_version, "result": res}]})
    json.dump(tasks, open(a.out, "w"), indent=1)
    if a.config:
        labels = "".join('    <Label value="%s"/>\n' % n for _, n in sorted(cats.items()))
        open(a.config, "w").write('<View>\n  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
                                  '  <RectangleLabels name="label" toName="image">\n%s  </RectangleLabels>\n</View>\n' % labels)
    print("%d tasks, %d proposed boxes -> %s%s" % (len(tasks), sum(len(t["predictions"][0]["result"]) for t in tasks),
                                                   a.out, (" + " + a.config) if a.config else ""))


def to_coco(a):
    export = json.load(open(a.export))
    images, anns, cats, problems = [], [], {}, []
    for k, t in enumerate(sorted(export, key=image_name)):
        name = image_name(t)
        sub = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not sub:
            problems.append("not submitted or skipped: " + name)
            continue
        res = [r for r in sub[-1]["result"] if r.get("type") == "rectanglelabels"]
        W = H = None
        for r in res:
            W, H = r["original_width"], r["original_height"]
        if W is None:  # empty frame: take the size from the image itself
            if not a.image_dir:
                problems.append("empty frame without --image-dir to read its size: " + name)
                continue
            W, H = jpeg_size(os.path.join(a.image_dir, name))
        images.append({"id": k + 1, "file_name": name, "width": W, "height": H})
        for r in res:
            v = r["value"]
            for label in v["rectanglelabels"]:
                cid = cats.setdefault(label, len(cats) + 1)
                x, y, w, h = v["x"] * W / 100, v["y"] * H / 100, v["width"] * W / 100, v["height"] * H / 100
                anns.append({"id": len(anns) + 1, "image_id": k + 1, "category_id": cid,
                             "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)], "area": round(w * h, 1),
                             "iscrowd": 0, "segmentation": []})
    if a.image_dir:
        on_disk = {f for f in os.listdir(a.image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
        missing = sorted(on_disk - {image_name(t) for t in export})
        problems += ["frame not in the export: " + m for m in missing]
    if problems:
        sys.exit("NOT converted, fix these in Label Studio and export again:\n  " + "\n  ".join(problems))
    json.dump({"info": {"description": "Ground truth (human-reviewed) from Label Studio export %s" % os.path.basename(a.export)},
               "images": images, "annotations": anns,
               "categories": [{"id": i, "name": n} for n, i in sorted(cats.items(), key=lambda kv: kv[1])]},
              open(a.out, "w"))
    empty = len(images) - len({x["image_id"] for x in anns})
    print("%d frames (%d empty), %d boxes %s -> %s" % (len(images), empty, len(anns),
                                                      {n: sum(1 for x in anns if x["category_id"] == i) for n, i in cats.items()}, a.out))


def jpeg_size(path):
    """(width, height) of a JPEG or PNG without third-party libraries."""
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
        f.seek(2)
        while True:
            marker, size = f.read(2), int.from_bytes(f.read(2), "big")
            if marker[0] != 0xFF:
                raise ValueError("not a JPEG: " + path)
            if marker[1] in (0xC0, 0xC1, 0xC2):
                f.read(1)
                h, w = int.from_bytes(f.read(2), "big"), int.from_bytes(f.read(2), "big")
                return w, h
            f.seek(size - 2, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("to-tasks", help="COCO pre-labels -> Label Studio tasks (+ config)")
    t.add_argument("coco")
    t.add_argument("--image-dir", default="frames_label", help="folder under LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT")
    t.add_argument("--out", default="label_studio_tasks.json")
    t.add_argument("--config", default="label_studio_config.xml", help="labelling config to write ('' = none)")
    t.add_argument("--model-version", default="prelabel")
    c = sub.add_parser("to-coco", help="Label Studio JSON export -> COCO ground truth")
    c.add_argument("export")
    c.add_argument("--image-dir", help="frames folder: checks every frame was labelled, sizes for empty frames")
    c.add_argument("--out", default="labels.json")
    a = p.parse_args()
    to_tasks(a) if a.cmd == "to-tasks" else to_coco(a)


if __name__ == "__main__":
    main()
