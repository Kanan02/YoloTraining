"""Check every ImDUSTRY5-1.5K label file against the integrity rules in the paper.

    python audit_labels.py                  # audit the dataset next to this file
    python audit_labels.py --json out.json  # also write the full report

Structural checks. Each finding is a violation, and the script exits non-zero
if there is any:

    fields      a row without exactly five fields
    class_id    a class identifier outside 0-10
    unit_range  a centre coordinate, width or height outside [0, 1]
    zero_size   a box of zero width or height
    border      a box whose extent crosses the image border
    duplicate   the same row repeated within one file
    aspect      a box more than twenty times longer than it is wide, in pixels

Annotation-rule checks, reported separately because they measure the protocol
rather than file integrity:

    min_size    a box under 12 pixels on its shorter side, below rule R4
    r1_area     a Support or Table box failing the area test of rule R1 - the box
                should cover at least 40% of the image while no component box
                covers 10% or more. Rule R12 annotates the finished-frame trolley
                as Support whether or not it passes, so a flagged Support box is
                not by itself an error.

Empty label files are listed rather than counted: an empty file marks an image
with no annotatable object.
"""
import argparse
import json
import os
import sys
from collections import Counter

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["Bolt", "Frame", "Wheel", "Wheel Support", "Wrench", "Box",
           "Flange", "Nut", "Support", "Table", "Washer"]
CONTEXT = {CLASSES.index("Support"), CLASSES.index("Table")}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
STRUCTURAL = ["fields", "class_id", "unit_range", "zero_size", "border", "duplicate", "aspect"]
MIN_SIDE_PX = 12
EPS = 1e-6


def audit_file(label_path, width, height):
    findings, boxes, seen = [], [], set()
    lines = [ln for ln in open(label_path, encoding="utf-8").read().splitlines() if ln.strip()]
    for number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            findings.append(("fields", number))
            continue
        if line.strip() in seen:
            findings.append(("duplicate", number))
        seen.add(line.strip())
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        if not 0 <= cls < len(CLASSES):
            findings.append(("class_id", number))
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            findings.append(("unit_range", number))
        if w <= 0 or h <= 0:
            findings.append(("zero_size", number))
            continue
        if x - w / 2 < -EPS or x + w / 2 > 1 + EPS or y - h / 2 < -EPS or y + h / 2 > 1 + EPS:
            findings.append(("border", number))
        wp, hp = w * width, h * height
        if max(wp / hp, hp / wp) > 20:
            findings.append(("aspect", number))
        if min(wp, hp) < MIN_SIDE_PX:
            findings.append(("min_size", number))
        boxes.append((cls, w * h))
    largest_component = max((a for c, a in boxes if c not in CONTEXT), default=0.0)
    for cls, area in boxes:
        if cls in CONTEXT and not (area >= 0.40 and largest_component < 0.10):
            findings.append(("r1_area", CLASSES[cls]))
    return findings, len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT, help="directory holding images/ and labels/")
    ap.add_argument("--json", default=None, help="write the full report to this file")
    args = ap.parse_args()

    structural, min_size, r1, empty, details = Counter(), 0, Counter(), [], []
    files = rows = 0
    for split in ("train", "val"):
        image_dir = os.path.join(args.root, "images", split)
        for name in sorted(os.listdir(image_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            label = os.path.join(args.root, "labels", split, stem + ".txt")
            if not os.path.exists(label):
                structural["missing_label_file"] += 1
                details.append({"file": f"{split}/{stem}", "check": "missing_label_file"})
                continue
            with Image.open(os.path.join(image_dir, name)) as im:
                width, height = im.size
            findings, n = audit_file(label, width, height)
            files += 1
            rows += n
            if n == 0:
                empty.append(f"{split}/{stem}")
            for check, where in findings:
                if check == "r1_area":
                    r1[where] += 1
                elif check == "min_size":
                    min_size += 1
                else:
                    structural[check] += 1
                details.append({"file": f"{split}/{stem}", "check": check, "where": where})

    print(f"label files {files}, rows {rows}")
    print("structural checks")
    for check in STRUCTURAL + (["missing_label_file"] if structural["missing_label_file"] else []):
        print(f"  {check:<12} {structural[check]}")
    print("annotation-rule checks")
    print(f"  min_size     {min_size}")
    print(f"  r1_area      {sum(r1.values())} {dict(r1) if r1 else ''}")
    print(f"empty label files {len(empty)} {empty if empty else ''}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"label_files": files, "rows": rows, "structural": dict(structural),
                       "min_size": min_size, "r1_area_flagged": dict(r1),
                       "empty_label_files": empty, "details": details}, f, indent=1)

    sys.exit(1 if sum(structural.values()) else 0)


if __name__ == "__main__":
    main()
