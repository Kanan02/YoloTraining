"""Evaluate a checkpoint on the official validation split.

    python evaluate.py                              # the released weights
    python evaluate.py --weights runs/yolov8s/weights/best.pt

Prints overall and per-class metrics and writes per_class_metrics.csv.
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ["Bolt", "Frame", "Wheel", "Wheel Support", "Wrench", "Box",
           "Flange", "Nut", "Support", "Table", "Washer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolov8s_v2.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="per_class_metrics.csv")
    args = ap.parse_args()

    from ultralytics import YOLO

    r = YOLO(args.weights).val(data=str(ROOT / "data.yaml"), imgsz=args.imgsz,
                               batch=args.batch, device=args.device, plots=False)

    print(f"\n{'':<16}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    print(f"{'all':<16}{r.box.mp:>8.3f}{r.box.mr:>8.3f}{r.box.map50:>9.3f}{r.box.map:>10.3f}")

    rows = []
    for i, c in enumerate(r.box.ap_class_index):
        row = {"class": CLASSES[int(c)],
               "precision": round(float(r.box.p[i]), 4),
               "recall": round(float(r.box.r[i]), 4),
               "AP50": round(float(r.box.ap50[i]), 4),
               "AP50_95": round(float(r.box.ap[i]), 4)}
        rows.append(row)
        print(f"{row['class']:<16}{row['precision']:>8.3f}{row['recall']:>8.3f}"
              f"{row['AP50']:>9.3f}{row['AP50_95']:>10.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
