
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

from detection_common import (
    build_coco_gt_from_yolo,
    collect_split_samples,
    get_class_names,
    parse_yolo_label_file,
)

# =========================
# USER SETTINGS
# =========================
DATA_YAML = r"C:/Users/Kenan/PycharmProjects/YoloTraining/datasets/my_dataset/data.yaml"
OUTPUT_DIR = Path(r"C:/Users/Kenan/PycharmProjects/YoloTraining/analysis/raw_stats")
# =========================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    ensure_dir(OUTPUT_DIR)
    class_names = get_class_names(DATA_YAML)

    split_image_counts = {}
    split_instance_counts = {}
    split_class_counts = defaultdict(Counter)
    image_level_rows = []
    bbox_rows = []
    all_objects_per_image = []
    resolutions = Counter()

    for split in ["train", "val"]:
        samples = collect_split_samples(DATA_YAML, split)
        split_image_counts[split] = len(samples)
        split_instance_counts[split] = 0

        for sample in samples:
            labels = parse_yolo_label_file(sample.label_path)
            obj_count = len(labels)
            split_instance_counts[split] += obj_count
            all_objects_per_image.append(obj_count)
            resolutions[(sample.width, sample.height)] += 1

            image_level_rows.append({
                "split": split,
                "image_id": sample.image_id,
                "file_name": sample.file_name,
                "width": sample.width,
                "height": sample.height,
                "objects_in_image": obj_count,
            })

            for cls, x_c, y_c, w, h in labels:
                split_class_counts[split][class_names[cls]] += 1
                bbox_rows.append({
                    "split": split,
                    "file_name": sample.file_name,
                    "class_id": cls,
                    "class_name": class_names[cls],
                    "bbox_width_norm": w,
                    "bbox_height_norm": h,
                    "bbox_area_norm": w * h,
                    "bbox_width_px": w * sample.width,
                    "bbox_height_px": h * sample.height,
                    "bbox_area_px": (w * sample.width) * (h * sample.height),
                    "image_width": sample.width,
                    "image_height": sample.height,
                })

    total_images = sum(split_image_counts.values())
    total_instances = sum(split_instance_counts.values())

    summary = {
        "total_images": total_images,
        "train_images": split_image_counts["train"],
        "val_images": split_image_counts["val"],
        "total_instances": total_instances,
        "train_instances": split_instance_counts["train"],
        "val_instances": split_instance_counts["val"],
        "avg_objects_per_image": round(mean(all_objects_per_image), 4) if all_objects_per_image else 0,
        "median_objects_per_image": round(median(all_objects_per_image), 4) if all_objects_per_image else 0,
        "min_objects_per_image": min(all_objects_per_image) if all_objects_per_image else 0,
        "max_objects_per_image": max(all_objects_per_image) if all_objects_per_image else 0,
        "unique_resolutions": len(resolutions),
        "most_common_resolutions": [
            {"width": w, "height": h, "count": c}
            for (w, h), c in resolutions.most_common(10)
        ],
        "avg_bbox_area_norm": round(mean([r["bbox_area_norm"] for r in bbox_rows]), 6) if bbox_rows else 0,
        "median_bbox_area_norm": round(median([r["bbox_area_norm"] for r in bbox_rows]), 6) if bbox_rows else 0,
    }

    # Save summary JSON
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save class counts by split
    class_rows = []
    total_class_counts = Counter()
    for split in ["train", "val"]:
        for class_name in class_names:
            count = split_class_counts[split][class_name]
            total_class_counts[class_name] += count
            class_rows.append({
                "split": split,
                "class_name": class_name,
                "instances": count,
            })
    for class_name in class_names:
        class_rows.append({
            "split": "total",
            "class_name": class_name,
            "instances": total_class_counts[class_name],
        })

    save_csv(OUTPUT_DIR / "class_counts_by_split.csv", class_rows, ["split", "class_name", "instances"])
    save_csv(OUTPUT_DIR / "image_level_stats.csv", image_level_rows,
             ["split", "image_id", "file_name", "width", "height", "objects_in_image"])
    save_csv(OUTPUT_DIR / "bbox_stats.csv", bbox_rows,
             ["split", "file_name", "class_id", "class_name", "bbox_width_norm", "bbox_height_norm",
              "bbox_area_norm", "bbox_width_px", "bbox_height_px", "bbox_area_px",
              "image_width", "image_height"])

    # Also save GT in COCO format for later re-use
    build_coco_gt_from_yolo(DATA_YAML, "val", class_names, OUTPUT_DIR / "val_gt_coco.json")

    # Plot 1: objects per image histogram
    plt.figure(figsize=(8, 5))
    bins = range(0, max(all_objects_per_image) + 2) if all_objects_per_image else [0, 1]
    plt.hist(all_objects_per_image, bins=bins, edgecolor="black")
    plt.xlabel("Objects per image")
    plt.ylabel("Number of images")
    plt.title("Objects per image distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "objects_per_image_hist.png", dpi=200)
    plt.close()

    # Plot 2: class counts total
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, [total_class_counts[c] for c in class_names], edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Instances")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution_total.png", dpi=200)
    plt.close()

    # Plot 3: bbox area distribution
    plt.figure(figsize=(8, 5))
    areas = [r["bbox_area_norm"] for r in bbox_rows]
    plt.hist(areas, bins=30, edgecolor="black")
    plt.xlabel("Normalized bbox area")
    plt.ylabel("Frequency")
    plt.title("Normalized bounding-box area distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bbox_area_distribution.png", dpi=200)
    plt.close()

    print("Done.")
    print(f"Summary saved to: {OUTPUT_DIR / 'summary.json'}")
    print(f"Class counts saved to: {OUTPUT_DIR / 'class_counts_by_split.csv'}")
    print(f"Image-level stats saved to: {OUTPUT_DIR / 'image_level_stats.csv'}")
    print(f"BBox stats saved to: {OUTPUT_DIR / 'bbox_stats.csv'}")


if __name__ == "__main__":
    main()
