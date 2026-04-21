
from __future__ import annotations

import csv
import json
from pathlib import Path

from ultralytics import YOLO

from detection_common import (
    build_coco_gt_from_yolo,
    collect_split_samples,
    compute_pr_by_class,
    get_class_names,
    run_coco_eval,
    xyxy_to_xywh,
)

# =========================
# USER SETTINGS
# =========================
DATA_YAML = r"C:/Users/Kenan/PycharmProjects/YoloTraining/datasets/my_dataset/data.yaml"
MODEL_WEIGHTS = r"C:/Users/Kenan/PycharmProjects/YoloTraining/runs/detect-train18/weights/best.pt"
OUTPUT_DIR = Path(r"C:/Users/Kenan/PycharmProjects/YoloTraining/analysis/yolov8n_metrics")
SPLIT = "val"
PRED_CONF_FOR_EXPORT = 0.001
PR_SCORE_THRESHOLD = 0.25
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
    samples = collect_split_samples(DATA_YAML, SPLIT)

    gt_json = OUTPUT_DIR / f"{SPLIT}_gt_coco.json"
    pred_json = OUTPUT_DIR / f"{SPLIT}_preds_coco.json"

    build_coco_gt_from_yolo(DATA_YAML, SPLIT, class_names, gt_json)

    model = YOLO(MODEL_WEIGHTS)
    preds_coco = []

    image_paths = [str(s.image_path) for s in samples]
    sample_by_name = {s.file_name: s for s in samples}

    results = model.predict(
        source=image_paths,
        conf=PRED_CONF_FOR_EXPORT,
        iou=0.7,
        verbose=False,
        stream=True,
    )

    for result in results:
        file_name = Path(result.path).name
        sample = sample_by_name[file_name]
        if result.boxes is None:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(xyxy, confs, clss):
            bbox = xyxy_to_xywh(box.tolist())
            preds_coco.append({
                "image_id": sample.image_id,
                "category_id": int(cls) + 1,
                "bbox": bbox,
                "score": float(score),
            })

    pred_json.write_text(json.dumps(preds_coco, indent=2), encoding="utf-8")

    global_map, per_class_ap = run_coco_eval(gt_json, pred_json, class_names)
    global_pr, per_class_pr = compute_pr_by_class(
        gt_json, pred_json, class_names,
        iou_threshold=0.5,
        score_threshold=PR_SCORE_THRESHOLD
    )

    # Merge AP + PR tables
    ap_by_name = {r["class_name"]: r for r in per_class_ap}
    merged_rows = []
    for row in per_class_pr:
        ap_row = ap_by_name[row["class_name"]]
        merged_rows.append({
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "TP": row["TP"],
            "FP": row["FP"],
            "FN": row["FN"],
            "Precision@0.5": row["Precision@0.5"],
            "Recall@0.5": row["Recall@0.5"],
            "AP50": ap_row["AP50"],
            "AP50_95": ap_row["AP50_95"],
            "AR50_95": ap_row["AR50_95"],
        })

    save_csv(
        OUTPUT_DIR / "per_class_metrics.csv",
        merged_rows,
        ["class_id", "class_name", "TP", "FP", "FN", "Precision@0.5", "Recall@0.5", "AP50", "AP50_95", "AR50_95"]
    )

    summary = {
        "model_weights": MODEL_WEIGHTS,
        "split": SPLIT,
        "prediction_conf_for_export": PRED_CONF_FOR_EXPORT,
        "precision_recall_score_threshold": PR_SCORE_THRESHOLD,
        "global_precision_at_0_5": global_pr["Precision@0.5"],
        "global_recall_at_0_5": global_pr["Recall@0.5"],
        "global_AP50": global_map["AP50"],
        "global_AP50_95": global_map["AP50_95"],
    }
    (OUTPUT_DIR / "global_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Predictions JSON: {pred_json}")
    print(f"Per-class CSV: {OUTPUT_DIR / 'per_class_metrics.csv'}")
    print(f"Global metrics JSON: {OUTPUT_DIR / 'global_metrics.json'}")


if __name__ == "__main__":
    main()
