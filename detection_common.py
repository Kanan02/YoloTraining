
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from PIL import Image

try:
    import yaml
except ImportError as e:
    raise ImportError("Please install pyyaml: pip install pyyaml") from e

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as e:
    raise ImportError("Please install pycocotools: pip install pycocotools") from e


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    width: int
    height: int
    image_id: int
    file_name: str


def load_data_yaml(data_yaml_path):
    path = Path(data_yaml_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML at {path}")
    return data


def get_class_names(data_yaml_path=None, fallback=None):
    if data_yaml_path:
        data = load_data_yaml(data_yaml_path)
        names = data.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        if isinstance(names, list):
            return names
    if fallback:
        return fallback
    raise ValueError("Could not determine class names. Provide data.yaml or fallback names.")


def _resolve_path(root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else (root / p)


def get_split_dirs(data_yaml_path, split):
    root = Path(data_yaml_path).resolve().parent
    data = load_data_yaml(data_yaml_path)
    images_dir = _resolve_path(root, data[split])
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    return images_dir, labels_dir


def find_image_for_label(images_dir, label_path):
    base = label_path.stem
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for label {label_path.name} in {images_dir}")


def parse_yolo_label_file(label_path):
    out = []
    path = Path(label_path)
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {path} at line {line_no}: {line}")
            cls = int(float(parts[0]))
            x_c, y_c, w, h = map(float, parts[1:])
            out.append((cls, x_c, y_c, w, h))
    return out


def yolo_to_xyxy_abs(x_c, y_c, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    xc = x_c * img_w
    yc = y_c * img_h
    x1 = max(0.0, xc - bw / 2.0)
    y1 = max(0.0, yc - bh / 2.0)
    x2 = min(float(img_w), xc + bw / 2.0)
    y2 = min(float(img_h), yc + bh / 2.0)
    return [x1, y1, x2, y2]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect_split_samples(data_yaml_path, split):
    images_dir, labels_dir = get_split_dirs(data_yaml_path, split)
    label_paths = sorted(labels_dir.glob("*.txt"))
    samples = []
    for idx, label_path in enumerate(label_paths, start=1):
        image_path = find_image_for_label(images_dir, label_path)
        with Image.open(image_path) as img:
            width, height = img.size
        samples.append(Sample(
            image_path=image_path,
            label_path=label_path,
            width=width,
            height=height,
            image_id=idx,
            file_name=image_path.name,
        ))
    return samples


def build_coco_gt_from_yolo(data_yaml_path, split, class_names, out_json_path=None):
    samples = collect_split_samples(data_yaml_path, split)
    images = []
    annotations = []
    ann_id = 1
    for sample in samples:
        images.append({
            "id": sample.image_id,
            "file_name": sample.file_name,
            "width": sample.width,
            "height": sample.height,
        })
        labels = parse_yolo_label_file(sample.label_path)
        for cls, x_c, y_c, w, h in labels:
            xyxy = yolo_to_xyxy_abs(x_c, y_c, w, h, sample.width, sample.height)
            xywh = xyxy_to_xywh(xyxy)
            annotations.append({
                "id": ann_id,
                "image_id": sample.image_id,
                "category_id": cls + 1,
                "bbox": xywh,
                "area": xywh[2] * xywh[3],
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    gt = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {"description": "Converted from YOLO format"},
        "licenses": [],
    }
    if out_json_path:
        out_path = Path(out_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return gt


def load_coco_from_json(json_path):
    coco = COCO()
    with Path(json_path).open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    coco.dataset = dataset
    coco.createIndex()
    return coco


def run_coco_eval(gt_json_path, pred_json_path, class_names):
    coco_gt = load_coco_from_json(gt_json_path)
    with Path(pred_json_path).open("r", encoding="utf-8") as f:
        preds = json.load(f)

    if len(preds) == 0:
        raise ValueError("Prediction list is empty. COCO evaluation cannot run.")

    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "AP50_95": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
    }

    precisions = coco_eval.eval["precision"]
    recalls = coco_eval.eval["recall"]
    iou_thresholds = list(coco_eval.params.iouThrs)
    ap50_index = iou_thresholds.index(0.5)

    per_class = []
    for class_idx, class_name in enumerate(class_names):
        precision_class = precisions[:, :, class_idx, 0, -1]
        precision_class = precision_class[precision_class > -1]
        ap5095 = float(precision_class.mean()) if precision_class.size else float("nan")

        precision_ap50 = precisions[ap50_index, :, class_idx, 0, -1]
        precision_ap50 = precision_ap50[precision_ap50 > -1]
        ap50 = float(precision_ap50.mean()) if precision_ap50.size else float("nan")

        recall_class = recalls[:, class_idx, 0, -1]
        recall_class = recall_class[recall_class > -1]
        ar5095 = float(recall_class.mean()) if recall_class.size else float("nan")

        per_class.append({
            "class_id": class_idx,
            "class_name": class_name,
            "AP50": ap50,
            "AP50_95": ap5095,
            "AR50_95": ar5095,
        })
    return stats, per_class


def compute_pr_by_class(gt_json_path, pred_json_path, class_names, iou_threshold=0.5, score_threshold=0.25):
    gt = json.loads(Path(gt_json_path).read_text(encoding="utf-8"))
    preds = json.loads(Path(pred_json_path).read_text(encoding="utf-8"))

    gt_by_image_class = defaultdict(list)
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        box = [x, y, x + w, y + h]
        gt_by_image_class[(ann["image_id"], ann["category_id"])].append(box)

    pred_by_image_class = defaultdict(list)
    for pred in preds:
        if pred["score"] < score_threshold:
            continue
        x, y, w, h = pred["bbox"]
        box = [x, y, x + w, y + h]
        pred_by_image_class[(pred["image_id"], pred["category_id"])].append((pred["score"], box))

    per_class_rows = []
    total_tp = total_fp = total_fn = 0
    image_ids = {img["id"] for img in gt["images"]}

    for class_idx, class_name in enumerate(class_names, start=1):
        tp = fp = fn = 0
        for image_id in image_ids:
            gt_boxes = gt_by_image_class.get((image_id, class_idx), [])
            pred_items = pred_by_image_class.get((image_id, class_idx), [])
            pred_items = sorted(pred_items, key=lambda x: x[0], reverse=True)

            matched = [False] * len(gt_boxes)

            for score, pred_box in pred_items:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if matched[gt_idx]:
                        continue
                    iou = box_iou_xyxy(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    matched[best_gt_idx] = True
                    tp += 1
                else:
                    fp += 1

            fn += matched.count(False)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_rows.append({
            "class_id": class_idx - 1,
            "class_name": class_name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision@0.5": precision,
            "Recall@0.5": recall,
        })
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall = {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision@0.5": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0,
        "Recall@0.5": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0,
    }
    return overall, per_class_rows
