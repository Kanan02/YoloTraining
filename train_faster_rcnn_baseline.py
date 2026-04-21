from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

try:
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
except Exception:
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights = None

from detection_common import (
    build_coco_gt_from_yolo,
    collect_split_samples,
    compute_pr_by_class,
    get_class_names,
    parse_yolo_label_file,
    run_coco_eval,
    yolo_to_xyxy_abs,
    xyxy_to_xywh,
)

# =========================
# USER SETTINGS
# =========================
DATA_YAML = r"C:/Users/Kenan/PycharmProjects/YoloTraining/datasets/my_dataset/data.yaml"
OUTPUT_DIR = Path(r"C:/Users/Kenan/PycharmProjects/YoloTraining/analysis/faster_rcnn_mbv3_320")

NUM_EPOCHS = 12
BATCH_SIZE = 1          # safest for 2 GB VRAM
NUM_WORKERS = 0         # safest on Windows
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


class YoloDetectionDataset(Dataset):
    def __init__(self, data_yaml_path: str, split: str):
        self.class_names = get_class_names(data_yaml_path)
        self.samples = collect_split_samples(data_yaml_path, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample.image_path).convert("RGB")
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        labels = parse_yolo_label_file(sample.label_path)

        boxes = []
        labels_out = []
        areas = []

        for cls, x_c, y_c, w, h in labels:
            xyxy = yolo_to_xyxy_abs(x_c, y_c, w, h, sample.width, sample.height)
            x1, y1, x2, y2 = xyxy

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(xyxy)
            labels_out.append(cls + 1)  # 0 is background in Faster R-CNN
            areas.append((x2 - x1) * (y2 - y1))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels_out, dtype=torch.int64) if labels_out else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([sample.image_id]),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels_out),), dtype=torch.int64),
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes: int):
    """
    num_classes = number of foreground classes + background
    """
    if FasterRCNN_MobileNet_V3_Large_320_FPN_Weights is not None:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )
    else:
        # fallback for older torchvision
        model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def export_predictions_to_coco_json(model, dataset: YoloDetectionDataset, out_json_path: Path, score_threshold: float = 0.001):
    model.eval()
    model.to(DEVICE)

    preds = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for image_tensor, target in loader:
        img = image_tensor[0].to(DEVICE)
        outputs = model([img])[0]
        image_id = int(target[0]["image_id"].item())

        boxes = outputs["boxes"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if float(score) < score_threshold:
                continue

            preds.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": xyxy_to_xywh(box.tolist()),
                "score": float(score),
            })

    out_json_path.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    return preds


def train_one_epoch(model, optimizer, data_loader, epoch: int):
    model.train()
    running_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += float(losses.item())

    avg_loss = running_loss / max(1, len(data_loader))
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - loss: {avg_loss:.4f}")
    return avg_loss


def main():
    ensure_dir(OUTPUT_DIR)

    class_names = get_class_names(DATA_YAML)
    num_classes = len(class_names) + 1  # + background

    train_dataset = YoloDetectionDataset(DATA_YAML, "train")
    val_dataset = YoloDetectionDataset(DATA_YAML, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = create_model(num_classes=num_classes).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, NUM_EPOCHS // 2),
        gamma=0.1,
    )

    best_map = -1.0
    best_weights_path = OUTPUT_DIR / "best_fasterrcnn_mobilenet_v3_320_fpn.pt"
    gt_json = OUTPUT_DIR / "val_gt_coco.json"

    build_coco_gt_from_yolo(DATA_YAML, "val", class_names, gt_json)

    history = []

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, optimizer, train_loader, epoch)
        lr_scheduler.step()

        pred_json = OUTPUT_DIR / f"val_preds_epoch_{epoch + 1}.json"
        export_predictions_to_coco_json(
            model,
            val_dataset,
            pred_json,
            score_threshold=PRED_CONF_FOR_EXPORT,
        )

        global_map, _ = run_coco_eval(gt_json, pred_json, class_names)
        val_map = global_map["AP50_95"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_AP50": global_map["AP50"],
            "val_AP50_95": global_map["AP50_95"],
        })

        print(
            f"Epoch {epoch + 1} - "
            f"val AP50: {global_map['AP50']:.4f}, "
            f"val AP50:0.95: {global_map['AP50_95']:.4f}"
        )

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_weights_path)
            print(f"Saved new best model to: {best_weights_path}")

    # final eval using best model
    model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))

    final_pred_json = OUTPUT_DIR / "val_preds_best.json"
    export_predictions_to_coco_json(
        model,
        val_dataset,
        final_pred_json,
        score_threshold=PRED_CONF_FOR_EXPORT,
    )

    global_map, per_class_ap = run_coco_eval(gt_json, final_pred_json, class_names)
    global_pr, per_class_pr = compute_pr_by_class(
        gt_json,
        final_pred_json,
        class_names,
        iou_threshold=0.5,
        score_threshold=PR_SCORE_THRESHOLD,
    )

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
        OUTPUT_DIR / "training_history.csv",
        history,
        ["epoch", "train_loss", "val_AP50", "val_AP50_95"],
    )

    save_csv(
        OUTPUT_DIR / "per_class_metrics.csv",
        merged_rows,
        ["class_id", "class_name", "TP", "FP", "FN", "Precision@0.5", "Recall@0.5", "AP50", "AP50_95", "AR50_95"],
    )

    summary = {
        "best_weights": str(best_weights_path),
        "device": DEVICE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "global_precision_at_0_5": global_pr["Precision@0.5"],
        "global_recall_at_0_5": global_pr["Recall@0.5"],
        "global_AP50": global_map["AP50"],
        "global_AP50_95": global_map["AP50_95"],
    }

    (OUTPUT_DIR / "global_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Best weights: {best_weights_path}")
    print(f"Training history: {OUTPUT_DIR / 'training_history.csv'}")
    print(f"Per-class metrics: {OUTPUT_DIR / 'per_class_metrics.csv'}")
    print(f"Global metrics: {OUTPUT_DIR / 'global_metrics.json'}")


if __name__ == "__main__":
    main()
