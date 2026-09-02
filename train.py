"""Reproduce the ImDUSTRY5-1.5K YOLOv8 baselines.

    python train.py                    # YOLOv8s, the reported baseline
    python train.py --model yolov8n.pt # the smaller baseline

Every setting that affects the result is fixed here, including the seed, so a
clean run reproduces the published numbers on the official split. Trained on a
single RTX 4060 (8 GB); lower --batch if your card has less memory.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained checkpoint")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="960 rather than 640: Bolt, Nut and Washer are small")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    name = args.name or Path(args.model).stem
    YOLO(args.model).train(
        data=str(ROOT / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name=name,
        seed=0,
        deterministic=True,
        patience=30,
        cos_lr=True,
        close_mosaic=15,
        workers=4,
        plots=True,
    )
    print(f"\nweights -> {ROOT / 'runs' / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
