"""
train_custom_yolo.py  —  Member 2: Fine-tune YOLOv8s on custom phone dataset
CSCI435 AI Exam Proctoring System — University of Wollongong in Dubai

Usage (local dataset):
    python src/train_custom_yolo.py \
        --data-yaml roboflow_dataset/data.yaml \
        --base-weights yolov8s.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 8 \
        --device cpu
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8s for exam proctoring")

    # Dataset
    parser.add_argument("--data-yaml", type=str, required=True,
                        help="Path to data.yaml (YOLOv8 format)")

    # Model
    parser.add_argument("--base-weights", type=str, default="yolov8s.pt",
                        help="Starting weights (default: yolov8s.pt)")

    # Training hyperparameters
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--device",  type=str,   default="cpu",
                        help="'cpu' or '0' for GPU")
    parser.add_argument("--workers", type=int,   default=2,
                        help="DataLoader workers (use 0 on Windows if errors occur)")

    # Output
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Directory to save training runs")
    parser.add_argument("--name",    type=str, default="proctor_train",
                        help="Run name (output saved to project/name/)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────────
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found at: {data_yaml.resolve()}")
        print("  Make sure you extracted the Roboflow ZIP and the path is correct.")
        sys.exit(1)

    weights = Path(args.base_weights)
    if not weights.exists():
        print(f"[ERROR] Base weights not found at: {weights.resolve()}")
        print("  Download yolov8s.pt from: https://github.com/ultralytics/assets/releases")
        sys.exit(1)

    # ── Import ultralytics ─────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run:  pip install ultralytics")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Loading base model : {args.base_weights}")
    print(f"  Dataset            : {args.data_yaml}")
    print(f"  Epochs             : {args.epochs}")
    print(f"  Image size         : {args.imgsz}")
    print(f"  Batch size         : {args.batch}")
    print(f"  Device             : {args.device}")
    print(f"  Output             : {args.project}/{args.name}/")
    print(f"{'='*60}\n")

    model = YOLO(str(weights))

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[INFO] Starting fine-tuning...")
    results = model.train(
        data=str(data_yaml.resolve()),   # absolute path avoids working-dir issues
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,                   # overwrite previous run with same name
        pretrained=True,                 # start from base weights (transfer learning)
        patience=15,                     # early stopping if no improvement for 15 epochs
        save=True,
        plots=True,                      # saves confusion matrix, PR curve, etc.
        verbose=True,
    )

    # ── Report results ────────────────────────────────────────────────────────
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best weights saved to : {best_weights.resolve()}")
    print(f"\n  To use the fine-tuned model, run:")
    print(f"    python src/object_detection_proctor.py \\")
    print(f"        --webcam 0 \\")
    print(f"        --model {best_weights}")
    print(f"{'='*60}\n")

    # ── Quick validation on test set ──────────────────────────────────────────
    print("[INFO] Running validation on test set...")
    try:
        val_results = model.val(
            data=str(data_yaml.resolve()),
            split="test",
            device=args.device,
        )
        print(f"\n  mAP@0.5        : {val_results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95   : {val_results.box.map:.4f}")
        print(f"  Precision      : {val_results.box.mp:.4f}")
        print(f"  Recall         : {val_results.box.mr:.4f}")
    except Exception as e:
        print(f"[WARNING] Validation step failed: {e}")
        print("  You can run validation manually with evaluate_model.py")

    print("\n[INFO] Done. Check your runs/detect/proctor_train/ folder for all outputs.")


if __name__ == "__main__":
    main()
