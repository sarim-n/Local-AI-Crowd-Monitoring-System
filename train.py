#!/usr/bin/env python3
"""
CLI entrypoint for fine-tuning a YOLO people detector on the crowd monitoring dataset.
"""

import argparse
import sys
from src.trainer import PeopleDetectorTrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO people detector for local crowd monitoring (Phase 1)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="label-1/data.yaml",
        help="Path to dataset configuration YAML (default: label-1/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO base model (e.g. yolov8n.pt, yolo11n.pt, yolov8s.pt, default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs (default: 25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. '0' for CUDA GPU or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output project directory for training runs (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="crowd_people_detector",
        help="Name of the training experiment run (default: crowd_people_detector)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes (default: 2, set 0 for single-process)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
    )

    trainer = PeopleDetectorTrainer(config)
    try:
        summary = trainer.train()
        print("\nTraining completed successfully!")
        print(f"Best weights: {summary['best_weights_path']}")
        print(f"Output directory: {summary['output_directory']}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

