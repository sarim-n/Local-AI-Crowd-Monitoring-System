#!/usr/bin/env python3
"""
CLI entrypoint for evaluating a fine-tuned YOLO people detector on held-out splits.
"""

import argparse
import sys
from src.evaluator import EvaluationConfig, PeopleDetectorEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned YOLO people detector (Phase 1)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/crowd_people_detector/weights/best.pt",
        help="Path to trained model weights .pt (default: runs/detect/crowd_people_detector/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="label-1/data.yaml",
        help="Path to dataset configuration YAML (default: label-1/data.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. '0' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output project directory for evaluation artifacts (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="eval_test",
        help="Name of evaluation run folder (default: eval_test)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU threshold (default: 0.6)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = EvaluationConfig(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        conf=args.conf,
        iou=args.iou,
    )

    evaluator = PeopleDetectorEvaluator(config)
    try:
        report = evaluator.evaluate()
        print(f"\nEvaluation completed. Report saved in: {report['output_directory']}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

