#!/usr/bin/env python3
"""
CLI script to benchmark people counting accuracy against ground-truth dataset annotations.
"""

import argparse
import sys
from src.analytics.counting_evaluator import evaluate_dataset_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate People Counting Accuracy against Ground Truth")
    parser.add_argument(
        "--weights",
        default="runs/detect/crowd_people_detector/weights/best.pt",
        help="Path to trained YOLO weights",
    )
    parser.add_argument(
        "--test-dir",
        default="label-1/test",
        help="Path to test split directory with images/ and labels/",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of test frames to evaluate (default: 25)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device (e.g. '0' or 'cpu')",
    )
    args = parser.parse_args()

    print(f"Benchmarking counting accuracy on {args.samples} test frames...")
    try:
        res = evaluate_dataset_samples(
            weights_path=args.weights,
            test_dir=args.test_dir,
            num_samples=args.samples,
            conf_threshold=args.conf,
            device=args.device,
        )
        print("\n" + res.summary())
        return 0
    except Exception as e:
        print(f"[ERROR] Counting benchmark failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

