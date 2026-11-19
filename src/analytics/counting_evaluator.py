"""
Counting accuracy evaluation module.
Compares people counts against ground-truth annotations or manually supplied counts.
Calculates MAE, RMSE, MAPE, and Counting Accuracy percentage with clear mathematical explanation.
"""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class CountingEvaluationResult:
    num_samples: int
    total_ground_truth: int
    total_predicted: int
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    overall_accuracy_pct: float
    mean_sample_accuracy_pct: float
    sample_details: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = (
        "Accuracy Calculation:\n"
        "  1. Absolute Error per sample: e_i = |Pred_i - GT_i|\n"
        "  2. Mean Absolute Error (MAE) = (1/N) * sum(e_i)\n"
        "  3. Root Mean Squared Error (RMSE) = sqrt((1/N) * sum(e_i^2))\n"
        "  4. Overall Counting Accuracy = max(0, 1 - (sum(|Pred_i - GT_i|) / sum(GT_i))) * 100%\n"
        "  5. Mean Sample Accuracy = (1/N) * sum(max(0, 1 - (|Pred_i - GT_i| / max(1, GT_i)))) * 100%"
    )

    def summary(self) -> str:
        lines = [
            "================ PEOPLE COUNTING BENCHMARK REPORT ================",
            f"Samples Evaluated : {self.num_samples}",
            f"Total Ground Truth: {self.total_ground_truth} people",
            f"Total Predicted   : {self.total_predicted} people",
            f"Net Discrepancy   : {self.total_predicted - self.total_ground_truth:+d} people",
            "",
            "--- Accuracy Metrics ---",
            f"  Mean Absolute Error (MAE)       : {self.mean_absolute_error:.2f} people/frame",
            f"  Root Mean Squared Error (RMSE)  : {self.root_mean_squared_error:.2f}",
            f"  Mean Abs Percentage Error (MAPE): {self.mean_absolute_percentage_error:.1f}%",
            f"  Overall Counting Accuracy       : {self.overall_accuracy_pct:.2f}%",
            f"  Mean Per-Sample Accuracy        : {self.mean_sample_accuracy_pct:.2f}%",
            "",
            "--- Calculation Methodology ---",
            self.explanation,
            "=================================================================="
        ]
        return "\n".join(lines)


def calculate_counting_accuracy(
    gt_counts: List[int],
    pred_counts: List[int],
    sample_names: Optional[List[str]] = None,
) -> CountingEvaluationResult:
    """Compute benchmark counting metrics across sample pairs."""
    n = len(gt_counts)
    if n == 0 or len(pred_counts) != n:
        raise ValueError("Ground-truth and prediction lists must be non-empty and of equal length.")

    total_gt = sum(gt_counts)
    total_pred = sum(pred_counts)

    abs_errors = [abs(p - g) for g, p in zip(gt_counts, pred_counts)]
    sq_errors = [(p - g) ** 2 for g, p in zip(gt_counts, pred_counts)]

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)

    # Per-sample accuracies and MAPEs
    sample_accs = []
    mapes = []
    details = []

    for i in range(n):
        g = gt_counts[i]
        p = pred_counts[i]
        err = abs(p - g)
        denom = max(1, g)
        sample_acc = max(0.0, 1.0 - (err / denom)) * 100.0
        mape_i = (err / denom) * 100.0
        sample_accs.append(sample_acc)
        mapes.append(mape_i)

        name = sample_names[i] if sample_names and i < len(sample_names) else f"sample_{i+1}"
        details.append({
            "sample": name,
            "ground_truth": g,
            "predicted": p,
            "error": p - g,
            "abs_error": err,
            "accuracy_pct": round(sample_acc, 1)
        })

    # Overall dataset-level accuracy
    if total_gt > 0:
        overall_acc = max(0.0, 1.0 - (sum(abs_errors) / total_gt)) * 100.0
    else:
        overall_acc = 100.0 if total_pred == 0 else 0.0

    mean_sample_acc = sum(sample_accs) / n
    mean_mape = sum(mapes) / n

    return CountingEvaluationResult(
        num_samples=n,
        total_ground_truth=total_gt,
        total_predicted=total_pred,
        mean_absolute_error=round(mae, 3),
        root_mean_squared_error=round(rmse, 3),
        mean_absolute_percentage_error=round(mean_mape, 2),
        overall_accuracy_pct=round(overall_acc, 2),
        mean_sample_accuracy_pct=round(mean_sample_acc, 2),
        sample_details=details,
    )


def evaluate_dataset_samples(
    weights_path: str = "runs/detect/crowd_people_detector/weights/best.pt",
    test_dir: str = "label-1/test",
    num_samples: int = 25,
    conf_threshold: float = 0.25,
    device: Optional[str] = None,
) -> CountingEvaluationResult:
    """
    Run ground-truth comparison on real labeled test samples from the dataset.
    Reads verified person bounding boxes from YOLO label files as ground truth,
    runs the trained detector, and computes counting accuracy.
    """
    test_path = Path(test_dir).resolve()
    images_dir = test_path / "images"
    labels_dir = test_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Test directory not found at {test_path}")

    model = YOLO(weights_path)
    img_files = sorted(list(images_dir.glob("*.jpg")))[:num_samples]

    gt_counts = []
    pred_counts = []
    sample_names = []

    for img_p in img_files:
        # Read ground-truth labels
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        gt_count = 0
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            # Filter class 0 (person)
            for line in lines:
                parts = line.split()
                if parts and int(parts[0]) == 0:
                    gt_count += 1

        # Run model inference
        res = model.predict(source=str(img_p), conf=conf_threshold, classes=[0], device=device, verbose=False)
        if res and len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
            boxes_np = res[0].boxes.xyxy.cpu().numpy()
            confs_np = res[0].boxes.conf.cpu().numpy()
            tids = [None] * len(boxes_np)
            from src.monitoring.tracker import suppress_duplicate_and_contained_boxes
            filt_boxes, _, _ = suppress_duplicate_and_contained_boxes(
                boxes_np, confs_np, tids, containment_threshold=0.65
            )
            pred_count = len(filt_boxes)
        else:
            pred_count = 0

        gt_counts.append(gt_count)
        pred_counts.append(pred_count)
        sample_names.append(img_p.name[:25])

    return calculate_counting_accuracy(gt_counts, pred_counts, sample_names)

