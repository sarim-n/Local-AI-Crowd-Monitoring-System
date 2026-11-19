"""
Model evaluation module for the local YOLO people detector.
Evaluates the fine-tuned model on held-out test sets and reports
Precision, Recall, mAP50, and mAP50-95 with saved reports.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluator")


@dataclass
class EvaluationConfig:
    weights: str = "runs/detect/crowd_people_detector/weights/best.pt"
    data: str = "label-1/data.yaml"
    split: str = "test"
    imgsz: int = 640
    batch: int = 16
    device: Optional[str] = None
    project: str = "runs/detect"
    name: str = "evaluation"
    save_json: bool = True
    save_hybrid: bool = False
    conf: float = 0.25
    iou: float = 0.6


class PeopleDetectorEvaluator:
    """Evaluates YOLO people detection models on designated splits."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._resolve_device()

    def _resolve_device(self):
        if self.config.device is None:
            if torch.cuda.is_available():
                self.config.device = "0"
            else:
                self.config.device = "cpu"

    def evaluate(self) -> Dict[str, Any]:
        weights_path = Path(self.config.weights).resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        data_path = Path(self.config.data).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset config not found at: {data_path}")

        logger.info(f"Loading weights: {weights_path}")
        model = YOLO(str(weights_path))

        eval_dir = Path(self.config.project).resolve() / self.config.name
        eval_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Evaluating on '{self.config.split}' split with imgsz={self.config.imgsz}, "
            f"batch={self.config.batch}, device={self.config.device}"
        )

        metrics = model.val(
            data=str(data_path),
            split=self.config.split,
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            device=self.config.device,
            project=str(Path(self.config.project).resolve()),
            name=self.config.name,
            exist_ok=True,
            conf=self.config.conf,
            iou=self.config.iou,
            verbose=True
        )

        # Extract primary metrics
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        fitness = float(metrics.fitness)

        # Per-class metrics
        class_metrics = {}
        for idx, cls_name in enumerate(metrics.names.values()):
            class_metrics[cls_name] = {
                "precision": round(float(metrics.box.p[idx]), 4) if len(metrics.box.p) > idx else round(precision, 4),
                "recall": round(float(metrics.box.r[idx]), 4) if len(metrics.box.r) > idx else round(recall, 4),
                "mAP50": round(float(metrics.box.ap50[idx]), 4) if len(metrics.box.ap50) > idx else round(map50, 4),
                "mAP50_95": round(float(metrics.box.ap[idx]), 4) if len(metrics.box.ap) > idx else round(map50_95, 4),
            }

        speed_ms = {k: round(v, 2) for k, v in metrics.speed.items()} if hasattr(metrics, "speed") else {}

        report_data = {
            "model_weights": str(weights_path),
            "dataset_config": str(data_path),
            "split_evaluated": self.config.split,
            "image_size": self.config.imgsz,
            "device": self.config.device,
            "overall_metrics": {
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
                "Fitness": round(fitness, 4)
            },
            "class_metrics": class_metrics,
            "latency_ms": speed_ms,
            "output_directory": str(eval_dir)
        }

        # Save JSON report
        json_report_path = eval_dir / f"evaluation_report_{self.config.split}.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Save text summary
        txt_report_path = eval_dir / f"evaluation_report_{self.config.split}.txt"
        summary_text = self._format_summary(report_data)
        with open(txt_report_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print("\n" + summary_text)
        logger.info(f"Evaluation report saved to: {json_report_path}")
        return report_data

    def _format_summary(self, data: Dict[str, Any]) -> str:
        lines = [
            "================ EVALUATION RESULTS REPORT ================",
            f"Weights: {data['model_weights']}",
            f"Dataset: {data['dataset_config']}",
            f"Split  : {data['split_evaluated']}",
            f"Device : {data['device']}",
            "",
            "--- Summary Detection Metrics ---",
            f"  Precision (P)   : {data['overall_metrics']['Precision']:.4f} ({data['overall_metrics']['Precision']*100:.2f}%)",
            f"  Recall (R)      : {data['overall_metrics']['Recall']:.4f} ({data['overall_metrics']['Recall']*100:.2f}%)",
            f"  mAP @ 0.50      : {data['overall_metrics']['mAP50']:.4f} ({data['overall_metrics']['mAP50']*100:.2f}%)",
            f"  mAP @ 0.50:0.95 : {data['overall_metrics']['mAP50_95']:.4f} ({data['overall_metrics']['mAP50_95']*100:.2f}%)",
            ""
        ]
        if data.get("latency_ms"):
            lines.append("--- Inference Latency (per image) ---")
            for step, ms in data["latency_ms"].items():
                lines.append(f"  {step:<12}: {ms:.2f} ms")
            total_time = sum(data["latency_ms"].values())
            fps = 1000.0 / total_time if total_time > 0 else 0
            lines.append(f"  Total Latency: {total_time:.2f} ms (~{fps:.1f} FPS)")
        lines.append("===========================================================")
        return "\n".join(lines)

