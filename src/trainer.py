"""
Model fine-tuning pipeline for local YOLO people detection.
Handles configurable hyperparameters, hardware detection, training-time augmentations,
and artifact management outside the source dataset.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from ultralytics import YOLO

from src.dataset_validator import DatasetValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")


@dataclass
class TrainingConfig:
    data: str = "label-1/data.yaml"
    model: str = "yolov8n.pt"
    epochs: int = 25
    imgsz: int = 640
    batch: int = 16
    device: Optional[str] = None
    project: str = "runs/detect"
    name: str = "crowd_people_detector"
    seed: int = 42
    workers: int = 2
    patience: int = 10
    save: bool = True
    exist_ok: bool = True
    pretrained: bool = True
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    # Training-time augmentations
    hsv_h: float = 0.015     # HSV-Hue variation
    hsv_s: float = 0.7       # HSV-Saturation variation (handles B&W CCTV vs color footage)
    hsv_v: float = 0.4       # HSV-Value/brightness variation (handles lighting changes)
    degrees: float = 5.0     # Small rotation for surveillance camera angles
    translate: float = 0.1   # Camera panning/translation
    scale: float = 0.5       # Scale gain (handles crowds far and near)
    shear: float = 2.0       # Shear angle
    perspective: float = 0.0 # Perspective distortion
    flipud: float = 0.0      # Vertical flip (people remain upright)
    fliplr: float = 0.5      # Horizontal flip (people walking left/right)
    mosaic: float = 1.0      # Mosaic augmentation for multi-person scenes
    close_mosaic: int = 5    # Disable mosaic in final epochs for stable convergence


class PeopleDetectorTrainer:
    """Orchestrates validation and fine-tuning of YOLO models for crowd monitoring."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._resolve_device()

    def _resolve_device(self):
        if self.config.device is None:
            if torch.cuda.is_available():
                self.config.device = "0"
                logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}. Using device='0'.")
            else:
                self.config.device = "cpu"
                logger.info("No CUDA GPU detected. Using device='cpu'.")
        else:
            logger.info(f"Using explicitly configured device='{self.config.device}'.")

    def validate_dataset(self) -> None:
        """Run pre-training integrity checks on dataset before training."""
        logger.info(f"Checking dataset configuration at: {self.config.data}")
        validator = DatasetValidator(self.config.data, expected_class="person")
        result = validator.validate(check_image_readability=True)
        print(result.summary())

        if not result.is_valid:
            error_details = "\n - ".join(result.errors)
            raise ValueError(f"Dataset validation failed with {len(result.errors)} errors:\n - {error_details}")

        if result.session_leakage_found:
            logger.warning("Potential recording session overlap found between splits.")

        logger.info("Dataset validation passed successfully.")

    def train(self) -> Dict[str, Any]:
        """Validate dataset and run fine-tuning."""
        # 1. Pre-training verification
        self.validate_dataset()

        # 2. Initialize pretrained model
        logger.info(f"Loading base model: {self.config.model}")
        model = YOLO(self.config.model)

        # 3. Prepare output directory
        project_dir = Path(self.config.project).resolve()
        project_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        logger.info(
            f"Starting training: {self.config.epochs} epochs, imgsz={self.config.imgsz}, "
            f"batch={self.config.batch}, device={self.config.device}"
        )

        train_args = {
            "data": str(Path(self.config.data).resolve()),
            "epochs": self.config.epochs,
            "imgsz": self.config.imgsz,
            "batch": self.config.batch,
            "device": self.config.device,
            "project": str(project_dir),
            "name": self.config.name,
            "seed": self.config.seed,
            "workers": self.config.workers,
            "patience": self.config.patience,
            "save": self.config.save,
            "exist_ok": self.config.exist_ok,
            "pretrained": self.config.pretrained,
            "optimizer": self.config.optimizer,
            "lr0": self.config.lr0,
            "lrf": self.config.lrf,
            # Augmentations
            "hsv_h": self.config.hsv_h,
            "hsv_s": self.config.hsv_s,
            "hsv_v": self.config.hsv_v,
            "degrees": self.config.degrees,
            "translate": self.config.translate,
            "scale": self.config.scale,
            "shear": self.config.shear,
            "perspective": self.config.perspective,
            "flipud": self.config.flipud,
            "fliplr": self.config.fliplr,
            "mosaic": self.config.mosaic,
            "close_mosaic": self.config.close_mosaic,
            "verbose": True
        }

        # Ultralytics train call
        results = model.train(**train_args)
        elapsed_sec = time.time() - start_time

        output_dir = project_dir / self.config.name
        best_weights = output_dir / "weights" / "best.pt"
        last_weights = output_dir / "weights" / "last.pt"

        # Extract validation metrics achieved on validation split during training
        val_map50 = getattr(results.box, "map50", 0.0) if hasattr(results, "box") else 0.0
        val_map = getattr(results.box, "map", 0.0) if hasattr(results, "box") else 0.0
        val_p = getattr(results.box, "mp", 0.0) if hasattr(results, "box") else 0.0
        val_r = getattr(results.box, "mr", 0.0) if hasattr(results, "box") else 0.0

        summary = {
            "status": "COMPLETED",
            "model_type": self.config.model,
            "epochs_trained": self.config.epochs,
            "image_size": self.config.imgsz,
            "batch_size": self.config.batch,
            "device_used": self.config.device,
            "training_duration_seconds": round(elapsed_sec, 2),
            "best_weights_path": str(best_weights),
            "last_weights_path": str(last_weights),
            "output_directory": str(output_dir),
            "val_metrics": {
                "precision": round(float(val_p), 4),
                "recall": round(float(val_r), 4),
                "mAP50": round(float(val_map50), 4),
                "mAP50_95": round(float(val_map), 4)
            },
            "config": asdict(self.config)
        }

        summary_file = output_dir / "training_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training completed in {elapsed_sec:.1f}s.")
        logger.info(f"Best weights saved at: {best_weights}")
        logger.info(f"Summary report written to: {summary_file}")

        return summary

