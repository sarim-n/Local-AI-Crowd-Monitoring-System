"""
Dataset validation and integrity checking module for YOLO people detection.
Verifies dataset configuration, file integrity, annotations, and split distributions.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml
from PIL import Image


@dataclass
class ValidationResult:
    is_valid: bool
    yaml_path: str
    classes: List[str]
    split_counts: Dict[str, int] = field(default_factory=dict)
    annotation_counts: Dict[str, int] = field(default_factory=dict)
    background_counts: Dict[str, int] = field(default_factory=dict)
    total_images: int = 0
    total_annotations: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    session_leakage_found: bool = False

    def summary(self) -> str:
        status_str = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"================ DATASET VALIDATION REPORT [{status_str}] ================",
            f"Config File: {self.yaml_path}",
            f"Classes: {self.classes}",
            f"Total Images: {self.total_images}",
            f"Total Bounding Boxes: {self.total_annotations}",
            "",
            "--- Split Breakdown ---"
        ]
        for split, count in self.split_counts.items():
            pct = (count / self.total_images * 100) if self.total_images > 0 else 0
            boxes = self.annotation_counts.get(split, 0)
            bg = self.background_counts.get(split, 0)
            lines.append(
                f"  {split:<8}: {count:>4} images ({pct:>5.1f}%) | "
                f"{boxes:>5} annotations | {bg:>3} background (empty) images"
            )

        if self.errors:
            lines.append("\n--- ERRORS ---")
            for err in self.errors:
                lines.append(f"  [ERROR] {err}")

        if self.warnings:
            lines.append("\n--- WARNINGS ---")
            for w in self.warnings:
                lines.append(f"  [WARNING] {w}")

        lines.append("=================================================================")
        return "\n".join(lines)


def resolve_split_path(yaml_file: Path, split_path_str: str, root_path_str: Optional[str] = None) -> Path:
    """Resolve a split path relative to data.yaml directory, root path, or Roboflow directory."""
    p = Path(split_path_str)
    if p.is_absolute() and p.exists():
        return p

    # 1. If explicit 'path' is provided in yaml:
    if root_path_str:
        root_path = Path(root_path_str)
        if not root_path.is_absolute():
            root_path = (yaml_file.parent / root_path).resolve()
        candidate = (root_path / p).resolve()
        if candidate.exists():
            return candidate

    # 2. Directly relative to yaml directory:
    candidate = (yaml_file.parent / p).resolve()
    if candidate.exists():
        return candidate

    # 3. Strip leading '../' or '..\\' (common Roboflow export quirk) relative to yaml directory:
    cleaned = split_path_str.replace("../", "").replace("..\\", "")
    candidate = (yaml_file.parent / cleaned).resolve()
    if candidate.exists():
        return candidate

    # 4. Relative to current working directory:
    candidate = Path(split_path_str).resolve()
    if candidate.exists():
        return candidate

    # 5. Relative to yaml parent's parent:
    candidate = (yaml_file.parent.parent / p).resolve()
    if candidate.exists():
        return candidate

    return (yaml_file.parent / cleaned).resolve()


def extract_base_identity(filename: str) -> str:
    """
    Extract base identifier by removing Roboflow hash suffixes (.rf.<hex>.ext)
    and common file extensions.
    """
    clean = re.sub(r'\.rf\.[0-9a-fA-F]+\.(jpg|jpeg|png|bmp|webp)', '', filename, flags=re.IGNORECASE)
    clean = re.sub(r'_(jpg|jpeg|png|bmp|webp)$', '', clean, flags=re.IGNORECASE)
    return clean


class DatasetValidator:
    """Validates YOLO object detection datasets for crowd monitoring."""

    def __init__(self, data_yaml_path: str, expected_class: str = "person"):
        self.yaml_path = Path(data_yaml_path).resolve()
        self.expected_class = expected_class

    def validate(self, check_image_readability: bool = True) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if not self.yaml_path.exists():
            return ValidationResult(
                is_valid=False,
                yaml_path=str(self.yaml_path),
                classes=[],
                errors=[f"Dataset config file not found: {self.yaml_path}"]
            )

        # Parse YAML
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                yaml_path=str(self.yaml_path),
                classes=[],
                errors=[f"Failed to parse YAML file: {e}"]
            )

        # Check classes
        raw_names = data_cfg.get("names", [])
        if isinstance(raw_names, dict):
            classes = [raw_names[k] for k in sorted(raw_names.keys())]
        elif isinstance(raw_names, list):
            classes = raw_names
        else:
            classes = []

        if not classes:
            errors.append("No class names found in data.yaml ('names' field).")
        elif self.expected_class and self.expected_class not in classes:
            warnings.append(
                f"Expected class '{self.expected_class}' not found in dataset classes: {classes}"
            )

        nc = data_cfg.get("nc")
        if nc is not None and nc != len(classes):
            warnings.append(f"'nc' ({nc}) does not match number of classes in 'names' ({len(classes)}).")

        # Resolve splits
        splits_to_check = {}
        root_path_str = data_cfg.get("path")
        for split_key in ["train", "val", "test"]:
            # YOLO accepts 'val' or 'valid'
            val_path_str = data_cfg.get(split_key) or data_cfg.get("valid" if split_key == "val" else split_key)
            if val_path_str:
                resolved = resolve_split_path(self.yaml_path, val_path_str, root_path_str=root_path_str)
                if resolved.exists():
                    splits_to_check[split_key] = resolved
                else:
                    errors.append(f"Directory for split '{split_key}' not found: {resolved}")
            elif split_key in ["train", "val"]:
                errors.append(f"Required split '{split_key}' missing from data.yaml")

        if errors:
            return ValidationResult(
                is_valid=False,
                yaml_path=str(self.yaml_path),
                classes=classes,
                errors=errors,
                warnings=warnings
            )

        split_counts: Dict[str, int] = {}
        annotation_counts: Dict[str, int] = {}
        background_counts: Dict[str, int] = {}
        split_bases: Dict[str, Set[str]] = {}
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        total_images = 0
        total_annotations = 0

        for split_name, img_dir in splits_to_check.items():
            label_dir = img_dir.parent / "labels"
            if not label_dir.exists():
                # Check sibling directory
                candidate_lbl = img_dir.parent.parent / "labels" / split_name
                if candidate_lbl.exists():
                    label_dir = candidate_lbl
                else:
                    errors.append(f"Labels directory not found for '{split_name}' at {label_dir}")
                    continue

            image_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
            split_counts[split_name] = len(image_files)
            total_images += len(image_files)

            bases_in_split = set()
            split_box_count = 0
            split_bg_count = 0

            for img_file in image_files:
                base_id = extract_base_identity(img_file.name)
                bases_in_split.add(base_id)

                # Optional readability check
                if check_image_readability:
                    try:
                        with Image.open(img_file) as im:
                            im.verify()
                    except Exception as e:
                        errors.append(f"Corrupt or unreadable image: {img_file.name} ({e})")
                        continue

                # Check corresponding label file
                lbl_file = label_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    warnings.append(f"Missing label file for image: {img_file.name}")
                    continue

                try:
                    with open(lbl_file, "r", encoding="utf-8") as lf:
                        lines = [line.strip() for line in lf if line.strip()]
                except Exception as e:
                    errors.append(f"Error reading label file {lbl_file.name}: {e}")
                    continue

                if not lines:
                    split_bg_count += 1
                    continue

                for line_idx, line in enumerate(lines, 1):
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(
                            f"Invalid YOLO label format in {lbl_file.name} line {line_idx}: expected 5 tokens, got {len(parts)}"
                        )
                        continue

                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = (float(v) for v in parts[1:])
                    except ValueError:
                        errors.append(f"Non-numeric values in {lbl_file.name} line {line_idx}: '{line}'")
                        continue

                    if cls_id < 0 or cls_id >= len(classes):
                        errors.append(
                            f"Class ID {cls_id} out of bounds [0, {len(classes)-1}] in {lbl_file.name}"
                        )

                    # Normalized bounds check
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                        errors.append(
                            f"Center coords out of bounds [0, 1] in {lbl_file.name}: ({xc}, {yc})"
                        )
                    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        errors.append(
                            f"Box dimension out of bounds (0, 1] in {lbl_file.name}: w={w}, h={h}"
                        )

                    split_box_count += 1
                    total_annotations += 1

            annotation_counts[split_name] = split_box_count
            background_counts[split_name] = split_bg_count
            split_bases[split_name] = bases_in_split

        # Split Leakage & Ratio Checks
        session_leakage = False
        all_splits = list(split_bases.keys())
        for i in range(len(all_splits)):
            for j in range(i + 1, len(all_splits)):
                s1, s2 = all_splits[i], all_splits[j]
                overlap = split_bases[s1].intersection(split_bases[s2])
                if overlap:
                    session_leakage = True
                    sample = list(overlap)[:5]
                    errors.append(
                        f"Data leakage detected between '{s1}' and '{s2}': {len(overlap)} matching base images. Sample: {sample}"
                    )

        # Proportions check
        if total_images > 0 and "train" in split_counts:
            train_pct = split_counts["train"] / total_images * 100
            if train_pct < 50.0:
                warnings.append(f"Train split is unusually small: {train_pct:.1f}% of dataset.")
            elif train_pct > 90.0:
                warnings.append(f"Train split is unusually large: {train_pct:.1f}% of dataset.")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            yaml_path=str(self.yaml_path),
            classes=classes,
            split_counts=split_counts,
            annotation_counts=annotation_counts,
            background_counts=background_counts,
            total_images=total_images,
            total_annotations=total_annotations,
            errors=errors,
            warnings=warnings,
            session_leakage_found=session_leakage
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate YOLO crowd detection dataset")
    parser.add_argument("--data", default="label-1/data.yaml", help="Path to data.yaml")
    parser.add_argument("--class-name", default="person", help="Expected detection class name")
    args = parser.parse_args()

    validator = DatasetValidator(args.data, expected_class=args.class_name)
    res = validator.validate()
    print(res.summary())
    exit(0 if res.is_valid else 1)
