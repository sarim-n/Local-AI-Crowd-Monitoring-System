"""
Object detection and tracking engine for people monitoring using YOLO and ByteTrack.
Extracts bounding boxes, tracking IDs, and bottom-center footprint coordinates.
Includes containment and duplicate box suppression to eliminate redundant sub-boxes (e.g. face/torso).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    bottom_center: Tuple[int, int]  # (x_bottom, y_bottom)
    confidence: float
    track_id: Optional[int]
    class_id: int = 0
    class_name: str = "person"


def is_same_person_duplicate(
    b_big: np.ndarray,
    b_small: np.ndarray,
    conf_big: float = 1.0,
    conf_small: float = 1.0,
    ios_threshold: float = 0.65,
) -> bool:
    """
    Carefully determines whether b_small is a redundant duplicate detection
    (face, upper torso, lower body, or shifted full-body) belonging to the
    EXACT SAME PERSON in b_big, while strictly preserving distinct individuals.

    Args:
        b_big: [x1, y1, x2, y2] coordinates of the larger box (e.g. full body).
        b_small: [x1, y1, x2, y2] coordinates of the smaller candidate sub-box.
        conf_big: Confidence of the larger box.
        conf_small: Confidence of the smaller box.
        ios_threshold: Minimum Intersection-over-Smaller (containment) ratio.

    Returns:
        True if b_small is a duplicate sub-box of the same person; False if distinct.
    """
    w_big = b_big[2] - b_big[0]
    h_big = b_big[3] - b_big[1]
    w_small = b_small[2] - b_small[0]
    h_small = b_small[3] - b_small[1]

    if w_big <= 0 or h_big <= 0 or w_small <= 0 or h_small <= 0:
        return False

    # Intersection rectangle
    ix1 = max(b_big[0], b_small[0])
    iy1 = max(b_big[1], b_small[1])
    ix2 = min(b_big[2], b_small[2])
    iy2 = min(b_big[3], b_small[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_small = w_small * h_small
    area_big = w_big * h_big
    union = area_big + area_small - inter
    iou = inter / union if union > 0 else 0.0
    ios = inter / area_small if area_small > 0 else 0.0

    # Rule 1: High IoU (>= 0.48)
    # If two boxes overlap by >= 48% IoU, they occupy substantially the same space.
    if iou >= 0.48:
        return True

    # If containment is below threshold, cannot be a contained duplicate
    if ios < ios_threshold:
        return False

    cx_big = (b_big[0] + b_big[2]) / 2.0
    cx_small = (b_small[0] + b_small[2]) / 2.0
    center_x_diff = abs(cx_small - cx_big) / w_big

    top_diff = abs(b_small[1] - b_big[1]) / h_big
    bottom_diff = abs(b_small[3] - b_big[3]) / h_big

    horiz_inter = iw / w_small if w_small > 0 else 0.0
    width_ratio = w_small / w_big
    height_ratio = h_small / h_big
    aspect_small = h_small / w_small if w_small > 0 else 0.0

    # Rule 2: Upper body / Torso / Face duplicate (shares head)
    # Head aligned: top_diff <= 0.15 (allows head tilt/hair boundary variations)
    if top_diff <= 0.15 and horiz_inter >= 0.65:
        # A1: Upper-body / torso (covers upper 30% to 90% of body)
        if 0.30 <= height_ratio <= 0.90 and center_x_diff <= 0.25 and width_ratio >= 0.40:
            return True
        # A2: Face / head (covers upper <= 35% of body)
        if (
            b_small[3] <= b_big[1] + 0.38 * h_big
            and height_ratio <= 0.35
            and 0.60 <= aspect_small <= 2.0
            and center_x_diff <= 0.25
        ):
            return True

    # Rule 3: Ground / Feet Alignment (same person standing at the same spot on the floor)
    # Bottom aligned (bottom_diff <= 0.10) AND height ratio >= 0.65 (full body duplicate, not child)
    if bottom_diff <= 0.10 and height_ratio >= 0.65 and horiz_inter >= 0.65 and center_x_diff <= 0.30:
        return True

    # Rule 4: Edge of Screen boundary duplicates (entering / exiting frame)
    # Both boxes touch or share the same image boundary (e.g. left edge x1 near 0 or right edge)
    shares_left_edge = abs(b_big[0] - b_small[0]) < 15 and (b_big[0] < 20 or b_small[0] < 20)
    shares_right_edge = abs(b_big[2] - b_small[2]) < 15
    if (shares_left_edge or shares_right_edge) and (top_diff <= 0.15 or bottom_diff <= 0.10):
        if ios >= 0.70 and height_ratio >= 0.50:
            return True

    return False


def suppress_duplicate_and_contained_boxes(
    boxes: np.ndarray,
    confs: np.ndarray,
    track_ids: List[Optional[int]],
    containment_threshold: float = 0.65,
    duplicate_iou_threshold: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[int]]]:
    """
    Suppresses redundant sub-boxes where a smaller box (e.g. face or upper torso)
    belongs to the same person, while strictly preserving distinct individuals who
    are standing, sitting, or walking inside/behind a larger person's bounding box.

    Args:
        boxes: Array of shape (N, 4) in xyxy format.
        confs: Array of shape (N,) confidence scores.
        track_ids: List of track IDs for each box.
        containment_threshold: Fraction of smaller box enclosed in larger box (default 0.65).
        duplicate_iou_threshold: Standard IoU threshold for duplicate suppression (default 0.50).

    Returns:
        Filtered (boxes, confs, track_ids)
    """
    n = len(boxes)
    if n <= 1:
        return boxes, confs, track_ids

    # Compute areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue

        x1_i, y1_i, x2_i, y2_i = boxes[i]
        area_i = areas[i]

        for j in range(n):
            if i == j or not keep[j]:
                continue

            x1_j, y1_j, x2_j, y2_j = boxes[j]
            area_j = areas[j]

            # Intersection rectangle
            ix1 = max(x1_i, x1_j)
            iy1 = max(y1_i, y1_j)
            ix2 = min(x2_i, x2_j)
            iy2 = min(y2_i, y2_j)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih

            if inter <= 0:
                continue

            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0.0

            # 1. Standard duplicate suppression for near-identical boxes
            if iou > duplicate_iou_threshold:
                if confs[j] < confs[i]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

            # 2. Same-Person Sub-Box Suppression:
            elif area_j < area_i:
                if is_same_person_duplicate(
                    b_big=boxes[i],
                    b_small=boxes[j],
                    conf_big=float(confs[i]),
                    conf_small=float(confs[j]),
                    ios_threshold=containment_threshold,
                ):
                    keep[j] = False

            elif area_i < area_j:
                if is_same_person_duplicate(
                    b_big=boxes[j],
                    b_small=boxes[i],
                    conf_big=float(confs[j]),
                    conf_small=float(confs[i]),
                    ios_threshold=containment_threshold,
                ):
                    keep[i] = False
                    break

    keep_indices = [idx for idx, k in enumerate(keep) if k]
    filtered_boxes = boxes[keep_indices]
    filtered_confs = confs[keep_indices]
    filtered_track_ids = [track_ids[idx] for idx in keep_indices]

    return filtered_boxes, filtered_confs, filtered_track_ids


class CrowdDetectorTracker:
    """Manages people detection and ByteTrack tracking for crowd monitoring."""

    def __init__(
        self,
        weights_path: str = "runs/detect/crowd_people_detector/weights/best.pt",
        device: Optional[str] = None,
        tracker_type: str = "bytetrack.yaml",
    ):
        self.weights_path = Path(weights_path)
        self.tracker_type = tracker_type
        self.unique_track_ids: Set[int] = set()

        self._validate_weights()
        self.device = self._resolve_device(device)
        self.model = YOLO(str(self.weights_path))

    def _validate_weights(self) -> None:
        """Validate that model weights exist; raise informative error if missing."""
        if not self.weights_path.exists():
            resolved = self.weights_path.resolve()
            raise FileNotFoundError(
                f"Model weights not found at '{resolved}'.\n"
                f"To fix this issue:\n"
                f"  1. Verify the weights file path in the sidebar/configuration.\n"
                f"  2. Or run Phase 1 fine-tuning to generate the checkpoint:\n"
                f"     python train.py --epochs 25 --data label-1/data.yaml\n"
                f"  3. Default expected path: runs/detect/crowd_people_detector/weights/best.pt"
            )

    @staticmethod
    def _resolve_device(requested_device: Optional[str]) -> str:
        if requested_device is not None and requested_device != "auto":
            return requested_device
        return "0" if torch.cuda.is_available() else "cpu"

    def reset_session(self) -> None:
        """Reset session-level tracking state."""
        self.unique_track_ids.clear()

    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.6,
        enable_tracking: bool = True,
        draw_annotations: bool = True,
        show_track_id: bool = True,
        suppress_duplicates: bool = True,
        containment_threshold: float = 0.65,
    ) -> Tuple[List[Detection], np.ndarray]:
        """
        Run detection and tracking on a single frame.

        Args:
            frame: BGR numpy image frame.
            conf_threshold: Minimum detection confidence score.
            iou_threshold: IoU threshold for NMS / association.
            enable_tracking: Whether to enable ByteTrack tracking IDs.
            draw_annotations: Whether to draw bounding boxes and labels on output frame.
            show_track_id: Whether to show tracking IDs on labels.
            suppress_duplicates: Whether to suppress duplicate and contained sub-boxes.
            containment_threshold: Fraction of smaller box enclosed in larger box to suppress (default 0.65).

        Returns:
            Tuple of (list of Detection objects, annotated BGR frame).
        """
        annotated_frame = frame.copy()
        detections: List[Detection] = []

        if enable_tracking:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_type,
                classes=[0],  # person only
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=frame,
                classes=[0],
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False,
            )

        if not results or len(results) == 0:
            return detections, annotated_frame

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return detections, annotated_frame

        xyxy = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        track_ids = (
            boxes_obj.id.int().cpu().numpy().tolist()
            if (boxes_obj.id is not None and enable_tracking)
            else [None] * len(xyxy)
        )

        # Apply anatomical same-person duplicate suppression
        if suppress_duplicates and len(xyxy) > 1:
            xyxy, confs, track_ids = suppress_duplicate_and_contained_boxes(
                boxes=xyxy,
                confs=confs,
                track_ids=track_ids,
                containment_threshold=containment_threshold,
                duplicate_iou_threshold=0.50,
            )

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = (int(v) for v in box)
            conf = float(confs[i])
            tid = int(track_ids[i]) if track_ids[i] is not None else None

            if tid is not None:
                self.unique_track_ids.add(tid)

            # Bottom-center coordinate (footprint)
            x_bottom = int((x1 + x2) / 2)
            y_bottom = int(y2)
            bottom_center = (x_bottom, y_bottom)

            det = Detection(
                box=(x1, y1, x2, y2),
                bottom_center=bottom_center,
                confidence=conf,
                track_id=tid,
                class_id=0,
                class_name="person",
            )
            detections.append(det)

            if draw_annotations:
                box_color = (255, 178, 50)  # BGR cyan/blue

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw bottom-center footprint point
                cv2.circle(annotated_frame, (x_bottom, y_bottom), 4, (0, 0, 255), -1)

                # Build label string
                if tid is not None and show_track_id:
                    label = f"ID:{tid} person {conf:.2f}"
                else:
                    label = f"person {conf:.2f}"

                # Label background tag
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                tag_y1 = max(0, y1 - lh - 6)
                cv2.rectangle(
                    annotated_frame,
                    (x1, tag_y1),
                    (x1 + lw + 6, tag_y1 + lh + 6),
                    box_color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 3, tag_y1 + lh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return detections, annotated_frame
