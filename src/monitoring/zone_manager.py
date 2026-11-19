"""
Crowd density management and monitoring module.
Calculates frame-level crowd density based on total bounding boxes per frame,
supports distinct Low, Moderate, High, Critical thresholds, sustained overcrowding timers,
and optional custom rectangular monitoring zones.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class DensityStatus(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# BGR Color map for status
STATUS_COLORS = {
    DensityStatus.LOW: (0, 200, 0),         # Vibrant Green
    DensityStatus.MODERATE: (0, 215, 255),  # Amber / Yellow
    DensityStatus.HIGH: (0, 140, 255),      # Orange
    DensityStatus.CRITICAL: (0, 0, 255),    # Red
}

STATUS_BG_HEX = {
    DensityStatus.LOW: "#28a745",
    DensityStatus.MODERATE: "#ffc107",
    DensityStatus.HIGH: "#fd7e14",
    DensityStatus.CRITICAL: "#dc3545",
}


@dataclass
class FrameCrowdThresholds:
    """Configurable thresholds for overall frame crowd density based on total bounding boxes."""
    low: int = 5         # <= 5: Low / Normal
    moderate: int = 12   # 6 - 12: Moderate
    high: int = 18       # 13 - 18: High / Crowded
    critical: int = 22   # >= 22: Critical / Overcrowded
    sustained_seconds: float = 3.0


class FrameCrowdMonitor:
    """Monitors whole-frame crowd density from total detected people per frame."""

    def __init__(self, thresholds: Optional[FrameCrowdThresholds] = None):
        self.thresholds = thresholds or FrameCrowdThresholds()
        self.current_count: int = 0
        self.status: DensityStatus = DensityStatus.LOW
        self.overcrowded_since: Optional[float] = None
        self.in_active_incident: bool = False
        self.active_incident_id: Optional[str] = None
        self.total_incidents_triggered: int = 0

    def reset(self) -> None:
        self.current_count = 0
        self.status = DensityStatus.LOW
        self.overcrowded_since = None
        self.in_active_incident = False
        self.active_incident_id = None

    def update(self, total_people_count: int, now: Optional[float] = None) -> Tuple[DensityStatus, bool, float]:
        """
        Update with current frame people count and evaluate sustained overcrowding.

        Returns:
            Tuple of (status: DensityStatus, should_trigger_incident: bool, duration_at_critical: float)
        """
        if now is None:
            now = time.time()

        self.current_count = total_people_count

        # Density status evaluation
        if self.current_count <= self.thresholds.low:
            self.status = DensityStatus.LOW
        elif self.current_count <= self.thresholds.moderate:
            self.status = DensityStatus.MODERATE
        elif self.current_count < self.thresholds.critical:
            self.status = DensityStatus.HIGH
        else:
            self.status = DensityStatus.CRITICAL

        # Sustained overcrowding check
        if self.current_count >= self.thresholds.critical:
            if self.overcrowded_since is None:
                self.overcrowded_since = now
            duration = now - self.overcrowded_since

            if not self.in_active_incident and duration >= self.thresholds.sustained_seconds:
                self.in_active_incident = True
                self.total_incidents_triggered += 1
                return self.status, True, duration

            return self.status, False, duration
        else:
            # Drop below critical threshold -> reset overcrowding timer and active incident flag
            self.overcrowded_since = None
            self.in_active_incident = False
            self.active_incident_id = None
            return self.status, False, 0.0


@dataclass
class ZoneThresholds:
    low: int = 2
    moderate: int = 5
    high: int = 8
    critical: int = 10
    sustained_seconds: float = 3.0


@dataclass
class MonitoringZone:
    id: str
    name: str
    x_min_pct: float  # 0.0 to 1.0 (relative to frame width)
    y_min_pct: float  # 0.0 to 1.0 (relative to frame height)
    x_max_pct: float  # 0.0 to 1.0
    y_max_pct: float  # 0.0 to 1.0
    thresholds: ZoneThresholds = field(default_factory=ZoneThresholds)
    current_count: int = 0
    status: DensityStatus = DensityStatus.LOW

    # Sustained overcrowding tracking state
    overcrowded_since: Optional[float] = None
    in_active_incident: bool = False
    active_incident_id: Optional[str] = None
    total_incidents_triggered: int = 0

    def get_pixel_coords(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Convert normalized percentages to pixel bounds (x1, y1, x2, y2)."""
        x1 = max(0, min(frame_w - 1, int(self.x_min_pct * frame_w)))
        y1 = max(0, min(frame_h - 1, int(self.y_min_pct * frame_h)))
        x2 = max(x1 + 1, min(frame_w, int(self.x_max_pct * frame_w)))
        y2 = max(y1 + 1, min(frame_h, int(self.y_max_pct * frame_h)))
        return x1, y1, x2, y2

    def contains_point(self, pt: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        """Check if bottom-center point (x, y) falls inside this zone."""
        x, y = pt
        x1, y1, x2, y2 = self.get_pixel_coords(frame_w, frame_h)
        return x1 <= x <= x2 and y1 <= y <= y2

    def update_status(self) -> DensityStatus:
        """Evaluate density status based on current count and distinct thresholds."""
        cnt = self.current_count
        if cnt <= self.thresholds.low:
            self.status = DensityStatus.LOW
        elif cnt <= self.thresholds.moderate:
            self.status = DensityStatus.MODERATE
        elif cnt < self.thresholds.critical:
            self.status = DensityStatus.HIGH
        else:
            self.status = DensityStatus.CRITICAL

        if cnt < self.thresholds.critical:
            self.overcrowded_since = None
            self.in_active_incident = False
            self.active_incident_id = None

        return self.status

    def check_sustained_overcrowding(self, now: Optional[float] = None) -> Tuple[bool, float]:
        if now is None:
            now = time.time()

        if self.current_count >= self.thresholds.critical:
            if self.overcrowded_since is None:
                self.overcrowded_since = now
            duration = now - self.overcrowded_since

            if not self.in_active_incident and duration >= self.thresholds.sustained_seconds:
                self.in_active_incident = True
                self.total_incidents_triggered += 1
                return True, duration
            return False, duration
        else:
            self.overcrowded_since = None
            self.in_active_incident = False
            self.active_incident_id = None
            return False, 0.0


class ZoneManager:
    """Manages collection of optional monitoring zones and computes real-time crowd metrics."""

    def __init__(self, initial_zones: Optional[List[MonitoringZone]] = None):
        self.zones: Dict[str, MonitoringZone] = {}
        if initial_zones:
            for z in initial_zones:
                self.zones[z.id] = z
        # By default: no forced Zone A or Zone B! Clean screen.

    def add_zone(self, zone: MonitoringZone) -> None:
        self.zones[zone.id] = zone

    def remove_zone(self, zone_id: str) -> None:
        if zone_id in self.zones:
            del self.zones[zone_id]

    def reset_counts(self) -> None:
        for z in self.zones.values():
            z.current_count = 0
            z.status = DensityStatus.LOW
            z.overcrowded_since = None
            z.in_active_incident = False
            z.active_incident_id = None

    def update_detections(
        self,
        bottom_centers: List[Tuple[int, int]],
        frame_w: int,
        frame_h: int,
    ) -> Dict[str, int]:
        """Count person detections in each custom zone using their bottom-center footprints."""
        counts = {z_id: 0 for z_id in self.zones.keys()}

        for pt in bottom_centers:
            for z_id, zone in self.zones.items():
                if zone.contains_point(pt, frame_w, frame_h):
                    counts[z_id] += 1

        for z_id, cnt in counts.items():
            self.zones[z_id].current_count = cnt
            self.zones[z_id].update_status()

        return counts

    def evaluate_incidents(self, now: Optional[float] = None) -> List[Tuple[MonitoringZone, float]]:
        if now is None:
            now = time.time()
        triggered = []
        for zone in self.zones.values():
            should_trigger, duration = zone.check_sustained_overcrowding(now)
            if should_trigger:
                triggered.append((zone, duration))
        return triggered

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.18) -> np.ndarray:
        """Draw optional zone overlays only if custom zones are configured."""
        if not self.zones:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()
        now = time.time()

        for zone in self.zones.values():
            x1, y1, x2, y2 = zone.get_pixel_coords(w, h)
            color = STATUS_COLORS.get(zone.status, (0, 200, 0))

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            border_thickness = 3 if zone.status == DensityStatus.CRITICAL else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, border_thickness)

            badge_text = f"{zone.name} | Count: {zone.current_count} [{zone.status.value}]"
            if zone.overcrowded_since is not None and zone.current_count >= zone.thresholds.critical:
                elapsed = now - zone.overcrowded_since
                badge_text += f" (CRITICAL: {elapsed:.1f}s / {zone.thresholds.sustained_seconds:.0f}s)"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            badge_x2 = min(w, x1 + tw + 10)

            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                frame,
                badge_text,
                (x1 + 5, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame
