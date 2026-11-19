"""
Heatmap generation module for crowd position and movement density.
Accumulates person footprint positions across the monitoring session and renders colormaps.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class CrowdHeatmap:
    """Session-level cumulative heatmap of detected people footprint locations."""

    def __init__(self, kernel_size: int = 41, sigma: float = 12.0):
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.grid: Optional[np.ndarray] = None
        self.grid_shape: Optional[Tuple[int, int]] = None  # (height, width)
        self.total_points_accumulated: int = 0

    def _init_grid_if_needed(self, height: int, width: int) -> None:
        if self.grid is None or self.grid_shape != (height, width):
            self.grid = np.zeros((height, width), dtype=np.float32)
            self.grid_shape = (height, width)

    def reset(self) -> None:
        """Clear accumulated positions for a new session."""
        if self.grid is not None:
            self.grid.fill(0.0)
        self.total_points_accumulated = 0

    def update(self, points: List[Tuple[int, int]], frame_w: int, frame_h: int) -> None:
        """
        Accumulate detected footprint locations (x_bottom, y_bottom).
        """
        if not points:
            return

        self._init_grid_if_needed(frame_h, frame_w)
        radius = self.kernel_size // 2

        # Create 2D Gaussian kernel patch
        ax = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * self.sigma**2)).astype(np.float32)

        for px, py in points:
            if not (0 <= px < frame_w and 0 <= py < frame_h):
                continue

            # Determine bounds for kernel application
            x1 = max(0, px - radius)
            y1 = max(0, py - radius)
            x2 = min(frame_w, px + radius + 1)
            y2 = min(frame_h, py + radius + 1)

            kx1 = x1 - (px - radius)
            ky1 = y1 - (py - radius)
            kx2 = kx1 + (x2 - x1)
            ky2 = ky1 + (y2 - y1)

            self.grid[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
            self.total_points_accumulated += 1

    def get_color_heatmap(
        self,
        frame_w: int,
        frame_h: int,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Generate standalone colorized heatmap image (BGR).
        """
        self._init_grid_if_needed(frame_h, frame_w)

        max_val = np.max(self.grid)
        if max_val > 0:
            # Normalized 0-255 using square-root scaling for better contrast in low-density areas
            norm = np.clip(self.grid / max_val, 0, 1)
            norm = np.sqrt(norm) * 255.0
            norm_uint8 = norm.astype(np.uint8)
        else:
            norm_uint8 = np.zeros((frame_h, frame_w), dtype=np.uint8)

        color_map = cv2.applyColorMap(norm_uint8, colormap)
        return color_map

    def blend_overlay(
        self,
        frame: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
        threshold: float = 0.05,
    ) -> np.ndarray:
        """
        Blend heatmap on top of live video frame only where people have walked.

        Args:
            frame: Original BGR frame.
            alpha: Heatmap blend strength (0.0 to 1.0).
            colormap: OpenCV colormap constant.
            threshold: Minimum relative intensity (0.0 - 1.0) to display heat.
        """
        h, w = frame.shape[:2]
        self._init_grid_if_needed(h, w)

        max_val = np.max(self.grid)
        if max_val <= 0:
            return frame.copy()

        # Normalize grid 0.0 to 1.0
        norm_grid = np.clip(self.grid / max_val, 0.0, 1.0)

        # Apply threshold mask: only blend where activity exists
        active_mask = norm_grid > threshold
        if not np.any(active_mask):
            return frame.copy()

        norm_uint8 = (np.sqrt(norm_grid) * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(norm_uint8, colormap)

        blended = frame.copy()
        # Pixel-wise alpha blending for active areas
        mask_3d = np.repeat(active_mask[:, :, np.newaxis], 3, axis=2)
        np.copyto(
            blended,
            cv2.addWeighted(frame, 1.0 - alpha, heat_bgr, alpha, 0),
            where=mask_3d,
        )

        return blended

