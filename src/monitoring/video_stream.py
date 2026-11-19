"""
Robust video capture and stream ingestion module.
Supports Webcams, uploaded video files, CCTV/RTSP streams, and synthetic/dataset feeds
with comprehensive error handling and graceful stream disconnect recovery.
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


class StreamType(str, Enum):
    WEBCAM = "Webcam"
    VIDEO_FILE = "Uploaded Video"
    RTSP = "CCTV / RTSP Stream"
    DEMO = "Demo Dataset Sequence"


class VideoStreamSource:
    """Manages video frame ingestion with validation and clean error reporting."""

    def __init__(
        self,
        stream_type: StreamType,
        source: Optional[str] = None,
        cam_index: int = 0,
        loop: bool = True,
    ):
        self.stream_type = stream_type
        self.source = source
        self.cam_index = cam_index
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30.0

        # For Demo mode
        self.demo_images: List[Path] = []
        self.demo_index: int = 0

    def open(self) -> Tuple[bool, str]:
        """
        Attempt to open the stream.
        Returns:
            Tuple of (success: bool, message: str)
        """
        self.release()

        if self.stream_type == StreamType.DEMO:
            test_dir = Path("label-1/test/images")
            if not test_dir.exists():
                return False, f"Demo dataset directory not found at '{test_dir.resolve()}'."
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            self.demo_images = sorted(
                [f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            )
            if not self.demo_images:
                return False, "No test images found in demo directory."
            self.demo_index = 0
            # Read first image to obtain resolution
            first = cv2.imread(str(self.demo_images[0]))
            if first is None:
                return False, "Failed to read demo test images."
            self.frame_height, self.frame_width = first.shape[:2]
            self.fps = 10.0
            self.is_opened = True
            return True, f"Loaded demo dataset sequence ({len(self.demo_images)} frames)."

        elif self.stream_type == StreamType.WEBCAM:
            try:
                # DirectShow backend on Windows is fast and reliable
                cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.cam_index)

                if not cap.isOpened():
                    return (
                        False,
                        f"Cannot open Webcam index {self.cam_index}.\n"
                        f"The camera may be disconnected, in use by another app, or blocked by Windows privacy settings.",
                    )

                # Test read
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    cap.release()
                    return (
                        False,
                        f"Camera at index {self.cam_index} opened but returned no frames.\n"
                        f"Check if camera shutter is open or permissions are granted in Windows Settings -> Camera.",
                    )

                self.cap = cap
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cam_fps = cap.get(cv2.CAP_PROP_FPS)
                self.fps = cam_fps if cam_fps > 0 else 30.0
                self.is_opened = True
                return True, f"Webcam {self.cam_index} connected ({self.frame_width}x{self.frame_height} @ {self.fps:.0f} FPS)."

            except Exception as e:
                return False, f"Webcam initialization error: {e}"

        elif self.stream_type == StreamType.VIDEO_FILE:
            if not self.source:
                return False, "No video file specified."
            video_path = Path(self.source)
            if not video_path.exists():
                return False, f"Video file not found at: {video_path.resolve()}"

            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return (
                        False,
                        f"Cannot open video file '{video_path.name}'. File may be corrupt or in an unsupported format.",
                    )

                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    cap.release()
                    return False, f"Video file '{video_path.name}' contains unreadable video frames."

                # Reset to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.cap = cap
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_fps = cap.get(cv2.CAP_PROP_FPS)
                self.fps = vid_fps if vid_fps > 0 else 30.0
                self.is_opened = True
                return True, f"Video '{video_path.name}' loaded ({self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS)."

            except Exception as e:
                return False, f"Error opening video file: {e}"

        elif self.stream_type == StreamType.RTSP:
            if not self.source or not self.source.strip():
                return False, "RTSP / CCTV stream URL cannot be empty."

            url = self.source.strip()
            try:
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    return (
                        False,
                        f"Cannot connect to RTSP/CCTV stream at '{url}'.\n"
                        f"Please verify:\n"
                        f"  1. Network connectivity and camera power.\n"
                        f"  2. Correct RTSP URL format (e.g. rtsp://user:pass@ip:port/stream).\n"
                        f"  3. Camera credentials and firewall/port access.",
                    )

                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    cap.release()
                    return (
                        False,
                        f"Connected to stream URL at '{url}', but stream timed out waiting for first frame.",
                    )

                self.cap = cap
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                rtsp_fps = cap.get(cv2.CAP_PROP_FPS)
                self.fps = rtsp_fps if rtsp_fps > 0 else 25.0
                self.is_opened = True
                return True, f"RTSP stream connected ({self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS)."

            except Exception as e:
                return False, f"RTSP connection error: {e}"

        return False, f"Unsupported stream type: {self.stream_type}"

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], bool]:
        """
        Read the next frame from the stream.
        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray], is_eof: bool)
        """
        if not self.is_opened:
            return False, None, True

        if self.stream_type == StreamType.DEMO:
            if not self.demo_images:
                return False, None, True

            img_path = self.demo_images[self.demo_index]
            frame = cv2.imread(str(img_path))
            self.demo_index += 1

            is_eof = False
            if self.demo_index >= len(self.demo_images):
                if self.loop:
                    self.demo_index = 0
                else:
                    is_eof = True

            if frame is None:
                return False, None, is_eof

            return True, frame, is_eof

        if self.cap is None:
            return False, None, True

        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.stream_type == StreamType.VIDEO_FILE and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame, False
            return False, None, True

        return True, frame, False

    def release(self) -> None:
        """Release any opened capture resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_opened = False
        self.demo_images = []
        self.demo_index = 0

