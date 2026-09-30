"""
services/detect_track/models.py
================================
IBVAP P1 — Shared data-model types for detection and tracking.

These are the ONLY structs passed between analytics modules.
If you need to add a field, add it here and update tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── COCO class mapping (subset used by IBVAP) ─────────────────────────────────
COCO_CLASSES: dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Class IDs the detector will report (others filtered out)
DETECT_CLASS_IDS: list[int] = sorted(COCO_CLASSES.keys())

# Per-class BGR colours for the renderer
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (0,   200,  50),   # person     — green
    1:  (200, 100,   0),   # bicycle    — blue-ish
    2:  (0,   120, 255),   # car        — orange
    3:  (255,   0, 200),   # motorcycle — pink
    5:  (0,   200, 200),   # bus        — cyan
    7:  (50,   50, 255),   # truck      — red
}
DEFAULT_COLOR = (180, 180, 180)


# ── Raw detection (pre-tracker) ────────────────────────────────────────────────

@dataclass(slots=True)
class Detection:
    """
    Single object detection output from the YOLO model.
    bbox is in pixel coordinates (x1, y1, x2, y2).
    """
    bbox:       tuple[float, float, float, float]   # x1 y1 x2 y2 pixels
    conf:       float
    class_id:   int
    class_name: str

    @property
    def width(self)  -> float: return self.bbox[2] - self.bbox[0]
    @property
    def height(self) -> float: return self.bbox[3] - self.bbox[1]
    @property
    def area(self)   -> float: return self.width * self.height
    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    def to_dict(self) -> dict:
        return {
            "bbox":       list(self.bbox),
            "conf":       round(self.conf, 4),
            "class_id":   self.class_id,
            "class_name": self.class_name,
        }


# ── Tracked object (post-tracker) ─────────────────────────────────────────────

@dataclass
class Track:
    """
    A detection associated with a persistent track ID by ByteTrack.
    This is the primary data type consumed by ALL downstream analytics.
    """
    track_id:   int
    bbox:       tuple[float, float, float, float]   # x1 y1 x2 y2 pixels
    conf:       float
    class_id:   int
    class_name: str
    frame_id:   int
    camera_id:  str
    timestamp:  float = field(default_factory=time.monotonic)

    # Optional — filled in by behaviour module
    velocity:   Optional[tuple[float, float]] = None  # (vx, vy) px/s
    dwell_s:    float = 0.0                           # seconds in current zone

    @property
    def width(self)  -> float: return self.bbox[2] - self.bbox[0]
    @property
    def height(self) -> float: return self.bbox[3] - self.bbox[1]
    @property
    def area(self)   -> float: return self.width * self.height
    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)
    @property
    def color(self) -> tuple[int, int, int]:
        return CLASS_COLORS.get(self.class_id, DEFAULT_COLOR)

    def to_dict(self) -> dict:
        return {
            "track_id":   self.track_id,
            "bbox":       [round(v, 1) for v in self.bbox],
            "conf":       round(self.conf, 4),
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "frame_id":   self.frame_id,
            "camera_id":  self.camera_id,
            "timestamp":  round(self.timestamp, 6),
            "center":     [round(v, 1) for v in self.center],
        }
