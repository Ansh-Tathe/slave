"""
services/analytics/anpr/plate_detector.py
==========================================
IBVAP P3 — License plate region detector.

Two backends (selected by PlateDetector(mode=...)):

  "crop"  (default, always available)
      ─ Crops a heuristic region from the vehicle bounding box.
        Plates are typically in the lower 35% of a car, full width of a
        motorcycle.  Fast, no extra model download.

  "yolo"  (higher accuracy, requires plate model weights)
      ─ Runs a small YOLO model trained specifically on license plates.
        Pass  model_path="models/plate_detector.pt"  to activate.
        Public weights: keremberke/yolov8n-license-plate-detection (HF Hub).

Output: List[PlateCandidate]  — one per detected plate region, with pixel bbox
        relative to the ORIGINAL frame (not the vehicle crop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from services.detect_track.models import Track

# Vehicle classes we run ANPR on
_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Minimum pixel area for a plate crop to be worth OCR-ing
_MIN_PLATE_AREA = 800   # px²


@dataclass
class PlateCandidate:
    """A cropped image region that likely contains a license plate."""
    crop:       np.ndarray          # BGR uint8, the plate image
    bbox_frame: Tuple[int,int,int,int]  # (x1,y1,x2,y2) in original frame
    track_id:   int
    conf:       float               # detector confidence (1.0 for heuristic)


class PlateDetector:
    """
    Detects license plate regions within vehicle bounding boxes.

    Parameters
    ----------
    mode        : "crop" | "yolo"
    model_path  : path to YOLO plate weights (only used when mode="yolo")
    device      : "cuda:0" | "cpu"
    half        : FP16 inference (mode="yolo" only)
    conf        : minimum detection confidence (mode="yolo" only)
    """

    def __init__(
        self,
        mode:       Literal["crop", "yolo"] = "crop",
        model_path: str = "models/plate_detector.pt",
        device:     str = "cuda:0",
        half:       bool = True,
        conf:       float = 0.55,
    ) -> None:
        self.mode = mode
        self._model = None

        if mode == "yolo":
            try:
                from ultralytics import YOLO
                self._model = YOLO(model_path)
                logger.info(f"Plate detector (YOLO): {model_path}")
            except Exception as e:
                logger.warning(
                    f"YOLO plate model failed to load ({e}). "
                    "Falling back to heuristic crop mode."
                )
                self.mode = "crop"

        if self.mode == "crop":
            logger.info("Plate detector: heuristic crop mode")

        self._device = device
        self._half   = half
        self._conf   = conf

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame:  np.ndarray,
        tracks: List[Track],
    ) -> List[PlateCandidate]:
        """
        Return plate candidates for all vehicle tracks in this frame.

        Parameters
        ----------
        frame  : full BGR frame
        tracks : active tracks from ByteTracker

        Returns
        -------
        List[PlateCandidate]  (may be empty)
        """
        candidates: List[PlateCandidate] = []

        for track in tracks:
            if track.class_name not in _VEHICLE_CLASSES:
                continue

            if self.mode == "yolo" and self._model is not None:
                candidates.extend(self._detect_yolo(frame, track))
            else:
                cand = self._detect_crop(frame, track)
                if cand:
                    candidates.append(cand)

        return candidates

    # ── Heuristic crop ────────────────────────────────────────────────────────

    @staticmethod
    def _detect_crop(frame: np.ndarray, track: Track) -> Optional[PlateCandidate]:
        """
        Crop the region of the vehicle bbox most likely to contain a plate.

        Car / truck / bus : lower 35% of the vehicle bbox
        Motorcycle        : lower 50% (plate often on rear at mid-height)
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in track.bbox)

        # Clamp to frame
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_frame, x2); y2 = min(h_frame, y2)

        bw = x2 - x1
        bh = y2 - y1
        if bw < 30 or bh < 30:
            return None

        if track.class_name == "motorcycle":
            frac = 0.50
        else:
            frac = 0.35

        # Take lower *frac* of vehicle bbox
        py1 = int(y2 - bh * frac)
        crop = frame[py1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] * crop.shape[1] < _MIN_PLATE_AREA:
            return None

        return PlateCandidate(
            crop       = crop,
            bbox_frame = (x1, py1, x2, y2),
            track_id   = track.track_id,
            conf       = 1.0,   # heuristic, no detector confidence
        )

    # ── YOLO plate detector ───────────────────────────────────────────────────

    def _detect_yolo(
        self, frame: np.ndarray, track: Track
    ) -> List[PlateCandidate]:
        """Run YOLO on the vehicle crop to find plate bboxes precisely."""
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_frame, x2); y2 = min(h_frame, y2)

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return []

        results = self._model.predict(
            vehicle_crop,
            conf=self._conf,
            device=self._device,
            half=self._half,
            verbose=False,
        )

        candidates = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                px1, py1_, px2, py2_ = (int(v) for v in box.xyxy[0].cpu())
                plate_crop = vehicle_crop[py1_:py2_, px1:px2]
                if plate_crop.size == 0:
                    continue
                # Convert back to frame coordinates
                fx1 = x1 + px1; fy1 = y1 + py1_
                fx2 = x1 + px2; fy2 = y1 + py2_
                candidates.append(PlateCandidate(
                    crop       = plate_crop,
                    bbox_frame = (fx1, fy1, fx2, fy2),
                    track_id   = track.track_id,
                    conf       = float(box.conf[0].cpu()),
                ))
        return candidates
