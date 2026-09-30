"""
services/detect_track/detector.py
===================================
IBVAP P1 — YOLOv8 FP16 object detector.

Wraps ultralytics YOLO with:
  - FP16 inference (half=True)
  - Class filtering (persons + vehicles only)
  - Configurable confidence / IoU thresholds
  - Warm-up on first call

Input:  np.ndarray  (BGR frame, HxWx3 uint8)
Output: List[Detection]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from services.detect_track.models import (
    COCO_CLASSES,
    DETECT_CLASS_IDS,
    Detection,
)


class YOLODetector:
    """
    YOLOv8/v11 FP16 detector for persons and vehicles.

    Parameters
    ----------
    model_path : str
        Path to .pt weights or a model name (e.g. "yolov8n.pt").
        Auto-downloads from Ultralytics if not found locally.
    device : str
        "cuda:0" | "cpu"
    half : bool
        FP16 inference (requires CUDA).
    conf : float
        Detection confidence threshold.
    iou : float
        NMS IoU threshold.
    imgsz : int
        Inference image size (longest side). 640 is a good default.
    classes : list[int] | None
        COCO class IDs to keep. None = use DETECT_CLASS_IDS.
    """

    def __init__(
        self,
        model_path:  str = "yolov8n.pt",
        device:      str = "cuda:0",
        half:        bool = True,
        conf:        float = 0.40,
        iou:         float = 0.45,
        imgsz:       int = 640,
        classes:     Optional[List[int]] = None,
        warmup_runs: int = 3,
    ) -> None:
        from ultralytics import YOLO  # deferred import so tests can mock it

        self.conf   = conf
        self.iou    = iou
        self.imgsz  = imgsz
        self.device = device
        self.half   = half and ("cuda" in device)
        self.classes = classes if classes is not None else DETECT_CLASS_IDS

        logger.info(f"Loading YOLO model: {model_path} → {device} | half={self.half}")
        self._model = YOLO(model_path)

        # Warm-up: feed a blank frame so CUDA kernels are compiled
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image, HxWx3 uint8.

        Returns
        -------
        List[Detection]
            Detections sorted by confidence descending.
        """
        results = self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            verbose=False,
        )

        return self._parse_results(results)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _warmup(self, n: int) -> None:
        """Run n inference passes on a blank frame to trigger CUDA kernel compilation."""
        import torch
        h, w = 720, 1280
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        logger.info(f"Warming up detector ({n} passes on {w}x{h} blank frame) …")
        for _ in range(n):
            self._model.predict(
                blank,
                conf=self.conf,
                classes=self.classes,
                device=self.device,
                half=self.half,
                imgsz=self.imgsz,
                verbose=False,
            )
        logger.success("Detector warm-up complete.")

    @staticmethod
    def _parse_results(results) -> List[Detection]:  # type: ignore[no-untyped-def]
        """Convert ultralytics Results → List[Detection]."""
        detections: List[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        xyxy  = boxes.xyxy.cpu().numpy()     # (N, 4)  x1 y1 x2 y2
        confs = boxes.conf.cpu().numpy()     # (N,)
        clses = boxes.cls.cpu().numpy()      # (N,)

        for i in range(len(xyxy)):
            cls_id = int(clses[i])
            detections.append(Detection(
                bbox=(
                    float(xyxy[i, 0]),
                    float(xyxy[i, 1]),
                    float(xyxy[i, 2]),
                    float(xyxy[i, 3]),
                ),
                conf=float(confs[i]),
                class_id=cls_id,
                class_name=COCO_CLASSES.get(cls_id, str(cls_id)),
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections
