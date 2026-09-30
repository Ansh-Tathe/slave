"""
services/detect_track/tracker.py
==================================
IBVAP P1 — ByteTrack wrapper via ultralytics.

Ultralytics bundles ByteTrack (and BoT-SORT) natively.
We call model.track() which runs detection + tracking in one pass,
giving us persistent track IDs without a separate tracking library.

Input:  np.ndarray frame  +  detector config
Output: List[Track]

Why model.track() instead of a separate ByteTrack call?
  - ultralytics handles the Kalman filter state internally
  - persist=True keeps state across calls for the same model instance
  - BoT-SORT can be swapped in by changing tracker_cfg
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from services.detect_track.models import (
    COCO_CLASSES,
    DETECT_CLASS_IDS,
    Track,
)


class ByteTracker:
    """
    Wraps ultralytics YOLO + ByteTrack for detection + tracking in one call.

    Parameters
    ----------
    model_path : str
        YOLOv8 weights (same as detector).
    tracker_cfg : str
        "bytetrack.yaml" or "botsort.yaml" — bundled in ultralytics.
    device : str
    half : bool
    conf, iou, imgsz : float, float, int  — same as YOLODetector
    classes : list[int] | None
    """

    def __init__(
        self,
        model_path:  str = "yolov8n.pt",
        tracker_cfg: str = "bytetrack.yaml",
        device:      str = "cuda:0",
        half:        bool = True,
        conf:        float = 0.40,
        iou:         float = 0.45,
        imgsz:       int = 640,
        classes:     Optional[List[int]] = None,
        warmup_runs: int = 3,
    ) -> None:
        from ultralytics import YOLO

        self.conf        = conf
        self.iou         = iou
        self.imgsz       = imgsz
        self.device      = device
        self.half        = half and ("cuda" in device)
        self.tracker_cfg = tracker_cfg
        self.classes     = classes if classes is not None else DETECT_CLASS_IDS

        logger.info(
            f"Loading tracker model: {model_path} | tracker={tracker_cfg} | "
            f"device={device} | half={self.half}"
        )
        self._model = YOLO(model_path)

        if warmup_runs > 0:
            self._warmup(warmup_runs)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame:     np.ndarray,
        frame_id:  int,
        camera_id: str,
        timestamp: float,
    ) -> List[Track]:
        """
        Run detection + tracking on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray       BGR uint8
        frame_id : int           monotonic frame counter from FrameReader
        camera_id : str          camera identifier from config
        timestamp : float        wall-clock time from FrameReader

        Returns
        -------
        List[Track]
            Active tracks in this frame.  May be empty.
        """
        results = self._model.track(
            frame,
            persist=True,               # keep Kalman state between calls
            tracker=self.tracker_cfg,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            verbose=False,
        )

        return self._parse_results(results, frame_id, camera_id, timestamp)

    def reset(self) -> None:
        """Reset tracker state (call when switching cameras or after a gap)."""
        # Ultralytics doesn't expose a reset API directly; reload tracker state
        # by toggling persist=False on the next call — workaround: reinit model.
        from ultralytics import YOLO
        self._model = YOLO(self._model.ckpt_path)
        logger.info("Tracker state reset.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _warmup(self, n: int) -> None:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        logger.info(f"Warming up tracker ({n} passes) …")
        for _ in range(n):
            self._model.track(
                blank,
                persist=True,
                tracker=self.tracker_cfg,
                conf=self.conf,
                classes=self.classes,
                device=self.device,
                half=self.half,
                imgsz=self.imgsz,
                verbose=False,
            )
        logger.success("Tracker warm-up complete.")

    @staticmethod
    def _parse_results(
        results,
        frame_id:  int,
        camera_id: str,
        timestamp: float,
    ) -> List[Track]:
        tracks: List[Track] = []

        if not results or results[0].boxes is None:
            return tracks

        boxes = results[0].boxes

        # When no track IDs are assigned (e.g. first frame), boxes.id is None
        if boxes.id is None:
            return tracks

        xyxy     = boxes.xyxy.cpu().numpy()   # (N,4)
        confs    = boxes.conf.cpu().numpy()   # (N,)
        clses    = boxes.cls.cpu().numpy()    # (N,)
        ids      = boxes.id.cpu().numpy()     # (N,)  float32

        for i in range(len(xyxy)):
            cls_id = int(clses[i])
            tracks.append(Track(
                track_id=int(ids[i]),
                bbox=(
                    float(xyxy[i, 0]),
                    float(xyxy[i, 1]),
                    float(xyxy[i, 2]),
                    float(xyxy[i, 3]),
                ),
                conf=float(confs[i]),
                class_id=cls_id,
                class_name=COCO_CLASSES.get(cls_id, str(cls_id)),
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp=timestamp,
            ))

        return tracks
