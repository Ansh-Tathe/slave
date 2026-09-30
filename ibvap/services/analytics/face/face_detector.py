"""
services/analytics/face/face_detector.py
==========================================
IBVAP P6 -- Face detector.

Two backends:

  "insightface"  (default, recommended)
      Uses InsightFace RetinaFace via the insightface.app.FaceAnalysis API.
      Downloads ~330 MB of ONNX models on first run (buffalo_l pack).
      Returns bounding boxes, 5-point landmarks, and detection confidence.

  "yolo"
      Uses a YOLOv8-face ONNX model.  Much smaller (~6 MB), faster but
      slightly less accurate.  Requires a YOLO face weights file.
      Public weights: https://github.com/akanametov/yolo-face

Output: List[FaceDetection] -- one per detected face in the frame.

Quality filtering
-----------------
Faces are discarded when:
  - Detection confidence < min_det_conf
  - Bounding box area < min_area_px2  (too small/far away)
  - Blur score (Laplacian variance) < min_sharpness  (blurry/motion-blurred)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


@dataclass
class FaceDetection:
    """One detected face in a frame."""
    bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixels
    conf:        float                       # detection confidence 0-1
    landmarks:   Optional[np.ndarray]        # (5, 2) float32 or None
    sharpness:   float                       # Laplacian variance (quality proxy)
    crop:        np.ndarray                  # BGR face crop (aligned if landmarks available)
    frame_area:  int                         # total frame area (for relative-size checks)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2 - x1) * (y2 - y1))

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _laplacian_sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian as a sharpness metric (higher = sharper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """
    Affine-align face to a canonical 112x112 crop using 5-point landmarks.
    Falls back to plain bbox crop if alignment fails.
    """
    # Reference landmarks for 112x112 (ArcFace standard)
    _REF = np.array([
        [38.29, 51.70], [73.53, 51.50],
        [56.02, 71.74],
        [41.55, 92.37], [70.73, 92.10],
    ], dtype=np.float32)

    try:
        src = landmarks.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(
            src, _REF, method=cv2.LMEDS
        )
        if M is None:
            raise ValueError("affine estimation failed")
        aligned = cv2.warpAffine(img, M, (output_size, output_size),
                                 flags=cv2.INTER_LINEAR)
        return aligned
    except Exception:
        # Plain resize fallback
        h, w = img.shape[:2]
        if w == 0 or h == 0:
            return np.zeros((output_size, output_size, 3), dtype=np.uint8)
        return cv2.resize(img, (output_size, output_size))


# ── FaceDetector ─────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Detects and quality-filters faces in BGR frames.

    Parameters
    ----------
    backend         : "insightface" | "yolo"
    model_pack      : InsightFace model pack name (default "buffalo_l")
    yolo_model_path : path to YOLOv8-face .pt weights (backend="yolo" only)
    device          : "cuda:0" | "cpu"
    min_det_conf    : minimum detection confidence (0-1)
    min_area_px2    : minimum face bbox area in pixels (discard tiny faces)
    min_sharpness   : minimum Laplacian variance (discard blurry faces)
    det_size        : InsightFace detection resolution (width, height)
    """

    def __init__(
        self,
        backend:         Literal["insightface", "yolo"] = "insightface",
        model_pack:      str = "buffalo_l",
        yolo_model_path: Optional[str] = None,
        device:          str = "cuda:0",
        min_det_conf:    float = 0.60,
        min_area_px2:    int = 900,        # 30x30 min face
        min_sharpness:   float = 30.0,
        det_size:        Tuple[int, int] = (640, 640),
    ) -> None:
        self._backend      = backend
        self._min_conf     = min_det_conf
        self._min_area     = min_area_px2
        self._min_sharp    = min_sharpness
        self._device       = device
        self._det_size     = det_size
        self._app          = None   # InsightFace FaceAnalysis
        self._yolo_model   = None   # ultralytics YOLO

        self._ctx_id = 0 if "cuda" in device else -1   # InsightFace context

        if backend == "insightface":
            self._init_insightface(model_pack)
        elif backend == "yolo":
            self._init_yolo(yolo_model_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect and quality-filter faces in a BGR frame.

        Returns
        -------
        List[FaceDetection] sorted by confidence descending.
        """
        if frame is None or frame.size == 0:
            return []

        if self._backend == "insightface" and self._app is not None:
            return self._detect_insightface(frame)
        elif self._backend == "yolo" and self._yolo_model is not None:
            return self._detect_yolo(frame)
        else:
            return []   # backend not loaded

    @property
    def is_ready(self) -> bool:
        return (self._app is not None) or (self._yolo_model is not None)

    # ── Backends ──────────────────────────────────────────────────────────────

    def _init_insightface(self, model_pack: str) -> None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name       = model_pack,
                providers  = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                              if "cuda" in self._device
                              else ["CPUExecutionProvider"]),
            )
            self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
            logger.success(
                f"InsightFace FaceAnalysis ready — pack={model_pack}, "
                f"det_size={self._det_size}, ctx_id={self._ctx_id}"
            )
        except Exception as exc:
            logger.error(f"InsightFace init failed: {exc}")
            self._app = None

    def _init_yolo(self, model_path: Optional[str]) -> None:
        if not model_path or not Path(model_path).exists():
            logger.warning(
                f"YOLO face model not found: {model_path}. "
                "Pass --face-model <path> or use --face-backend insightface."
            )
            return
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(model_path)
            logger.success(f"YOLO face detector loaded: {model_path}")
        except Exception as exc:
            logger.error(f"YOLO face model init failed: {exc}")

    def _detect_insightface(self, frame: np.ndarray) -> List[FaceDetection]:
        """Run InsightFace detection + landmark extraction."""
        try:
            faces = self._app.get(frame)
        except Exception as exc:
            logger.warning(f"InsightFace get() error: {exc}")
            return []

        detections: List[FaceDetection] = []
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame

        for face in faces:
            conf = float(face.det_score)
            if conf < self._min_conf:
                continue

            box  = face.bbox.astype(int)
            x1   = max(0, box[0]); y1 = max(0, box[1])
            x2   = min(w_frame, box[2]); y2 = min(h_frame, box[3])
            area = (x2 - x1) * (y2 - y1)

            if area < self._min_area:
                continue

            # Landmarks (5-point)
            lm = face.kps.astype(np.float32) if hasattr(face, "kps") and face.kps is not None else None

            # Aligned face crop
            raw_crop = frame[y1:y2, x1:x2]
            if lm is not None:
                crop = _align_face(frame, lm)
            else:
                crop = cv2.resize(raw_crop, (112, 112)) if raw_crop.size > 0 else \
                       np.zeros((112, 112, 3), dtype=np.uint8)

            sharpness = _laplacian_sharpness(crop)
            if sharpness < self._min_sharp:
                continue

            detections.append(FaceDetection(
                bbox       = (x1, y1, x2, y2),
                conf       = conf,
                landmarks  = lm,
                sharpness  = sharpness,
                crop       = crop,
                frame_area = frame_area,
            ))

        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections

    def _detect_yolo(self, frame: np.ndarray) -> List[FaceDetection]:
        """Run YOLOv8-face detection."""
        try:
            results = self._yolo_model.predict(
                frame,
                conf=self._min_conf,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning(f"YOLO face predict error: {exc}")
            return []

        detections: List[FaceDetection] = []
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame

        if not results or results[0].boxes is None:
            return []

        for box_obj in results[0].boxes:
            conf = float(box_obj.conf[0].cpu())
            x1, y1, x2, y2 = (int(v) for v in box_obj.xyxy[0].cpu())
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_frame, x2); y2 = min(h_frame, y2)
            area = (x2 - x1) * (y2 - y1)

            if area < self._min_area:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (112, 112))
            sharpness    = _laplacian_sharpness(crop_resized)

            if sharpness < self._min_sharp:
                continue

            detections.append(FaceDetection(
                bbox       = (x1, y1, x2, y2),
                conf       = conf,
                landmarks  = None,
                sharpness  = sharpness,
                crop       = crop_resized,
                frame_area = frame_area,
            ))

        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections
