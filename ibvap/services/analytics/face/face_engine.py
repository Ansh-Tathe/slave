"""
services/analytics/face/face_engine.py
========================================
IBVAP P6 -- Face analytics orchestrator.

Pipeline per frame:
  1. FaceDetector.detect(frame)     -> List[FaceDetection]
  2. FaceEmbedder.embed(crop)       -> (512,) L2-normed embedding
  3. Watchlist.search(embedding)    -> (name, similarity) or None
  4. Emit:
       FACE_DETECTED   -- any face seen (quality-passed), not on watchlist
       FACE_WATCHLIST  -- face matches watchlist entry >= threshold

Deduplication:
  Per-track (tracked_id from the parent detect/track pipeline) or
  per-detection-position if no tracking is available.

Usage
-----
    engine = FaceEngine(watchlist_path="data/watchlist/watchlist.npz")
    events = engine.update(frame, tracks=tracks, frame_id=42, camera_id="cam_01")
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from services.analytics.face.face_detector import FaceDetection, FaceDetector
from services.analytics.face.face_embedder import FaceEmbedder
from services.analytics.face.watchlist     import Watchlist
from services.detect_track.models          import Track
from services.event_engine.models          import Event, EventType, Severity


class FaceEngine:
    """
    Complete face detection + recognition + watchlist matching pipeline.

    Parameters
    ----------
    watchlist_path          : path to .npz watchlist file (loaded on init)
    face_backend            : "insightface" | "yolo"
    face_model_pack         : InsightFace model pack name
    yolo_face_model_path    : YOLO face model path (backend="yolo")
    device                  : "cuda:0" | "cpu"
    min_det_conf            : face detection min confidence
    min_area_px2            : minimum face area to process
    min_sharpness           : Laplacian sharpness threshold
    match_threshold         : cosine similarity >= this -> watchlist match
    dedup_position_px       : faces within this px radius are same face
    dedup_window_s          : suppress same-position events for N seconds
    face_detected_severity  : severity for FACE_DETECTED events
    watchlist_severity      : severity for FACE_WATCHLIST events
    camera_id               : used in emitted events
    enable_face_detected    : emit FACE_DETECTED events (may be noisy)
    """

    def __init__(
        self,
        watchlist_path:        Optional[str] = None,
        face_backend:          str = "insightface",
        face_model_pack:       str = "buffalo_l",
        yolo_face_model_path:  Optional[str] = None,
        device:                str = "cuda:0",
        min_det_conf:          float = 0.60,
        min_area_px2:          int = 900,
        min_sharpness:         float = 30.0,
        match_threshold:       float = 0.45,
        dedup_position_px:     float = 60.0,
        dedup_window_s:        float = 30.0,
        face_detected_severity: Severity = Severity.LOW,
        watchlist_severity:    Severity = Severity.CRITICAL,
        camera_id:             str = "unknown",
        enable_face_detected:  bool = True,
    ) -> None:
        self._threshold    = match_threshold
        self._dedup_pos    = dedup_position_px
        self._dedup_w      = dedup_window_s
        self._fd_severity  = face_detected_severity
        self._wl_severity  = watchlist_severity
        self._camera       = camera_id
        self._en_detected  = enable_face_detected

        # Dedup: (cx_bucket, cy_bucket) -> last_fired_time
        self._last_fired: Dict[Tuple[int, int], float] = defaultdict(float)

        # Stats
        self._total_detected  = 0
        self._total_matched   = 0

        # Build sub-components
        self.detector = FaceDetector(
            backend         = face_backend,
            model_pack      = face_model_pack,
            yolo_model_path = yolo_face_model_path,
            device          = device,
            min_det_conf    = min_det_conf,
            min_area_px2    = min_area_px2,
            min_sharpness   = min_sharpness,
        )
        self.embedder = FaceEmbedder(device=device)
        self.watchlist = Watchlist()

        if watchlist_path:
            self.watchlist.load(watchlist_path)

        logger.info(
            f"FaceEngine ready — "
            f"backend={face_backend}, "
            f"match_threshold={match_threshold}, "
            f"watchlist_size={self.watchlist.size}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame:     np.ndarray,
        tracks:    Optional[List[Track]] = None,
        frame_id:  int = 0,
        camera_id: Optional[str] = None,
    ) -> Tuple[List[Event], List[FaceDetection]]:
        """
        Run the full face pipeline on one frame.

        Returns
        -------
        (events, face_detections)
          events          : List[Event] (FACE_DETECTED / FACE_WATCHLIST)
          face_detections : List[FaceDetection] for downstream rendering
        """
        cam    = camera_id or self._camera
        now    = time.monotonic()
        events: List[Event] = []

        # 1. Detect faces
        faces = self.detector.detect(frame)
        if not faces:
            return [], []

        self._total_detected += len(faces)

        # 2. For each face: embed + search watchlist
        for face in faces:
            # Dedup by position bucket
            cx, cy  = face.center
            bkt     = (int(cx // self._dedup_pos), int(cy // self._dedup_pos))
            last    = self._last_fired.get(bkt, 0.0)
            if (now - last) < self._dedup_w:
                continue
            self._last_fired[bkt] = now

            # 3. Embed
            emb = self.embedder.embed(face.crop)

            # 4. Match watchlist
            match = self.watchlist.search(emb, threshold=self._threshold)

            # Build a stub Track for the event (face detection, not obj-tracker track)
            stub_track = _face_stub_track(face, frame_id, cam)

            if match is not None:
                name, sim = match
                self._total_matched += 1
                events.append(Event(
                    event_type = EventType.FACE_WATCHLIST,
                    severity   = self._wl_severity,
                    camera_id  = cam,
                    track      = stub_track,
                    frame_id   = frame_id,
                    zone_id    = None,
                    metadata   = {
                        "name":        name,
                        "similarity":  round(sim, 4),
                        "threshold":   self._threshold,
                        "sharpness":   round(face.sharpness, 1),
                        "det_conf":    round(face.conf, 3),
                    },
                ))
                logger.info(
                    f"[{cam}] FACE_WATCHLIST  "
                    f"name='{name}'  sim={sim:.3f}  frame={frame_id}"
                )
            elif self._en_detected:
                events.append(Event(
                    event_type = EventType.FACE_DETECTED,
                    severity   = self._fd_severity,
                    camera_id  = cam,
                    track      = stub_track,
                    frame_id   = frame_id,
                    zone_id    = None,
                    metadata   = {
                        "sharpness": round(face.sharpness, 1),
                        "det_conf":  round(face.conf, 3),
                        "area_px2":  face.area,
                    },
                ))
                logger.debug(
                    f"[{cam}] FACE_DETECTED  "
                    f"conf={face.conf:.2f}  frame={frame_id}"
                )

        return events, faces

    def add_to_watchlist(
        self,
        name:  str,
        image: np.ndarray,
    ) -> bool:
        """
        Detect a face in `image`, embed it, and add to the watchlist.

        Returns True on success.
        """
        return self.watchlist.add_person_from_image(
            name     = name,
            image    = image,
            embedder = self.embedder,
            detector = self.detector,
        )

    def save_watchlist(self, path: str) -> None:
        self.watchlist.save(path)

    @property
    def stats(self) -> dict:
        return {
            "total_detected": self._total_detected,
            "total_matched":  self._total_matched,
            "watchlist_size": self.watchlist.size,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _find_associated_track(
        self,
        face:    FaceDetection,
        tracks:  List[Track],
        radius:  float = 100.0,
    ) -> Optional[Track]:
        """Find the closest object-tracker Track whose centroid is near the face."""
        if not tracks:
            return None
        fx, fy = face.center
        best   = min(
            tracks,
            key=lambda t: (t.center[0] - fx) ** 2 + (t.center[1] - fy) ** 2,
        )
        bx, by = best.center
        if ((bx - fx) ** 2 + (by - fy) ** 2) ** 0.5 <= radius:
            return best
        return None


# ── Stub Track for face events ────────────────────────────────────────────────

def _face_stub_track(face: FaceDetection, frame_id: int, camera_id: str) -> Track:
    """Create a minimal Track from a FaceDetection for use in events."""
    from services.detect_track.models import Track
    return Track(
        track_id   = -1,          # -1 = face-only (no object-tracker ID)
        bbox       = tuple(float(v) for v in face.bbox),
        conf       = face.conf,
        class_id   = -1,
        class_name = "face",
        frame_id   = frame_id,
        camera_id  = camera_id,
    )
