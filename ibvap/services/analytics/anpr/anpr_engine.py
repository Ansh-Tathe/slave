"""
services/analytics/anpr/anpr_engine.py
========================================
IBVAP P3 — ANPR orchestrator.

Chains: PlateDetector → PlateOCR → IndianPlateValidator → ANPR_READ Event.

Per-track deduplication:
  Each unique (track_id, normalised_plate_text) pair is reported only once
  per dedup_window_s seconds to avoid flooding the event log.

Interface:
    engine = ANPREngine(use_gpu=True)
    events = engine.update(frame, tracks, frame_id=42, camera_id="cam_01")
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from loguru import logger

from services.analytics.anpr.ocr_engine    import PlateOCR
from services.analytics.anpr.plate_detector import PlateDetector
from services.analytics.anpr.validator      import IndianPlateValidator, PlateRead
from services.detect_track.models           import Track
from services.event_engine.models           import Event, EventType, Severity

import numpy as np


# ── Watchlist stub (populated in P7) ─────────────────────────────────────────
# Replace with a real DB lookup in P7.
_WATCHLIST: set[str] = set()   # normalised plate strings


def add_to_watchlist(plate: str) -> None:
    _WATCHLIST.add(IndianPlateValidator._PlateValidator__clean(plate)  # type: ignore
                   if False else plate.upper().replace(" ", ""))


# ── ANPR Engine ───────────────────────────────────────────────────────────────

class ANPREngine:
    """
    Full ANPR pipeline for one camera stream.

    Parameters
    ----------
    detector_mode       : "crop" (always works) | "yolo" (needs plate model)
    plate_model_path    : path to YOLO plate weights (used if mode="yolo")
    use_gpu             : GPU for EasyOCR
    min_format_conf     : discard reads with format_confidence below this
    dedup_window_s      : suppress same plate + track for this many seconds
    camera_id           : used in emitted events
    """

    def __init__(
        self,
        detector_mode:    str   = "crop",
        plate_model_path: str   = "models/plate_detector.pt",
        use_gpu:          bool  = True,
        min_format_conf:  float = 0.40,
        dedup_window_s:   float = 30.0,
        camera_id:        str   = "unknown",
    ) -> None:
        self._detector  = PlateDetector(
            mode=detector_mode,
            model_path=plate_model_path,
            device="cuda:0" if use_gpu else "cpu",
        )
        self._ocr       = PlateOCR(use_gpu=use_gpu)
        self._validator = IndianPlateValidator()
        self._min_conf  = min_format_conf
        self._dedup_w   = dedup_window_s
        self._camera    = camera_id

        # dedup: (track_id, plate_text) → last_seen (monotonic)
        self._last_seen: Dict[Tuple[int, str], float] = defaultdict(float)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame:     np.ndarray,
        tracks:    List[Track],
        frame_id:  int,
        camera_id: Optional[str] = None,
    ) -> List[Event]:
        """
        Process one frame: detect → OCR → validate → emit events.

        Returns a list of ANPR_READ or ANPR_WATCHLIST Events.
        May be empty if no vehicles are visible or no plates are read.
        """
        cam = camera_id or self._camera
        events: List[Event] = []
        now = time.monotonic()

        # ── Detect plate regions ──────────────────────────────────────────
        candidates = self._detector.detect(frame, tracks)
        if not candidates:
            return []

        # ── OCR ───────────────────────────────────────────────────────────
        ocr_results = self._ocr.read_candidates(candidates)

        # ── Validate + emit events ────────────────────────────────────────
        for ocr in ocr_results:
            plate: PlateRead = self._validator.validate(
                ocr.text, ocr_confidence=ocr.confidence
            )

            if plate.format_confidence < self._min_conf:
                logger.debug(
                    f"[{cam}] ANPR low-conf skip: "
                    f"'{plate.normalised_text}' conf={plate.format_confidence:.2f}"
                )
                continue

            # Dedup
            key = (ocr.candidate.track_id, plate.normalised_text)
            if now - self._last_seen[key] < self._dedup_w:
                continue
            self._last_seen[key] = now

            # Find the originating track
            track = self._find_track(tracks, ocr.candidate.track_id)
            if track is None:
                continue

            watchlisted = plate.normalised_text in _WATCHLIST
            etype = EventType.ANPR_WATCHLIST if watchlisted else EventType.ANPR_READ
            sev   = Severity.CRITICAL if watchlisted else Severity.LOW

            events.append(Event(
                event_type = etype,
                severity   = sev,
                camera_id  = cam,
                track      = track,
                frame_id   = frame_id,
                zone_id    = None,
                metadata   = {
                    **plate.to_dict(),
                    "ocr_raw":       ocr.text,
                    "ocr_conf":      round(ocr.confidence, 4),
                    "watchlisted":   watchlisted,
                    "plate_bbox":    list(ocr.candidate.bbox_frame),
                },
            ))

            logger.info(
                f"[{cam}] ANPR  track={track.track_id}  "
                f"plate='{plate.normalised_text}'  "
                f"state={plate.state_name or '?'}  "
                f"valid={plate.is_valid_format}  "
                f"conf={plate.format_confidence:.2f}"
                + ("  ⚠️  WATCHLISTED" if watchlisted else "")
            )

        return events

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_track(tracks: List[Track], track_id: int) -> Optional[Track]:
        for t in tracks:
            if t.track_id == track_id:
                return t
        return None
