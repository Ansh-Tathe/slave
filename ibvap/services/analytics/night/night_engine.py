"""
services/analytics/night/night_engine.py
==========================================
IBVAP P4 — Night analytics orchestrator.

Responsibilities
----------------
1. Brightness gate  — call NightEnhancer.is_dark() per frame; only enhance
   when the frame is actually dark (avoids processing daytime frames).
2. Hour-of-day gate — optionally suppress NIGHT_MOVEMENT events outside
   the configured night hours (e.g. 20:00-05:00).
3. Enhancement     — call NightEnhancer.enhance() and return the
   brightened frame for downstream detect/track.
4. Event emission  — for every active track in a dark frame, emit a
   NIGHT_MOVEMENT event (deduplicated by track_id + window).

Interface
---------
    engine  = NightEngine(enhancer_mode="combined")
    bright, events = engine.update(frame, tracks, frame_id, camera_id)
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from loguru import logger

from services.analytics.night.night_enhancer import NightEnhancer
from services.detect_track.models import Track
from services.event_engine.models import Event, EventType, Severity


_DEFAULT_NIGHT_HOURS = {20, 21, 22, 23, 0, 1, 2, 3, 4, 5}


class NightEngine:
    """
    Night-time enhancement + movement detection orchestrator.

    Parameters
    ----------
    enhancer_mode   : "clahe" | "gamma" | "combined" | "dnn"
    dark_threshold  : mean LAB-L below this -> frame is dark (0-255)
    dnn_model_path  : Zero-DCE ONNX path (enhancer_mode="dnn" only)
    night_hours     : set of ints (0-23) — emit events only in these hours.
                      None = emit regardless of time.
    dedup_window_s  : suppress same track_id event for this many seconds
    event_severity  : Severity for NIGHT_MOVEMENT events
    camera_id       : used in emitted events
    always_enhance  : enhance every frame regardless of darkness level
    """

    def __init__(
        self,
        enhancer_mode:  Literal["clahe", "gamma", "combined", "dnn"] = "combined",
        dark_threshold: float = 80.0,
        dnn_model_path: Optional[str] = None,
        night_hours:    Optional[set] = None,
        dedup_window_s: float = 30.0,
        event_severity: Severity = Severity.HIGH,
        camera_id:      str = "unknown",
        always_enhance: bool = False,
    ) -> None:
        self._enhancer = NightEnhancer(
            mode=enhancer_mode,
            dark_threshold=dark_threshold,
            dnn_model_path=dnn_model_path,
        )
        self._night_hours    = night_hours
        self._dedup_window   = dedup_window_s
        self._severity       = event_severity
        self._camera         = camera_id
        self._always_enhance = always_enhance

        self._last_fired: Dict[int, float] = defaultdict(float)
        self._enhanced_frames: int = 0
        self._skipped_frames:  int = 0
        self.last_events: List[Event] = []

        logger.info(
            f"NightEngine ready — enhancer={enhancer_mode}, "
            f"dark_threshold={dark_threshold:.0f}, "
            f"dedup={dedup_window_s}s, "
            f"night_hours={sorted(night_hours) if night_hours else 'any'}"
        )

    # -- Public API ------------------------------------------------------------

    def update(
        self,
        frame:     np.ndarray,
        tracks:    List[Track],
        frame_id:  int,
        camera_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Event]]:
        """
        Process one frame end-to-end.

        Returns
        -------
        (enhanced_frame, events)
        """
        cam    = camera_id or self._camera
        bright = self.process_frame(frame)
        events = self._emit_events(frame, tracks, frame_id, cam)
        self.last_events = events
        return bright, events

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Return enhanced frame if dark (or always_enhance=True), else original.
        """
        if self._always_enhance or self._enhancer.is_dark(frame):
            self._enhanced_frames += 1
            return self._enhancer.enhance(frame)
        self._skipped_frames += 1
        return frame

    def is_night_hour(self) -> bool:
        """Return True if current wall-clock hour is in night_hours."""
        if self._night_hours is None:
            return True
        return datetime.now().hour in self._night_hours

    @property
    def enhancer(self) -> NightEnhancer:
        return self._enhancer

    @property
    def stats(self) -> dict:
        return {
            "enhanced_frames": self._enhanced_frames,
            "skipped_frames":  self._skipped_frames,
        }

    # -- Private ---------------------------------------------------------------

    def _emit_events(
        self,
        frame:     np.ndarray,
        tracks:    List[Track],
        frame_id:  int,
        camera_id: str,
    ) -> List[Event]:
        """Emit NIGHT_MOVEMENT for each track, with time gate + dedup."""
        if not self.is_night_hour():
            return []

        events: List[Event] = []
        now    = time.monotonic()
        lum    = round(self._enhancer.mean_luminance(frame), 1)

        for track in tracks:
            last = self._last_fired.get(track.track_id, 0.0)
            if (now - last) < self._dedup_window:
                continue
            self._last_fired[track.track_id] = now

            events.append(Event(
                event_type = EventType.NIGHT_MOVEMENT,
                severity   = self._severity,
                camera_id  = camera_id,
                track      = track,
                frame_id   = frame_id,
                zone_id    = None,
                metadata   = {
                    "luminance":      lum,
                    "enhancer_mode":  self._enhancer._mode,
                    "is_dark":        self._enhancer.is_dark(frame),
                },
            ))

            logger.info(
                f"[{camera_id}] NIGHT_MOVEMENT  "
                f"track={track.track_id}  class={track.class_name}  "
                f"lum={lum:.1f}  frame={frame_id}"
            )

        return events
