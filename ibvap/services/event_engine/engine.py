"""
services/event_engine/engine.py
================================
IBVAP P2 — Event engine: deduplication, snapshot capture, severity lookup.

Responsibilities
----------------
1. Deduplicate: suppress identical (event_type, track_id, zone_id) within
   a rolling time window (from rules.yaml → event_engine.dedup_window_s).
2. Snapshot: crop and save a JPEG of the track bbox + padding.
3. Severity override: apply per-rule severity from rules.yaml if defined.
4. Return only events that pass the dedup filter.

Does NOT do:
  - Clip recording (queued for P7 when we have async storage)
  - Webhook dispatch (P7)
  - Database write (P7)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from services.event_engine.models import Event, EventType, Severity


# ── Dedup key: (event_type, track_id, zone_id) ───────────────────────────────
_DedupKey = Tuple[str, int, Optional[str]]


class EventEngine:
    """
    Stateful event processing pipeline.

    Parameters
    ----------
    dedup_window_s  : suppress repeat events within this many seconds.
    snapshot_dir    : directory to save JPEG snapshots (created if missing).
    snapshot_quality: JPEG quality 0-100.
    snapshot_padding: extra pixels around bbox in snapshot.
    max_eps         : max events per second (rate limiter).
    """

    def __init__(
        self,
        dedup_window_s:    float = 30.0,
        snapshot_dir:      str   = "data/snapshots",
        snapshot_quality:  int   = 85,
        snapshot_padding:  int   = 30,
        max_eps:           float = 50.0,
    ) -> None:
        self._dedup_window    = dedup_window_s
        self._snap_dir        = Path(snapshot_dir)
        self._snap_dir.mkdir(parents=True, exist_ok=True)
        self._snap_quality    = snapshot_quality
        self._snap_padding    = snapshot_padding
        self._max_eps         = max_eps

        # dedup: key → last_fired_time (monotonic)
        self._last_fired: Dict[_DedupKey, float] = defaultdict(float)

        # rate limiter
        self._event_count: int   = 0
        self._rate_window: float = time.monotonic()

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        raw_events: List[Event],
        frame:      np.ndarray,
    ) -> List[Event]:
        """
        Filter, deduplicate, and enrich a list of raw events.

        Parameters
        ----------
        raw_events : events from FenceEngine (or any analytics module)
        frame      : current BGR frame — used to capture snapshots

        Returns
        -------
        List[Event] — only events that passed dedup and rate limits.
        """
        out: List[Event] = []
        now = time.monotonic()

        # Reset rate counter each second
        if now - self._rate_window >= 1.0:
            self._event_count = 0
            self._rate_window = now

        for evt in raw_events:
            # Rate limit
            if self._event_count >= self._max_eps:
                logger.warning("Event rate limit reached — dropping events this second")
                break

            key: _DedupKey = (
                evt.event_type.value,
                evt.track.track_id,
                evt.zone_id,
            )

            last = self._last_fired.get(key, 0.0)
            if (now - last) < self._dedup_window:
                continue   # suppressed

            self._last_fired[key] = now
            self._event_count += 1

            # Attach snapshot
            snap_path = self._save_snapshot(evt, frame)
            evt.snapshot_path = snap_path

            out.append(evt)

        return out

    @classmethod
    def from_config(cls, rules_yaml: str) -> "EventEngine":
        """Load settings from rules.yaml."""
        import yaml
        path = Path(rules_yaml)
        if not path.exists():
            logger.warning(f"rules.yaml not found: {path} — using defaults")
            return cls()

        with path.open() as f:
            raw = yaml.safe_load(f)

        ee = raw.get("event_engine", {})
        snap = ee.get("snapshot", {})

        return cls(
            dedup_window_s   = ee.get("dedup_window_s", 30.0),
            snapshot_quality = snap.get("quality", 85),
            snapshot_padding = snap.get("padding_px", 30),
            max_eps          = ee.get("max_events_per_second", 50.0),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_snapshot(self, evt: Event, frame: np.ndarray) -> Optional[str]:
        """Crop bbox region + padding from frame and save as JPEG."""
        try:
            h, w = frame.shape[:2]
            pad  = self._snap_padding
            x1   = max(0, int(evt.track.bbox[0]) - pad)
            y1   = max(0, int(evt.track.bbox[1]) - pad)
            x2   = min(w, int(evt.track.bbox[2]) + pad)
            y2   = min(h, int(evt.track.bbox[3]) + pad)

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            fname = f"{evt.event_id[:8]}_{evt.event_type.value}.jpg"
            path  = str(self._snap_dir / fname)
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, self._snap_quality])
            return path
        except Exception as e:
            logger.warning(f"Snapshot failed: {e}")
            return None
