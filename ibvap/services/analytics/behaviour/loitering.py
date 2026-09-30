"""
services/analytics/behaviour/loitering.py
==========================================
IBVAP P5 -- Zone-agnostic loitering (LOITERING) detector.

Difference from P2 DWELL
------------------------
P2 DWELL fires when a track stays inside a CONFIGURED POLYGON ZONE.
P5 LOITERING fires for ANY track that stays in a small radius regardless
of zone configuration -- useful for catching loiterers in unconfigured areas.

Algorithm
---------
For each track, record its first_seen timestamp and centroid when it first
appears.  Each subsequent frame, check:
  1. Has the track moved more than `movement_threshold_px` from its seed
     centroid?  If yes, reset the dwell clock (the person is moving).
  2. Has the track been stationary for >= `dwell_seconds`?
     If yes, emit LOITERING (once per dedup_window_s).

State is pruned when tracks disappear.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from services.detect_track.models import Track
from services.event_engine.models  import Event, EventType, Severity


@dataclass
class _LoiterState:
    seed_cx:      float   # centroid X when dwell clock started
    seed_cy:      float   # centroid Y when dwell clock started
    first_seen:   float   # monotonic timestamp when clock started
    alerted:      bool = False
    alert_time:   float = 0.0  # last alert monotonic time (for dedup)


class LoiteringDetector:
    """
    Detects loitering by tracking per-track stationary dwell time.

    Parameters
    ----------
    dwell_seconds        : seconds a track must stay stationary to trigger
    movement_threshold_px: movement > this resets the dwell clock
    classes              : which class_names to monitor (default: person)
    dedup_window_s       : suppress repeated LOITERING for same track
    severity             : event severity
    camera_id            : used in emitted events
    """

    def __init__(
        self,
        dwell_seconds:         float = 60.0,
        movement_threshold_px: float = 30.0,
        classes:               Optional[List[str]] = None,
        dedup_window_s:        float = 60.0,
        severity:              Severity = Severity.MEDIUM,
        camera_id:             str = "unknown",
    ) -> None:
        self._dwell_s    = dwell_seconds
        self._move_thr   = movement_threshold_px
        self._classes    = set(classes or ["person"])
        self._dedup_w    = dedup_window_s
        self._severity   = severity
        self._camera     = camera_id
        self._state: Dict[int, _LoiterState] = {}

        logger.info(
            f"LoiteringDetector ready — dwell={dwell_seconds}s, "
            f"move_threshold={movement_threshold_px}px, classes={self._classes}"
        )

    def update(
        self,
        tracks:    List[Track],
        frame_id:  int,
        camera_id: Optional[str] = None,
    ) -> List[Event]:
        cam    = camera_id or self._camera
        now    = time.monotonic()
        events: List[Event] = []

        active_ids: Set[int] = {t.track_id for t in tracks}

        # Prune disappeared tracks
        for tid in list(self._state.keys()):
            if tid not in active_ids:
                del self._state[tid]

        for track in tracks:
            if track.class_name not in self._classes:
                continue

            cx, cy = track.center
            tid    = track.track_id

            if tid not in self._state:
                # First time we see this track -- start clock
                self._state[tid] = _LoiterState(
                    seed_cx=cx, seed_cy=cy, first_seen=now
                )
                continue

            s = self._state[tid]

            # Check movement from seed centroid
            dist = ((cx - s.seed_cx) ** 2 + (cy - s.seed_cy) ** 2) ** 0.5
            if dist > self._move_thr:
                # Track moved -- reset dwell clock
                s.seed_cx    = cx
                s.seed_cy    = cy
                s.first_seen = now
                s.alerted    = False
                continue

            # Track is stationary -- check dwell time
            dwell = now - s.first_seen
            if dwell >= self._dwell_s:
                # Dedup: only alert once per dedup window
                if (now - s.alert_time) >= self._dedup_w:
                    s.alerted    = True
                    s.alert_time = now
                    events.append(Event(
                        event_type = EventType.LOITERING,
                        severity   = self._severity,
                        camera_id  = cam,
                        track      = track,
                        frame_id   = frame_id,
                        zone_id    = None,
                        metadata   = {
                            "dwell_s":          round(dwell, 1),
                            "seed_cx":          round(s.seed_cx, 1),
                            "seed_cy":          round(s.seed_cy, 1),
                            "movement_from_seed": round(dist, 1),
                        },
                    ))
                    logger.info(
                        f"[{cam}] LOITERING  track={tid}  "
                        f"class={track.class_name}  dwell={dwell:.1f}s  "
                        f"pos=({cx:.0f},{cy:.0f})"
                    )

        return events

    def reset(self) -> None:
        """Clear all state (call on camera change / stream restart)."""
        self._state.clear()
