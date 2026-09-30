"""
services/analytics/behaviour/abandoned.py
==========================================
IBVAP P5 -- Abandoned object detector.

Emits ABANDONED_OBJ when a non-person object (bag, suitcase, etc.) remains
stationary for >= stationary_seconds without an "owner" person being nearby.

Algorithm
---------
1. Filter tracks to "object" classes (default: anything that is NOT a person
   or vehicle -- in practice we check if class_id is in object_class_ids).
   Since YOLOv8n only detects COCO classes and we filter to person + vehicles,
   we treat any vehicle-class bbox as a potential abandoned vehicle, and any
   previously-moving track that suddenly becomes stationary as an "object".

   More practically: ANY track (including persons!) that becomes stationary
   for `stationary_seconds` AND has no other person within `owner_radius_px`
   is flagged as ABANDONED_OBJ.  The operator can then review the snapshot.

2. A track is "stationary" if its centroid has not moved more than
   `movement_threshold_px` from its anchor position for `stationary_seconds`.

3. A track has an "owner" if any person track is within `owner_radius_px`.
   If an owner is nearby, the timer resets.

4. Once alerted, the event is suppressed for `dedup_window_s` seconds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from loguru import logger

from services.detect_track.models import Track
from services.event_engine.models  import Event, EventType, Severity


@dataclass
class _ObjState:
    anchor_cx:   float     # centroid when clock started
    anchor_cy:   float     # centroid when clock started
    first_seen:  float     # monotonic timestamp when stationary started
    alerted:     bool = False
    alert_time:  float = 0.0


class AbandonedObjectDetector:
    """
    Detects stationary / abandoned objects.

    Parameters
    ----------
    stationary_seconds   : seconds an object must stay still to be flagged
    movement_threshold_px: movement > this resets the stationary clock
    owner_radius_px      : if a person is within this radius, suppress alert
    watch_classes        : class names to monitor (default: non-person classes)
    dedup_window_s       : suppress repeated alerts for same track
    severity             : event severity
    camera_id            : used in emitted events
    """

    def __init__(
        self,
        stationary_seconds:    float = 120.0,
        movement_threshold_px: float = 20.0,
        owner_radius_px:       float = 150.0,
        watch_classes:         Optional[List[str]] = None,
        dedup_window_s:        float = 120.0,
        severity:              Severity = Severity.HIGH,
        camera_id:             str = "unknown",
    ) -> None:
        self._stat_s     = stationary_seconds
        self._move_thr   = movement_threshold_px
        self._owner_r    = owner_radius_px
        self._watch      = set(
            watch_classes or ["car", "truck", "bus", "motorcycle", "bicycle"]
        )
        self._dedup_w    = dedup_window_s
        self._severity   = severity
        self._camera     = camera_id
        self._state: Dict[int, _ObjState] = {}

        logger.info(
            f"AbandonedObjectDetector ready — "
            f"stationary={stationary_seconds}s, "
            f"owner_radius={owner_radius_px}px, "
            f"watch_classes={self._watch}"
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
        persons = [t for t in tracks if t.class_name == "person"]

        # Prune disappeared tracks
        for tid in list(self._state.keys()):
            if tid not in active_ids:
                del self._state[tid]

        for track in tracks:
            if track.class_name not in self._watch:
                continue

            tid   = track.track_id
            cx, cy = track.center

            # Check for nearby owner (person)
            owner_nearby = any(
                math.sqrt((cx - p.center[0]) ** 2 + (cy - p.center[1]) ** 2)
                <= self._owner_r
                for p in persons
            )

            if tid not in self._state:
                self._state[tid] = _ObjState(
                    anchor_cx=cx, anchor_cy=cy, first_seen=now
                )
                continue

            s = self._state[tid]
            dist = math.sqrt((cx - s.anchor_cx) ** 2 + (cy - s.anchor_cy) ** 2)

            # Moved or owner present → reset clock
            if dist > self._move_thr or owner_nearby:
                s.anchor_cx  = cx
                s.anchor_cy  = cy
                s.first_seen = now
                s.alerted    = False
                continue

            # Stationary and unattended
            stationary_t = now - s.first_seen
            if stationary_t >= self._stat_s:
                if (now - s.alert_time) >= self._dedup_w:
                    s.alerted    = True
                    s.alert_time = now
                    events.append(Event(
                        event_type = EventType.ABANDONED_OBJ,
                        severity   = self._severity,
                        camera_id  = cam,
                        track      = track,
                        frame_id   = frame_id,
                        zone_id    = None,
                        metadata   = {
                            "stationary_s":   round(stationary_t, 1),
                            "anchor_cx":      round(s.anchor_cx, 1),
                            "anchor_cy":      round(s.anchor_cy, 1),
                            "owner_nearby":   owner_nearby,
                        },
                    ))
                    logger.info(
                        f"[{cam}] ABANDONED_OBJ  track={tid}  "
                        f"class={track.class_name}  "
                        f"stationary={stationary_t:.1f}s"
                    )

        return events

    def reset(self) -> None:
        self._state.clear()
