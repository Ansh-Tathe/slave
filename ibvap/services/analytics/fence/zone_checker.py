"""
services/analytics/fence/zone_checker.py
==========================================
IBVAP P2 — Polygon zone containment + dwell-time tracking.

Algorithm: Ray-casting point-in-polygon (O(n) for n vertices).
Coordinates: normalised [0,1] x [0,1] — resolution-independent.

State: per-track, per-zone dwell timers (first-seen timestamp).
A DWELL event is emitted when a track's continuous dwell exceeds the threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from services.detect_track.models import Track
from services.event_engine.models  import Event, EventType, Severity


# Polygon vertex: normalised (x, y) ∈ [0,1]
Point  = Tuple[float, float]
Polygon = List[Point]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def point_in_polygon(px: float, py: float, poly: Polygon) -> bool:
    """
    Ray-casting algorithm.  Returns True if (px, py) is inside poly.
    poly must be a list of (x, y) tuples in normalised coordinates.
    """
    n      = len(poly)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def pixel_to_norm(
    px: float, py: float,
    frame_w: int, frame_h: int,
) -> Point:
    """Convert pixel (x,y) to normalised [0,1] coords."""
    return (px / frame_w, py / frame_h)


# ── Per-zone config ───────────────────────────────────────────────────────────

@dataclass
class ZoneConfig:
    zone_id:        str
    name:           str
    polygon:        Polygon          # normalised points
    on_enter:       bool = True
    on_exit:        bool = False
    dwell_seconds:  float = 0.0      # 0 = no dwell alert
    classes:        List[str] = field(default_factory=lambda: ["person"])
    severity:       Severity = Severity.HIGH
    enabled:        bool = True


# ── State tracker ─────────────────────────────────────────────────────────────

@dataclass
class _TrackZoneState:
    inside:         bool  = False
    first_seen:     float = 0.0   # monotonic time when track entered zone
    dwell_alerted:  bool  = False # True once the dwell alert has fired


class ZoneChecker:
    """
    Stateful zone-containment checker.

    Call update() every frame with the current track list.
    It compares against the previous frame's state to detect
    ENTER / EXIT / DWELL transitions and emits Event objects.

    Parameters
    ----------
    zones       : list of ZoneConfig (loaded from zones.yaml)
    frame_w, frame_h : frame resolution (to convert pixel → norm)
    camera_id   : used to populate Event.camera_id
    """

    def __init__(
        self,
        zones:    List[ZoneConfig],
        frame_w:  int,
        frame_h:  int,
        camera_id: str,
    ) -> None:
        self._zones    = [z for z in zones if z.enabled]
        self._fw       = frame_w
        self._fh       = frame_h
        self._camera   = camera_id
        # state[zone_id][track_id]
        self._state:   Dict[str, Dict[int, _TrackZoneState]] = {
            z.zone_id: {} for z in self._zones
        }

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        tracks:   List[Track],
        frame_id: int,
    ) -> List[Event]:
        """
        Check all tracks against all zones.

        Returns a flat list of Events (ZONE_ENTER / ZONE_EXIT / DWELL).
        """
        events: List[Event] = []
        now = time.monotonic()

        active_ids = {t.track_id for t in tracks}

        for zone in self._zones:
            zone_state = self._state[zone.zone_id]

            # Remove state for tracks no longer visible
            gone = [tid for tid in zone_state if tid not in active_ids]
            for tid in gone:
                s = zone_state.pop(tid)
                # Emit EXIT if the track was inside when it disappeared
                # (only if on_exit is configured)
                # We don't emit here — track disappeared, might just be occlusion
                pass

            for track in tracks:
                # Filter by class
                if track.class_name not in zone.classes:
                    continue

                cx_norm, cy_norm = pixel_to_norm(
                    track.center[0], track.center[1],
                    self._fw, self._fh,
                )
                currently_inside = point_in_polygon(cx_norm, cy_norm, zone.polygon)

                tid = track.track_id
                s   = zone_state.get(tid, _TrackZoneState())

                # ── ENTER ────────────────────────────────────────────────────
                if currently_inside and not s.inside:
                    s.inside       = True
                    s.first_seen   = now
                    s.dwell_alerted = False
                    if zone.on_enter:
                        events.append(Event(
                            event_type  = EventType.ZONE_ENTER,
                            severity    = zone.severity,
                            camera_id   = self._camera,
                            track       = track,
                            frame_id    = frame_id,
                            zone_id     = zone.zone_id,
                            metadata    = {"zone_name": zone.name},
                        ))
                        logger.debug(
                            f"[{self._camera}] ZONE_ENTER  "
                            f"track={tid} zone={zone.zone_id}"
                        )

                # ── EXIT ─────────────────────────────────────────────────────
                elif not currently_inside and s.inside:
                    s.inside = False
                    if zone.on_exit:
                        dwell = now - s.first_seen
                        events.append(Event(
                            event_type  = EventType.ZONE_EXIT,
                            severity    = Severity.LOW,
                            camera_id   = self._camera,
                            track       = track,
                            frame_id    = frame_id,
                            zone_id     = zone.zone_id,
                            metadata    = {
                                "zone_name": zone.name,
                                "dwell_s":   round(dwell, 2),
                            },
                        ))

                # ── DWELL (loitering) ─────────────────────────────────────
                elif currently_inside and s.inside and zone.dwell_seconds > 0:
                    dwell = now - s.first_seen
                    if dwell >= zone.dwell_seconds and not s.dwell_alerted:
                        s.dwell_alerted = True
                        events.append(Event(
                            event_type  = EventType.DWELL,
                            severity    = zone.severity,
                            camera_id   = self._camera,
                            track       = track,
                            frame_id    = frame_id,
                            zone_id     = zone.zone_id,
                            metadata    = {
                                "zone_name": zone.name,
                                "dwell_s":   round(dwell, 2),
                            },
                        ))
                        logger.debug(
                            f"[{self._camera}] DWELL  "
                            f"track={tid} zone={zone.zone_id} "
                            f"dwell={dwell:.1f}s"
                        )

                zone_state[tid] = s

        return events

    def reset_track(self, track_id: int) -> None:
        """Remove all state for a specific track (call on track loss)."""
        for zone_state in self._state.values():
            zone_state.pop(track_id, None)
