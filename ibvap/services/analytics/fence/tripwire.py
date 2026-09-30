"""
services/analytics/fence/tripwire.py
======================================
IBVAP P2 — Tripwire line-crossing detector.

Algorithm:
  For each track, compare its centre position between consecutive frames.
  A crossing is detected when the segment (prev_centre → curr_centre) intersects
  the tripwire segment.  Direction is determined by the sign of the cross product.

Coordinates: normalised [0,1] x [0,1].

Direction vocabulary
--------------------
  "both"          — fire on any crossing
  "north_to_south" — top→bottom  (y increasing)
  "south_to_north" — bottom→top  (y decreasing)
  "east_to_west"   — right→left  (x decreasing)
  "west_to_east"   — left→right  (x increasing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from services.detect_track.models import Track
from services.event_engine.models  import Event, EventType, Severity

Point = Tuple[float, float]


# ── Geometry ──────────────────────────────────────────────────────────────────

def _cross2d(o: Point, a: Point, b: Point) -> float:
    """2-D cross product of OA × OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(
    p1: Point, p2: Point,   # moving segment (prev → curr centre)
    p3: Point, p4: Point,   # tripwire segment (start → end)
) -> bool:
    """True if segment p1-p2 crosses segment p3-p4 (proper intersection)."""
    d1 = _cross2d(p3, p4, p1)
    d2 = _cross2d(p3, p4, p2)
    d3 = _cross2d(p1, p2, p3)
    d4 = _cross2d(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False   # ignore collinear edge cases (rare at video FPS)


def crossing_direction(
    prev: Point,
    curr: Point,
    wire_start: Point,
    wire_end:   Point,
) -> str:
    """
    Returns the crossing direction as a human-readable string.
    Uses the sign of the cross product of (wire vector) × (movement vector).
    """
    # Wire direction vector
    wx = wire_end[0] - wire_start[0]
    wy = wire_end[1] - wire_start[1]
    # Movement vector
    mx = curr[0] - prev[0]
    my = curr[1] - prev[1]
    # Cross product z-component
    cross = wx * my - wy * mx

    # Determine cardinal description from movement
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if abs(dx) >= abs(dy):
        return "west_to_east" if dx > 0 else "east_to_west"
    else:
        return "north_to_south" if dy > 0 else "south_to_north"


def _direction_matches(actual: str, configured: str) -> bool:
    if configured == "both":
        return True
    return actual == configured


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TripwireConfig:
    wire_id:    str
    name:       str
    start:      Point           # normalised (x, y)
    end:        Point           # normalised (x, y)
    direction:  str = "both"    # both | north_to_south | … (see module docstring)
    classes:    List[str] = field(default_factory=lambda: ["person"])
    severity:   Severity = Severity.CRITICAL
    enabled:    bool = True


# ── Tripwire checker ──────────────────────────────────────────────────────────

class TripwireChecker:
    """
    Stateful tripwire crossing detector.

    Maintains the previous normalised centre for each tracked object.
    Call update() every frame; it returns crossing Events.
    """

    def __init__(
        self,
        wires:     List[TripwireConfig],
        frame_w:   int,
        frame_h:   int,
        camera_id: str,
    ) -> None:
        self._wires    = [w for w in wires if w.enabled]
        self._fw       = frame_w
        self._fh       = frame_h
        self._camera   = camera_id
        # Previous normalised centre per track_id
        self._prev:    Dict[int, Point] = {}

    def update(
        self,
        tracks:   List[Track],
        frame_id: int,
    ) -> List[Event]:
        events: List[Event] = []
        active_ids = {t.track_id for t in tracks}

        # Prune lost tracks
        for tid in list(self._prev.keys()):
            if tid not in active_ids:
                del self._prev[tid]

        for track in tracks:
            # Normalise current centre
            cx = track.center[0] / self._fw
            cy = track.center[1] / self._fh
            curr: Point = (cx, cy)

            prev = self._prev.get(track.track_id)
            if prev is not None:
                for wire in self._wires:
                    if track.class_name not in wire.classes:
                        continue

                    if segments_intersect(prev, curr, wire.start, wire.end):
                        direction = crossing_direction(
                            prev, curr, wire.start, wire.end
                        )
                        if not _direction_matches(direction, wire.direction):
                            continue

                        events.append(Event(
                            event_type = EventType.TRIPWIRE_CROSS,
                            severity   = wire.severity,
                            camera_id  = self._camera,
                            track      = track,
                            frame_id   = frame_id,
                            zone_id    = wire.wire_id,
                            metadata   = {
                                "wire_name": wire.name,
                                "direction": direction,
                                "configured_direction": wire.direction,
                            },
                        ))
                        logger.debug(
                            f"[{self._camera}] TRIPWIRE_CROSS  "
                            f"track={track.track_id} "
                            f"wire={wire.wire_id} dir={direction}"
                        )

            self._prev[track.track_id] = curr

        return events
