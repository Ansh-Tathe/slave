"""
tests/test_p2_fence.py
========================
P2 — Unit tests for fence analytics (zone checker, tripwire, fence engine).

All tests are pure geometry + state logic — NO GPU, NO video needed.
Synthetic Track objects are built directly.

Run:
    pytest tests/test_p2_fence.py -v
"""

from __future__ import annotations

import time
from typing import List

import pytest

from services.analytics.fence.zone_checker import (
    ZoneChecker, ZoneConfig, point_in_polygon, pixel_to_norm,
)
from services.analytics.fence.tripwire import (
    TripwireChecker, TripwireConfig, segments_intersect, crossing_direction,
)
from services.detect_track.models import Track
from services.event_engine.models  import EventType, Severity


# ── Track factory ─────────────────────────────────────────────────────────────

def make_track(
    track_id: int = 1,
    cx: float = 320.0,  # pixel centre x
    cy: float = 240.0,  # pixel centre y
    class_name: str = "person",
) -> Track:
    """Build a minimal Track with centre at (cx, cy) in a 640x480 frame."""
    hw, hh = 30.0, 60.0
    return Track(
        track_id   = track_id,
        bbox       = (cx - hw, cy - hh, cx + hw, cy + hh),
        conf       = 0.90,
        class_id   = 0,
        class_name = class_name,
        frame_id   = 1,
        camera_id  = "test",
    )


# ── Frame dimensions used throughout ─────────────────────────────────────────
FW, FH = 640, 480


# =============================================================================
# 1. Geometry helpers
# =============================================================================

class TestPointInPolygon:
    """Ray-casting correctness tests."""

    SQUARE = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]

    def test_centre_inside(self):
        assert point_in_polygon(0.5, 0.5, self.SQUARE) is True

    def test_corner_outside(self):
        assert point_in_polygon(0.0, 0.0, self.SQUARE) is False

    def test_edge_case_left(self):
        # Just inside left edge
        assert point_in_polygon(0.11, 0.5, self.SQUARE) is True

    def test_outside_right(self):
        assert point_in_polygon(0.95, 0.5, self.SQUARE) is False

    def test_triangle(self):
        tri = [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert point_in_polygon(0.5, 0.7, tri) is True
        assert point_in_polygon(0.1, 0.1, tri) is False

    def test_degenerate_single_point(self):
        # Degenerate polygon — should not crash
        result = point_in_polygon(0.5, 0.5, [(0.5, 0.5)])
        assert isinstance(result, bool)


class TestSegmentsIntersect:
    """Tripwire crossing geometry."""

    def test_clear_crossing(self):
        # Horizontal movement crosses vertical wire
        assert segments_intersect(
            (0.3, 0.5), (0.7, 0.5),   # movement: left → right
            (0.5, 0.0), (0.5, 1.0),   # wire: vertical centre
        ) is True

    def test_no_crossing_same_side(self):
        assert segments_intersect(
            (0.1, 0.5), (0.4, 0.5),   # movement stays left of wire
            (0.5, 0.0), (0.5, 1.0),
        ) is False

    def test_parallel_no_crossing(self):
        # Both segments are parallel horizontal lines
        assert segments_intersect(
            (0.0, 0.3), (1.0, 0.3),
            (0.0, 0.7), (1.0, 0.7),
        ) is False

    def test_t_intersection_no_proper_cross(self):
        # Movement ends exactly on the wire — not a proper crossing
        assert segments_intersect(
            (0.3, 0.5), (0.5, 0.5),
            (0.5, 0.0), (0.5, 1.0),
        ) is False


class TestCrossingDirection:
    def test_west_to_east(self):
        d = crossing_direction(
            (0.3, 0.5), (0.7, 0.5),  # moving right
            (0.5, 0.0), (0.5, 1.0),
        )
        assert d == "west_to_east"

    def test_north_to_south(self):
        d = crossing_direction(
            (0.5, 0.2), (0.5, 0.8),  # moving down
            (0.0, 0.5), (1.0, 0.5),
        )
        assert d == "north_to_south"


# =============================================================================
# 2. ZoneChecker
# =============================================================================

def _make_zone_checker(dwell_s: float = 0.0) -> ZoneChecker:
    zones = [ZoneConfig(
        zone_id       = "z1",
        name          = "Test Zone",
        polygon       = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        on_enter      = True,
        on_exit       = True,
        dwell_seconds = dwell_s,
        classes       = ["person"],
        severity      = Severity.HIGH,
    )]
    return ZoneChecker(zones, FW, FH, "test")


class TestZoneChecker:
    def test_enter_emits_event(self):
        checker = _make_zone_checker()
        # Track at x=160 (norm=0.25) → inside zone [0, 0.5]
        t = make_track(cx=160.0, cy=240.0)
        events = checker.update([t], frame_id=1)
        types = [e.event_type for e in events]
        assert EventType.ZONE_ENTER in types

    def test_no_event_when_already_inside(self):
        checker = _make_zone_checker()
        t = make_track(cx=160.0, cy=240.0)
        checker.update([t], frame_id=1)  # enter
        events2 = checker.update([t], frame_id=2)  # still inside
        assert not any(e.event_type == EventType.ZONE_ENTER for e in events2)

    def test_exit_emits_event(self):
        checker = _make_zone_checker()
        t_in  = make_track(cx=160.0, cy=240.0)  # inside
        t_out = make_track(cx=480.0, cy=240.0)  # outside (norm=0.75)
        checker.update([t_in],  frame_id=1)
        events = checker.update([t_out], frame_id=2)
        types = [e.event_type for e in events]
        assert EventType.ZONE_EXIT in types

    def test_outside_no_event(self):
        checker = _make_zone_checker()
        t = make_track(cx=480.0, cy=240.0)  # outside zone
        events = checker.update([t], frame_id=1)
        assert events == []

    def test_dwell_fires_after_threshold(self):
        checker = _make_zone_checker(dwell_s=0.05)  # 50 ms threshold
        t = make_track(cx=160.0, cy=240.0)
        checker.update([t], frame_id=1)   # enter
        time.sleep(0.07)                   # wait past threshold
        events = checker.update([t], frame_id=2)
        types = [e.event_type for e in events]
        assert EventType.DWELL in types

    def test_dwell_fires_only_once(self):
        checker = _make_zone_checker(dwell_s=0.05)
        t = make_track(cx=160.0, cy=240.0)
        checker.update([t], frame_id=1)
        time.sleep(0.07)
        events_a = checker.update([t], frame_id=2)
        events_b = checker.update([t], frame_id=3)  # still inside
        dwell_a = [e for e in events_a if e.event_type == EventType.DWELL]
        dwell_b = [e for e in events_b if e.event_type == EventType.DWELL]
        assert len(dwell_a) == 1
        assert len(dwell_b) == 0   # should NOT fire again

    def test_wrong_class_ignored(self):
        checker = _make_zone_checker()   # classes=["person"]
        t = make_track(cx=160.0, cy=240.0, class_name="car")
        events = checker.update([t], frame_id=1)
        assert events == []

    def test_event_has_correct_camera(self):
        checker = _make_zone_checker()
        t = make_track(cx=160.0, cy=240.0)
        events = checker.update([t], frame_id=1)
        assert all(e.camera_id == "test" for e in events)

    def test_multiple_tracks_independent(self):
        checker = _make_zone_checker()
        t1 = make_track(track_id=1, cx=160.0, cy=240.0)  # inside
        t2 = make_track(track_id=2, cx=480.0, cy=240.0)  # outside
        events = checker.update([t1, t2], frame_id=1)
        # Only t1 should trigger ZONE_ENTER
        enter_ids = [e.track.track_id for e in events
                     if e.event_type == EventType.ZONE_ENTER]
        assert 1 in enter_ids
        assert 2 not in enter_ids


# =============================================================================
# 3. TripwireChecker
# =============================================================================

def _make_tripwire_checker(direction: str = "both") -> TripwireChecker:
    wires = [TripwireConfig(
        wire_id   = "w1",
        name      = "Centre Wire",
        start     = (0.5, 0.0),
        end       = (0.5, 1.0),
        direction = direction,
        classes   = ["person"],
        severity  = Severity.CRITICAL,
    )]
    return TripwireChecker(wires, FW, FH, "test")


class TestTripwireChecker:
    def test_crossing_fires_event(self):
        checker = _make_tripwire_checker()
        t_left  = make_track(cx=200.0, cy=240.0)   # norm x=0.3125
        t_right = make_track(cx=440.0, cy=240.0)   # norm x=0.6875
        checker.update([t_left],  frame_id=1)   # establish prev position
        events = checker.update([t_right], frame_id=2)  # cross the wire
        # NOTE: track_id must be same for state to be tracked
        # Use same track_id:
        checker2 = _make_tripwire_checker()
        t1 = make_track(track_id=5, cx=200.0, cy=240.0)
        t2 = make_track(track_id=5, cx=440.0, cy=240.0)
        checker2.update([t1], frame_id=1)
        evts = checker2.update([t2], frame_id=2)
        types = [e.event_type for e in evts]
        assert EventType.TRIPWIRE_CROSS in types

    def test_no_crossing_stays_one_side(self):
        checker = _make_tripwire_checker()
        t1 = make_track(track_id=1, cx=100.0, cy=240.0)
        t2 = make_track(track_id=1, cx=200.0, cy=240.0)
        checker.update([t1], frame_id=1)
        evts = checker.update([t2], frame_id=2)
        assert not any(e.event_type == EventType.TRIPWIRE_CROSS for e in evts)

    def test_first_frame_no_event(self):
        """No event on first frame — no previous position to compare."""
        checker = _make_tripwire_checker()
        t = make_track(track_id=1, cx=320.0, cy=240.0)
        evts = checker.update([t], frame_id=1)
        assert evts == []

    def test_direction_filter_blocks_wrong_way(self):
        checker = _make_tripwire_checker(direction="east_to_west")
        t1 = make_track(track_id=1, cx=200.0, cy=240.0)  # left of wire
        t2 = make_track(track_id=1, cx=440.0, cy=240.0)  # right of wire
        checker.update([t1], frame_id=1)
        # west_to_east crossing — should NOT match "east_to_west"
        evts = checker.update([t2], frame_id=2)
        assert not any(e.event_type == EventType.TRIPWIRE_CROSS for e in evts)

    def test_event_metadata_has_direction(self):
        checker = _make_tripwire_checker()
        t1 = make_track(track_id=1, cx=200.0, cy=240.0)
        t2 = make_track(track_id=1, cx=440.0, cy=240.0)
        checker.update([t1], frame_id=1)
        evts = checker.update([t2], frame_id=2)
        for e in evts:
            if e.event_type == EventType.TRIPWIRE_CROSS:
                assert "direction" in e.metadata
