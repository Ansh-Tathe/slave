"""
tests/test_p5_behaviour.py
============================
IBVAP P5 -- Unit tests for suspicious-activity rules.

Coverage:
  LoiteringDetector   -- dwell clock, movement reset, dedup
  SpeedDetector       -- velocity estimation, RUNNING, WRONG_WAY
  GatheringDetector   -- cluster detection, duration gate, dedup
  AbandonedObjectDetector -- stationary clock, owner suppression, dedup
  BehaviourEngine     -- orchestration, all rules combined
"""

from __future__ import annotations

import math
import time
from typing import List

import pytest

from services.analytics.behaviour.abandoned    import AbandonedObjectDetector
from services.analytics.behaviour.behaviour_engine import BehaviourEngine
from services.analytics.behaviour.gathering    import GatheringDetector, _connected_components
from services.analytics.behaviour.loitering    import LoiteringDetector
from services.analytics.behaviour.speed        import SpeedDetector
from services.detect_track.models              import Track
from services.event_engine.models              import EventType, Severity


# =============================================================================
# Fixtures / helpers
# =============================================================================

def _track(
    track_id: int,
    cx: float = 100.0,
    cy: float = 100.0,
    class_name: str = "person",
) -> Track:
    hw = 30.0
    return Track(
        track_id   = track_id,
        bbox       = (cx - hw, cy - hw, cx + hw, cy + hw),
        conf       = 0.85,
        class_id   = 0 if class_name == "person" else 2,
        class_name = class_name,
        frame_id   = 1,
        camera_id  = "test",
    )


def _run_n_frames(
    detector,
    tracks_list: List[List[Track]],
    start_frame: int = 1,
) -> List:
    """Run detector for multiple frames, collecting all events."""
    all_events = []
    for i, tracks in enumerate(tracks_list):
        all_events.extend(detector.update(tracks, frame_id=start_frame + i))
    return all_events


# =============================================================================
# LoiteringDetector
# =============================================================================

class TestLoiteringDetector:

    def test_no_event_before_threshold(self):
        det   = LoiteringDetector(dwell_seconds=60.0, dedup_window_s=0)
        track = _track(1)
        # Only 1 frame -- no event
        events = det.update([track], frame_id=1)
        assert events == []

    def test_event_fires_after_dwell(self):
        det   = LoiteringDetector(dwell_seconds=0.05, dedup_window_s=0.0)
        track = _track(1, cx=100, cy=100)
        det.update([track], frame_id=1)
        time.sleep(0.1)
        events = det.update([track], frame_id=2)
        assert len(events) == 1
        assert events[0].event_type == EventType.LOITERING

    def test_movement_resets_dwell_clock(self):
        det = LoiteringDetector(dwell_seconds=0.05, movement_threshold_px=20.0, dedup_window_s=0.0)
        t   = _track(1, cx=100, cy=100)
        det.update([t], frame_id=1)
        time.sleep(0.06)
        # Move track significantly
        t2 = _track(1, cx=200, cy=200)
        events = det.update([t2], frame_id=2)
        assert events == [], "Movement should reset clock -- no LOITERING yet"

    def test_slight_movement_does_not_reset(self):
        det = LoiteringDetector(dwell_seconds=0.05, movement_threshold_px=50.0, dedup_window_s=0.0)
        t1  = _track(1, cx=100, cy=100)
        det.update([t1], frame_id=1)
        time.sleep(0.08)
        # Slight movement (< 50px threshold)
        t2  = _track(1, cx=110, cy=105)
        events = det.update([t2], frame_id=2)
        assert len(events) == 1, "Slight movement should NOT reset clock"

    def test_dedup_suppresses_repeat(self):
        det = LoiteringDetector(dwell_seconds=0.01, dedup_window_s=60.0)
        t   = _track(1)
        det.update([t], frame_id=1)
        time.sleep(0.02)
        ev1 = det.update([t], frame_id=2)
        ev2 = det.update([t], frame_id=3)
        assert len(ev1) == 1
        assert len(ev2) == 0, "Dedup should suppress second event"

    def test_non_person_class_ignored(self):
        det    = LoiteringDetector(classes=["person"], dwell_seconds=0.01, dedup_window_s=0.0)
        t_car  = _track(1, class_name="car")
        det.update([t_car], frame_id=1)
        time.sleep(0.02)
        events = det.update([t_car], frame_id=2)
        assert events == [], "Car should not trigger LOITERING (person-only)"

    def test_prune_disappeared_track(self):
        det = LoiteringDetector(dwell_seconds=0.01, dedup_window_s=0.0)
        t   = _track(1)
        det.update([t], frame_id=1)
        time.sleep(0.02)
        # Disappear
        det.update([], frame_id=2)
        assert 1 not in det._state, "State should be pruned when track disappears"

    def test_metadata_keys(self):
        det = LoiteringDetector(dwell_seconds=0.01, dedup_window_s=0.0)
        t   = _track(1)
        det.update([t], frame_id=1)
        time.sleep(0.02)
        events = det.update([t], frame_id=2)
        assert len(events) == 1
        meta = events[0].metadata
        assert "dwell_s" in meta and "seed_cx" in meta


# =============================================================================
# SpeedDetector
# =============================================================================

class TestSpeedDetector:

    def _move_track(self, det: SpeedDetector, track_id: int,
                    positions: List, class_name: str = "person", dt: float = 0.1) -> List:
        all_events = []
        for i, (cx, cy) in enumerate(positions):
            t = _track(track_id, cx=cx, cy=cy, class_name=class_name)
            all_events.extend(det.update([t], frame_id=i + 1))
            time.sleep(dt)
        return all_events

    def test_no_event_when_slow(self):
        det = SpeedDetector(run_threshold_px_per_s=500.0, min_frames=0, dedup_window_s=0)
        # Move 5px over ~0.1s = 50px/s (well below 500 threshold)
        events = self._move_track(det, 1, [(100,100),(102,100),(104,100),(106,100)], dt=0.1)
        run_events = [e for e in events if e.event_type == EventType.RUNNING]
        assert run_events == []

    def test_running_event_fires(self):
        det = SpeedDetector(run_threshold_px_per_s=50.0, min_frames=2, dedup_window_s=0.0, ema_alpha=1.0)
        # Move 100px in 0.05s = 2000px/s
        positions = [(100,100),(200,100),(300,100),(400,100)]
        events = self._move_track(det, 1, positions, dt=0.05)
        run_events = [e for e in events if e.event_type == EventType.RUNNING]
        assert len(run_events) >= 1

    def test_running_only_for_persons(self):
        det = SpeedDetector(run_threshold_px_per_s=10.0, running_classes=["person"],
                            min_frames=2, dedup_window_s=0.0, ema_alpha=1.0)
        positions = [(100,100),(200,100),(300,100)]
        events = self._move_track(det, 1, positions, class_name="car", dt=0.05)
        run_events = [e for e in events if e.event_type == EventType.RUNNING]
        assert run_events == [], "Cars should not trigger RUNNING"

    def test_get_speed_returns_float(self):
        det = SpeedDetector()
        assert det.get_speed(999) == 0.0
        t = _track(1, cx=100, cy=100)
        det.update([t], frame_id=1)
        time.sleep(0.05)
        t2 = _track(1, cx=200, cy=100)
        det.update([t2], frame_id=2)
        assert det.get_speed(1) >= 0.0

    def test_wrong_way_fires(self):
        det = SpeedDetector(
            allowed_direction="right",      # allowed = rightward (0 deg)
            direction_tolerance_deg=45.0,
            wrong_way_classes=["car"],
            min_frames=2,
            dedup_window_s=0.0,
            ema_alpha=1.0,
            run_threshold_px_per_s=99999,   # disable RUNNING
        )
        # Move car leftward (180 deg) -- wrong way
        positions = [(400,100),(300,100),(200,100),(100,100)]
        events = self._move_track(det, 1, positions, class_name="car", dt=0.05)
        ww = [e for e in events if e.event_type == EventType.WRONG_WAY]
        assert len(ww) >= 1

    def test_no_wrong_way_when_any(self):
        det = SpeedDetector(allowed_direction="any", min_frames=2,
                            dedup_window_s=0.0, ema_alpha=1.0)
        positions = [(400,100),(300,100),(200,100)]
        events = self._move_track(det, 1, positions, class_name="car", dt=0.05)
        ww = [e for e in events if e.event_type == EventType.WRONG_WAY]
        assert ww == [], "No wrong-way when allowed_direction='any'"

    def test_state_pruned_on_disappear(self):
        det = SpeedDetector()
        t = _track(1)
        det.update([t], frame_id=1)
        det.update([], frame_id=2)
        assert 1 not in det._state


# =============================================================================
# GatheringDetector
# =============================================================================

class TestGatheringDetector:

    def test_connected_components_single_cluster(self):
        tracks = [_track(i, cx=100 + i * 10, cy=100) for i in range(5)]
        comps  = _connected_components(tracks, proximity_px=50.0)
        assert len(comps) == 1
        assert len(comps[0]) == 5

    def test_connected_components_two_clusters(self):
        close  = [_track(i, cx=100 + i * 10, cy=100) for i in range(3)]
        far    = [_track(i + 10, cx=600 + i * 10, cy=100) for i in range(3)]
        comps  = _connected_components(close + far, proximity_px=50.0)
        assert len(comps) == 2

    def test_no_event_below_min_count(self):
        det = GatheringDetector(min_count=5, duration_s=0.0, dedup_window_s=0.0)
        tracks = [_track(i, cx=100 + i * 5, cy=100) for i in range(3)]
        events = det.update(tracks, frame_id=1)
        assert events == []

    def test_event_fires_after_duration(self):
        det = GatheringDetector(min_count=3, proximity_px=100.0,
                                duration_s=0.05, dedup_window_s=0.0)
        tracks = [_track(i, cx=100 + i * 10, cy=100) for i in range(5)]
        det.update(tracks, frame_id=1)
        time.sleep(0.1)
        events = det.update(tracks, frame_id=2)
        gather = [e for e in events if e.event_type == EventType.GATHERING]
        assert len(gather) >= 1

    def test_no_event_before_duration(self):
        det = GatheringDetector(min_count=3, proximity_px=100.0,
                                duration_s=60.0, dedup_window_s=0.0)
        tracks = [_track(i, cx=100 + i * 10, cy=100) for i in range(5)]
        events = det.update(tracks, frame_id=1)
        assert events == []

    def test_dedup_suppresses_repeat(self):
        det = GatheringDetector(min_count=3, proximity_px=100.0,
                                duration_s=0.01, dedup_window_s=60.0)
        tracks = [_track(i, cx=100 + i * 10, cy=100) for i in range(5)]
        det.update(tracks, frame_id=1)
        time.sleep(0.02)
        ev1 = det.update(tracks, frame_id=2)
        ev2 = det.update(tracks, frame_id=3)
        g1  = [e for e in ev1 if e.event_type == EventType.GATHERING]
        g2  = [e for e in ev2 if e.event_type == EventType.GATHERING]
        assert len(g1) >= 1
        assert len(g2) == 0

    def test_metadata_keys(self):
        det = GatheringDetector(min_count=3, proximity_px=100.0,
                                duration_s=0.01, dedup_window_s=0.0)
        tracks = [_track(i, cx=100 + i * 10, cy=100) for i in range(4)]
        det.update(tracks, frame_id=1)
        time.sleep(0.02)
        events = det.update(tracks, frame_id=2)
        gather = [e for e in events if e.event_type == EventType.GATHERING]
        assert len(gather) >= 1
        meta = gather[0].metadata
        assert "cluster_size" in meta and "track_ids" in meta

    def test_non_person_ignored(self):
        det = GatheringDetector(min_count=3, proximity_px=100.0,
                                duration_s=0.01, classes=["person"], dedup_window_s=0.0)
        cars = [_track(i, cx=100 + i * 10, cy=100, class_name="car") for i in range(5)]
        det.update(cars, frame_id=1)
        time.sleep(0.02)
        events = det.update(cars, frame_id=2)
        g = [e for e in events if e.event_type == EventType.GATHERING]
        assert g == [], "Cars should not trigger GATHERING (person-only)"


# =============================================================================
# AbandonedObjectDetector
# =============================================================================

class TestAbandonedObjectDetector:

    def test_no_event_before_threshold(self):
        det = AbandonedObjectDetector(stationary_seconds=60.0, dedup_window_s=0)
        t   = _track(1, class_name="car")
        events = det.update([t], frame_id=1)
        assert events == []

    def test_event_fires_after_stationary(self):
        det = AbandonedObjectDetector(stationary_seconds=0.05,
                                      owner_radius_px=10.0, dedup_window_s=0.0)
        t   = _track(1, class_name="car", cx=400, cy=300)
        det.update([t], frame_id=1)
        time.sleep(0.1)
        events = det.update([t], frame_id=2)
        aband  = [e for e in events if e.event_type == EventType.ABANDONED_OBJ]
        assert len(aband) == 1

    def test_owner_nearby_suppresses(self):
        det    = AbandonedObjectDetector(stationary_seconds=0.05,
                                         owner_radius_px=200.0, dedup_window_s=0.0)
        car    = _track(1, class_name="car",    cx=400, cy=300)
        person = _track(2, class_name="person", cx=410, cy=305)  # within 200px
        det.update([car, person], frame_id=1)
        time.sleep(0.1)
        events = det.update([car, person], frame_id=2)
        aband  = [e for e in events if e.event_type == EventType.ABANDONED_OBJ]
        assert aband == [], "Owner nearby should suppress ABANDONED_OBJ"

    def test_movement_resets_clock(self):
        det = AbandonedObjectDetector(stationary_seconds=0.05,
                                      movement_threshold_px=20.0, dedup_window_s=0.0)
        t1  = _track(1, class_name="car", cx=400, cy=300)
        det.update([t1], frame_id=1)
        time.sleep(0.06)
        # Move the car
        t2  = _track(1, class_name="car", cx=450, cy=300)
        events = det.update([t2], frame_id=2)
        aband  = [e for e in events if e.event_type == EventType.ABANDONED_OBJ]
        assert aband == [], "Movement should reset stationary clock"

    def test_person_not_in_default_watch_classes(self):
        det = AbandonedObjectDetector(stationary_seconds=0.05, dedup_window_s=0.0)
        t   = _track(1, class_name="person")
        det.update([t], frame_id=1)
        time.sleep(0.1)
        events = det.update([t], frame_id=2)
        aband  = [e for e in events if e.event_type == EventType.ABANDONED_OBJ]
        assert aband == [], "Persons are not in default watch_classes"

    def test_dedup_suppresses_repeat(self):
        det = AbandonedObjectDetector(stationary_seconds=0.01,
                                      dedup_window_s=60.0, owner_radius_px=1.0)
        t   = _track(1, class_name="car", cx=400, cy=300)
        det.update([t], frame_id=1)
        time.sleep(0.02)
        ev1 = det.update([t], frame_id=2)
        ev2 = det.update([t], frame_id=3)
        a1  = [e for e in ev1 if e.event_type == EventType.ABANDONED_OBJ]
        a2  = [e for e in ev2 if e.event_type == EventType.ABANDONED_OBJ]
        assert len(a1) == 1
        assert len(a2) == 0

    def test_metadata_keys(self):
        det = AbandonedObjectDetector(stationary_seconds=0.01,
                                      dedup_window_s=0.0, owner_radius_px=1.0)
        t   = _track(1, class_name="car", cx=400, cy=300)
        det.update([t], frame_id=1)
        time.sleep(0.02)
        events = det.update([t], frame_id=2)
        aband  = [e for e in events if e.event_type == EventType.ABANDONED_OBJ]
        assert len(aband) == 1
        meta = aband[0].metadata
        assert "stationary_s" in meta and "owner_nearby" in meta


# =============================================================================
# BehaviourEngine (orchestration)
# =============================================================================

class TestBehaviourEngine:

    def test_returns_empty_with_no_tracks(self):
        eng    = BehaviourEngine(camera_id="test")
        events = eng.update([], frame_id=1)
        assert events == []

    def test_loitering_through_engine(self):
        eng = BehaviourEngine(
            camera_id   = "test",
            loiter_dwell_s = 0.05,
            enable_running  = False,
            enable_gathering = False,
            enable_abandoned = False,
        )
        t = _track(1)
        eng.update([t], frame_id=1)
        time.sleep(0.1)
        events = eng.update([t], frame_id=2)
        loiter = [e for e in events if e.event_type == EventType.LOITERING]
        assert len(loiter) >= 1

    def test_gathering_through_engine(self):
        eng = BehaviourEngine(
            camera_id        = "test",
            gather_min_count = 3,
            gather_proximity_px = 100.0,
            gather_duration_s   = 0.05,
            enable_loitering = False,
            enable_running   = False,
            enable_abandoned = False,
        )
        tracks = [_track(i, cx=100 + i * 10, cy=100) for i in range(4)]
        eng.update(tracks, frame_id=1)
        time.sleep(0.1)
        events = eng.update(tracks, frame_id=2)
        gather = [e for e in events if e.event_type == EventType.GATHERING]
        assert len(gather) >= 1

    def test_abandoned_through_engine(self):
        eng = BehaviourEngine(
            camera_id            = "test",
            abandoned_stationary_s = 0.05,
            abandoned_owner_radius_px = 1.0,
            enable_loitering = False,
            enable_running   = False,
            enable_gathering = False,
        )
        t = _track(1, class_name="car", cx=400, cy=300)
        eng.update([t], frame_id=1)
        time.sleep(0.1)
        events = eng.update([t], frame_id=2)
        aband  = [e for e in events if e.event_type == EventType.ABANDONED_OBJ]
        assert len(aband) >= 1

    def test_velocity_annotated_on_tracks(self):
        eng = BehaviourEngine(camera_id="test", enable_running=True)
        t1  = _track(1, cx=100, cy=100)
        eng.update([t1], frame_id=1)
        time.sleep(0.05)
        t2  = _track(1, cx=200, cy=100)
        eng.update([t2], frame_id=2)
        # After speed update, track.velocity should be set
        assert t2.velocity is not None, "velocity should be annotated on track"
        assert isinstance(t2.velocity, tuple) and len(t2.velocity) == 2

    def test_reset_clears_all_state(self):
        eng = BehaviourEngine(camera_id="test")
        t = _track(1)
        eng.update([t], frame_id=1)
        eng.reset()
        assert eng._loitering._state == {}
        assert eng._speed._state == {}
        assert eng._gathering._clusters == {}
        assert eng._abandoned._state == {}

    def test_disable_all_rules_returns_empty(self):
        eng = BehaviourEngine(
            camera_id        = "test",
            enable_loitering = False,
            enable_running   = False,
            enable_gathering = False,
            enable_abandoned = False,
        )
        tracks = [_track(i) for i in range(5)]
        events = eng.update(tracks, frame_id=1)
        assert events == []
