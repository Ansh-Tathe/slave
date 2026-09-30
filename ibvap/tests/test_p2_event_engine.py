"""
tests/test_p2_event_engine.py
==============================
P2 — Unit tests for the event engine and logger.

No GPU, no video.  Events are built from synthetic Track objects.

Run:
    pytest tests/test_p2_event_engine.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from services.detect_track.models import Track
from services.event_engine.engine  import EventEngine
from services.event_engine.logger  import EventLogger
from services.event_engine.models  import Event, EventType, Severity


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id, bbox=(100.0, 100.0, 300.0, 400.0),
        conf=0.9, class_id=0, class_name="person",
        frame_id=1, camera_id="test",
    )


def make_event(
    etype: EventType = EventType.ZONE_ENTER,
    severity: Severity = Severity.HIGH,
    track_id: int = 1,
    zone_id: str = "z1",
) -> Event:
    return Event(
        event_type=etype,
        severity=severity,
        camera_id="test",
        track=make_track(track_id),
        frame_id=1,
        zone_id=zone_id,
    )


def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# =============================================================================
# 1. Event model
# =============================================================================

class TestEventModel:
    def test_event_id_is_uuid(self):
        e = make_event()
        import uuid
        uuid.UUID(e.event_id)  # should not raise

    def test_two_events_have_different_ids(self):
        e1 = make_event()
        e2 = make_event()
        assert e1.event_id != e2.event_id

    def test_to_dict_json_serialisable(self):
        e = make_event()
        d = e.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_required_keys(self):
        e = make_event()
        d = e.to_dict()
        for key in ("event_id", "event_type", "severity", "camera_id",
                    "track_id", "class_name", "bbox", "timestamp", "confirmed"):
            assert key in d, f"Missing key: {key}"

    def test_confirmed_defaults_none(self):
        e = make_event()
        assert e.confirmed is None

    def test_severity_ordering(self):
        assert Severity.LOW < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL


# =============================================================================
# 2. EventEngine — deduplication
# =============================================================================

class TestEventEngineDedup:
    def test_first_event_passes(self):
        engine = EventEngine(dedup_window_s=30.0, snapshot_dir="data/test_snaps")
        e = make_event()
        out = engine.process([e], blank_frame())
        assert len(out) == 1

    def test_duplicate_suppressed_within_window(self):
        engine = EventEngine(dedup_window_s=30.0, snapshot_dir="data/test_snaps")
        e1 = make_event()
        e2 = make_event()  # same type + track_id + zone_id
        engine.process([e1], blank_frame())
        out = engine.process([e2], blank_frame())
        assert len(out) == 0

    def test_different_track_id_not_suppressed(self):
        engine = EventEngine(dedup_window_s=30.0, snapshot_dir="data/test_snaps")
        e1 = make_event(track_id=1)
        e2 = make_event(track_id=2)
        engine.process([e1], blank_frame())
        out = engine.process([e2], blank_frame())
        assert len(out) == 1

    def test_different_zone_not_suppressed(self):
        engine = EventEngine(dedup_window_s=30.0, snapshot_dir="data/test_snaps")
        e1 = make_event(zone_id="z1")
        e2 = make_event(zone_id="z2")
        engine.process([e1], blank_frame())
        out = engine.process([e2], blank_frame())
        assert len(out) == 1

    def test_different_event_type_not_suppressed(self):
        engine = EventEngine(dedup_window_s=30.0, snapshot_dir="data/test_snaps")
        e1 = make_event(etype=EventType.ZONE_ENTER)
        e2 = make_event(etype=EventType.DWELL)
        engine.process([e1], blank_frame())
        out = engine.process([e2], blank_frame())
        assert len(out) == 1

    def test_event_passes_after_window_expires(self):
        engine = EventEngine(dedup_window_s=0.05, snapshot_dir="data/test_snaps")
        e1 = make_event()
        engine.process([e1], blank_frame())
        time.sleep(0.07)  # wait past window
        e2 = make_event()
        out = engine.process([e2], blank_frame())
        assert len(out) == 1

    def test_empty_input_returns_empty(self):
        engine = EventEngine(dedup_window_s=30.0, snapshot_dir="data/test_snaps")
        out = engine.process([], blank_frame())
        assert out == []

    def test_snapshot_path_attached(self, tmp_path):
        engine = EventEngine(
            dedup_window_s=30.0,
            snapshot_dir=str(tmp_path / "snaps"),
        )
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        e = make_event()
        out = engine.process([e], frame)
        assert len(out) == 1
        assert out[0].snapshot_path is not None
        assert Path(out[0].snapshot_path).exists()


# =============================================================================
# 3. EventLogger
# =============================================================================

class TestEventLogger:
    def test_log_writes_jsonl(self, tmp_path):
        log_path = str(tmp_path / "events.jsonl")
        el = EventLogger(log_path=log_path, console=False)
        e = make_event()
        e.snapshot_path = None
        el.log(e)
        lines = Path(log_path).read_text().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["event_type"] == "ZONE_ENTER"

    def test_log_many(self, tmp_path):
        log_path = str(tmp_path / "events.jsonl")
        el = EventLogger(log_path=log_path, console=False)
        events = [make_event(track_id=i) for i in range(5)]
        el.log_many(events)
        lines = Path(log_path).read_text().splitlines()
        assert len(lines) == 5

    def test_session_count_increments(self, tmp_path):
        log_path = str(tmp_path / "events.jsonl")
        el = EventLogger(log_path=log_path, console=False)
        assert el.session_count == 0
        el.log(make_event())
        el.log(make_event(track_id=2))
        assert el.session_count == 2

    def test_log_valid_json_per_line(self, tmp_path):
        log_path = str(tmp_path / "events.jsonl")
        el = EventLogger(log_path=log_path, console=False)
        for i in range(10):
            el.log(make_event(track_id=i, zone_id=f"z{i}"))
        for line in Path(log_path).read_text().splitlines():
            obj = json.loads(line)  # must parse cleanly
            assert "event_id" in obj
