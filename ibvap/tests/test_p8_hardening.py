"""
tests/test_p8_hardening.py
==========================
IBVAP P8 — Test Suite for Hardening, Configuration Validation,
Backpressure Buffering, Stream Watchdogs, and Capacity Profiling.
"""

from __future__ import annotations

import time
import pytest
import yaml
from pathlib import Path

from services.benchmarks.profiler import LatencyStats, PerformanceProfiler
from services.hardening.buffer import BoundedFrameBuffer, DropPolicy
from services.hardening.validator import ConfigValidator
from services.hardening.watchdog import StreamState, StreamWatchdog


# ── 1. Configuration Validation Tests ─────────────────────────────────────────

def test_config_validator_real_configs():
    """Verify that current repository configs in configs/ pass validation."""
    validator = ConfigValidator(config_dir="configs")
    report = validator.validate_all()
    assert report.is_valid, f"Validation failed on repo configs: {report.errors}"


def test_config_validator_duplicate_camera_id(tmp_path):
    cam_data = {
        "cameras": [
            {"id": "cam_01", "name": "Cam 1", "source": {"type": "rtsp", "url": "rtsp://localhost/1"}},
            {"id": "cam_01", "name": "Cam 2", "source": {"type": "rtsp", "url": "rtsp://localhost/2"}},
        ]
    }
    (tmp_path / "cameras.yaml").write_text(yaml.dump(cam_data), encoding="utf-8")
    (tmp_path / "zones.yaml").write_text("cameras: {}", encoding="utf-8")
    (tmp_path / "rules.yaml").write_text("rules: []", encoding="utf-8")

    validator = ConfigValidator(config_dir=str(tmp_path))
    report = validator.validate_all()
    assert not report.is_valid
    assert any("duplicate camera id 'cam_01'" in err for err in report.errors)


def test_config_validator_invalid_polygon_points(tmp_path):
    zone_data = {
        "cameras": {
            "cam_01": {
                "polygons": [
                    {"id": "poly_bad", "points": [[0.1, 0.1], [0.2, 0.2]]}  # Only 2 points!
                ]
            }
        }
    }
    (tmp_path / "cameras.yaml").write_text("cameras: []", encoding="utf-8")
    (tmp_path / "zones.yaml").write_text(yaml.dump(zone_data), encoding="utf-8")
    (tmp_path / "rules.yaml").write_text("rules: []", encoding="utf-8")

    validator = ConfigValidator(config_dir=str(tmp_path))
    report = validator.validate_all()
    assert not report.is_valid
    assert any("requires at least 3 points" in err for err in report.errors)


def test_config_validator_unknown_event_type(tmp_path):
    rules_data = {
        "rules": [
            {"id": "rule_bad", "event_type": "UNKNOWN_ALIEN_INVASION", "severity": "HIGH"}
        ]
    }
    (tmp_path / "cameras.yaml").write_text("cameras: []", encoding="utf-8")
    (tmp_path / "zones.yaml").write_text("cameras: {}", encoding="utf-8")
    (tmp_path / "rules.yaml").write_text(yaml.dump(rules_data), encoding="utf-8")

    validator = ConfigValidator(config_dir=str(tmp_path))
    report = validator.validate_all()
    assert not report.is_valid
    assert any("unknown event_type" in err for err in report.errors)


# ── 2. Bounded Buffer & Backpressure Tests ────────────────────────────────────

def test_buffer_drop_oldest():
    buf: BoundedFrameBuffer[int] = BoundedFrameBuffer(maxsize=3, policy=DropPolicy.DROP_OLDEST)
    buf.put(1)
    buf.put(2)
    buf.put(3)
    assert buf.qsize() == 3

    # Buffer full, put 4th item -> item 1 should be dropped
    buf.put(4)
    assert buf.qsize() == 3
    assert buf.total_dropped == 1

    # Remaining items should be [2, 3, 4]
    assert buf.get() == 2
    assert buf.get() == 3
    assert buf.get() == 4
    assert buf.empty()


def test_buffer_drop_newest():
    buf: BoundedFrameBuffer[int] = BoundedFrameBuffer(maxsize=2, policy=DropPolicy.DROP_NEWEST)
    assert buf.put(10) is True
    assert buf.put(20) is True

    # Buffer full, put 30 -> 30 rejected
    assert buf.put(30) is False
    assert buf.total_dropped == 1

    # Remaining items should still be [10, 20]
    assert buf.get() == 10
    assert buf.get() == 20


def test_buffer_stats_and_close():
    buf: BoundedFrameBuffer[str] = BoundedFrameBuffer(maxsize=2, policy=DropPolicy.DROP_OLDEST)
    buf.put("A")
    buf.put("B")
    buf.put("C")
    stats = buf.get_stats()
    assert stats["maxsize"] == 2
    assert stats["total_pushed"] == 3
    assert stats["total_dropped"] == 1
    assert stats["drop_rate_pct"] > 0

    buf.close()
    assert buf.is_closed
    assert buf.put("D") is False


# ── 3. Stream Watchdog Tests ──────────────────────────────────────────────────

def test_watchdog_healthy():
    dog = StreamWatchdog(camera_id="cam_test", timeout_s=1.0)
    dog.heartbeat(frame_id=10)
    assert dog.check_health() is True
    assert dog.state == StreamState.ONLINE


def test_watchdog_timeout_and_reconnect():
    reconnected = False

    def mock_reconnect():
        nonlocal reconnected
        reconnected = True
        return True

    dog = StreamWatchdog(
        camera_id="cam_test",
        timeout_s=0.2,
        reconnect_fn=mock_reconnect,
        base_delay_s=0.05,
    )
    dog.heartbeat(frame_id=1)
    time.sleep(0.3)  # Wait for timeout

    assert dog.check_health() is False
    assert dog.state == StreamState.STALLED

    success = dog.attempt_reconnect()
    assert success is True
    assert reconnected is True
    assert dog.state == StreamState.ONLINE
    assert dog.total_reconnects == 1


def test_watchdog_max_retries_failed():
    def failing_reconnect():
        return False

    dog = StreamWatchdog(
        camera_id="cam_fail",
        timeout_s=0.1,
        reconnect_fn=failing_reconnect,
        max_retries=2,
        base_delay_s=0.01,
    )
    dog.attempt_reconnect()
    assert dog.consecutive_failures == 1
    dog.attempt_reconnect()
    assert dog.consecutive_failures == 2
    # 3rd attempt exceeds max_retries
    dog.attempt_reconnect()
    assert dog.state == StreamState.FAILED


# ── 4. Profiler & Latency Stats Tests ─────────────────────────────────────────

def test_latency_stats_calculation():
    durations = [0.010, 0.020, 0.030, 0.040, 0.050]
    stats = LatencyStats.from_durations(durations)
    assert stats.count == 5
    assert stats.mean_ms == 30.0
    assert stats.p50_ms == 30.0
    assert stats.min_ms == 10.0
    assert stats.max_ms == 50.0
    assert stats.fps > 0


def test_profiler_hardware_info():
    profiler = PerformanceProfiler()
    info = profiler.get_hardware_info()
    assert "os" in info
    assert "cpu" in info
    assert "ram_gb" in info
    assert info["ram_gb"] > 0


def test_profiler_tracker_benchmark():
    profiler = PerformanceProfiler()
    stats = profiler.benchmark_tracker(num_frames=20, num_detections=5)
    assert stats.count == 20
    assert stats.mean_ms > 0
    assert stats.fps > 0


def test_profiler_fence_benchmark():
    profiler = PerformanceProfiler()
    stats = profiler.benchmark_fence_engine(num_frames=20, num_tracks=10)
    assert stats.count == 20
    assert stats.mean_ms > 0


def test_profiler_report_generation(tmp_path):
    profiler = PerformanceProfiler(data_dir=str(tmp_path))
    res = profiler.run_full_suite()
    report_file = profiler.generate_markdown_report(res, filename="test_report.md")
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "# IBVAP System Performance & Capacity Benchmark Report" in content
    assert "Multi-Stream Ingest & Pipeline Scaling" in content
