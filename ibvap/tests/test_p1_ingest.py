"""
tests/test_p1_ingest.py
========================
P1 tests for services/ingest/reader.py

These tests use a synthetic source (a tiny in-memory video written to a temp
file) so no network or real camera is needed.

Run:
    pytest tests/test_p1_ingest.py -v
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.ingest.reader import FrameData, FrameReader, ReaderStats


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_test_video(path: str, n_frames: int = 30, fps: float = 15.0) -> str:
    """Write a tiny 320x240 colour-gradient video to *path* and return it."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for i in range(n_frames):
        frame = np.full((240, 320, 3), [i * 8 % 255, 100, 200], dtype=np.uint8)
        w.write(frame)
    w.release()
    return path


@pytest.fixture
def test_video(tmp_path: Path) -> str:
    p = str(tmp_path / "test_clip.mp4")
    return _make_test_video(p, n_frames=30, fps=15.0)


@pytest.fixture
def file_camera_cfg(test_video: str) -> dict:
    return {
        "id": "cam_test",
        "source": {
            "type": "file",
            "path": test_video,
            "loop": False,
        },
        "capture": {
            "fps": 15,
            "target_fps": 10,
        },
        "analytics": {},
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFrameData:
    def test_fields_present(self):
        fd = FrameData(
            frame=np.zeros((240, 320, 3), dtype=np.uint8),
            frame_id=1,
            timestamp=time.monotonic(),
            camera_id="test",
            source_fps=15.0,
        )
        assert fd.frame_id == 1
        assert fd.camera_id == "test"
        assert fd.frame.shape == (240, 320, 3)


class TestFrameReader:

    def test_reader_starts_and_stops(self, file_camera_cfg: dict):
        reader = FrameReader(file_camera_cfg)
        reader.start()
        assert reader.is_alive
        reader.stop()
        # Thread should stop within 5 s
        time.sleep(0.5)
        assert not reader.is_alive

    def test_reads_frames(self, file_camera_cfg: dict):
        reader = FrameReader(file_camera_cfg, queue_maxsize=8)
        reader.start()

        frames_received = []
        deadline = time.monotonic() + 5.0   # 5 s timeout
        while time.monotonic() < deadline:
            fd = reader.read(timeout=0.5)
            if fd:
                frames_received.append(fd)
            if len(frames_received) >= 5:
                break

        reader.stop()
        assert len(frames_received) >= 5, (
            f"Expected >=5 frames, got {len(frames_received)}"
        )

    def test_frame_id_monotonic(self, file_camera_cfg: dict):
        reader = FrameReader(file_camera_cfg, queue_maxsize=16)
        reader.start()

        ids = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(ids) < 8:
            fd = reader.read(timeout=0.5)
            if fd:
                ids.append(fd.frame_id)
        reader.stop()

        assert ids == sorted(ids), "Frame IDs must be strictly monotonic"
        assert len(set(ids)) == len(ids), "Frame IDs must be unique"

    def test_frame_shape_bgr(self, file_camera_cfg: dict):
        reader = FrameReader(file_camera_cfg, queue_maxsize=4)
        reader.start()

        fd = reader.read(timeout=5.0)
        reader.stop()

        assert fd is not None, "Should have received at least one frame"
        assert fd.frame.ndim == 3
        assert fd.frame.shape[2] == 3   # BGR channels
        assert fd.frame.dtype == np.uint8

    def test_camera_id_correct(self, file_camera_cfg: dict):
        reader = FrameReader(file_camera_cfg)
        reader.start()
        fd = reader.read(timeout=5.0)
        reader.stop()

        assert fd is not None
        assert fd.camera_id == "cam_test"

    def test_stats_increments(self, file_camera_cfg: dict):
        reader = FrameReader(file_camera_cfg, queue_maxsize=4)
        reader.start()
        time.sleep(1.5)
        reader.stop()

        assert reader.stats.frames_read > 0, "frames_read must be > 0"

    def test_invalid_source_type_raises(self):
        bad_cfg = {
            "id": "bad",
            "source": {"type": "ftp", "url": "ftp://example.com"},
            "capture": {"target_fps": 10},
        }
        with pytest.raises(ValueError, match="Unknown source type"):
            FrameReader(bad_cfg)

    def test_nonexistent_file_does_not_crash_start(self, tmp_path: Path):
        """Reader must start gracefully even if file doesn't exist."""
        cfg = {
            "id": "cam_bad_file",
            "source": {
                "type": "file",
                "path": str(tmp_path / "nonexistent.mp4"),
                "loop": False,
            },
            "capture": {"target_fps": 10},
        }
        reader = FrameReader(cfg)
        reader.start()   # should not raise
        fd = reader.read(timeout=2.0)
        reader.stop()
        # May or may not return a frame; just must not crash
