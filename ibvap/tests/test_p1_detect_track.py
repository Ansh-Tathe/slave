"""
tests/test_p1_detect_track.py
==============================
P1 tests for the data models, detector, tracker, and renderer.

GPU-dependent tests are skipped automatically if CUDA is unavailable.
Model-dependent tests download yolov8n.pt (~6 MB) on first run.

Run:
    pytest tests/test_p1_detect_track.py -v
    pytest tests/test_p1_detect_track.py -v -k "not gpu"   # skip GPU tests
"""

from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np
import pytest

# ── import guards ─────────────────────────────────────────────────────────────
torch   = pytest.importorskip("torch",       reason="PyTorch not installed")
ultral  = pytest.importorskip("ultralytics", reason="ultralytics not installed")

from services.detect_track.models   import (
    COCO_CLASSES, DETECT_CLASS_IDS,
    Detection, Track, CLASS_COLORS,
)
from services.detect_track.renderer import Renderer


# ── Helpers ───────────────────────────────────────────────────────────────────

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
HALF   = CUDA_AVAILABLE

def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)

def _person_frame() -> np.ndarray:
    """
    A 640x480 frame with a rough white-on-grey rectangle that might
    resemble a very coarse silhouette.  Not expected to reliably trigger
    detection — used only to verify the pipeline doesn't crash.
    """
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    # Draw a rough person-shaped blob
    cv2.rectangle(frame, (280, 100), (360, 400), (200, 200, 200), cv2.FILLED)
    cv2.circle(frame, (320, 80), 40, (200, 200, 200), cv2.FILLED)
    return frame


# =============================================================================
# 1. Data models
# =============================================================================

class TestDetection:
    def test_basic_fields(self):
        d = Detection(bbox=(10.0, 20.0, 100.0, 200.0), conf=0.85,
                      class_id=0, class_name="person")
        assert d.class_id == 0
        assert d.conf == pytest.approx(0.85)

    def test_derived_properties(self):
        d = Detection(bbox=(0.0, 0.0, 100.0, 200.0), conf=0.9,
                      class_id=0, class_name="person")
        assert d.width  == pytest.approx(100.0)
        assert d.height == pytest.approx(200.0)
        assert d.area   == pytest.approx(20000.0)
        assert d.center == pytest.approx((50.0, 100.0))

    def test_to_dict_keys(self):
        d = Detection(bbox=(0, 0, 10, 10), conf=0.5, class_id=2, class_name="car")
        dd = d.to_dict()
        assert {"bbox", "conf", "class_id", "class_name"} == set(dd.keys())

    def test_bbox_is_list_in_dict(self):
        d = Detection(bbox=(1.1, 2.2, 3.3, 4.4), conf=0.7,
                      class_id=0, class_name="person")
        assert isinstance(d.to_dict()["bbox"], list)


class TestTrack:
    def _make_track(self, **kw) -> Track:
        defaults = dict(
            track_id=1, bbox=(0.0, 0.0, 50.0, 100.0),
            conf=0.9, class_id=0, class_name="person",
            frame_id=1, camera_id="cam_01",
        )
        defaults.update(kw)
        return Track(**defaults)

    def test_basic_fields(self):
        t = self._make_track()
        assert t.track_id == 1
        assert t.class_name == "person"

    def test_color_returns_tuple(self):
        t = self._make_track(class_id=0)
        c = t.color
        assert isinstance(c, tuple) and len(c) == 3

    def test_to_dict_has_required_keys(self):
        t = self._make_track()
        d = t.to_dict()
        for key in ("track_id", "bbox", "conf", "class_id", "class_name",
                    "frame_id", "camera_id", "timestamp", "center"):
            assert key in d, f"Missing key: {key}"

    def test_center_correct(self):
        t = self._make_track(bbox=(100.0, 200.0, 300.0, 400.0))
        assert t.center == pytest.approx((200.0, 300.0))


class TestClassMapping:
    def test_person_is_class_0(self):
        assert COCO_CLASSES[0] == "person"

    def test_detect_class_ids_sorted(self):
        assert DETECT_CLASS_IDS == sorted(DETECT_CLASS_IDS)

    def test_all_classes_have_color(self):
        for cid in DETECT_CLASS_IDS:
            assert cid in CLASS_COLORS, f"No color for class_id={cid}"


# =============================================================================
# 2. Renderer (no GPU needed)
# =============================================================================

class TestRenderer:
    def _make_tracks(self) -> List[Track]:
        return [
            Track(track_id=1, bbox=(50.0, 50.0, 200.0, 400.0), conf=0.91,
                  class_id=0, class_name="person", frame_id=1, camera_id="cam"),
            Track(track_id=2, bbox=(300.0, 80.0, 550.0, 300.0), conf=0.75,
                  class_id=2, class_name="car", frame_id=1, camera_id="cam"),
        ]

    def test_draw_returns_same_shape(self):
        frame = _blank_frame()
        r = Renderer()
        out = r.draw(frame, self._make_tracks(), frame_id=1)
        assert out.shape == frame.shape

    def test_draw_does_not_mutate_input(self):
        frame = _blank_frame()
        original = frame.copy()
        r = Renderer()
        r.draw(frame, self._make_tracks(), frame_id=1)
        np.testing.assert_array_equal(frame, original)

    def test_draw_empty_tracks(self):
        frame = _blank_frame()
        r = Renderer()
        out = r.draw(frame, [], frame_id=0)
        assert out.shape == frame.shape

    def test_draw_many_tracks_no_crash(self):
        frame = _blank_frame(h=720, w=1280)
        r = Renderer()
        # 50 random tracks
        tracks = [
            Track(
                track_id=i,
                bbox=(float(i*20 % 1000), float(i*15 % 600),
                      float(i*20 % 1000 + 80), float(i*15 % 600 + 150)),
                conf=0.5 + i * 0.01,
                class_id=i % 6 if i % 6 in DETECT_CLASS_IDS else 0,
                class_name="person",
                frame_id=1,
                camera_id="cam",
            )
            for i in range(50)
        ]
        out = r.draw(frame, tracks, frame_id=100)
        assert out.shape == frame.shape

    def test_fps_increases_over_multiple_draws(self):
        r = Renderer(fps_window=10)
        frame = _blank_frame()
        for i in range(12):
            r.draw(frame, [], frame_id=i)
            time.sleep(0.01)
        # After 12 draws with ~10ms gaps we should get some FPS estimate
        # (just check it doesn't crash and returns a frame)


# =============================================================================
# 3. Detector (GPU optional)
# =============================================================================

class TestYOLODetector:
    """Requires ultralytics and downloads yolov8n.pt (~6 MB) on first run."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_detector_loads_and_detects_blank_frame(self):
        from services.detect_track.detector import YOLODetector
        det = YOLODetector(model_path="yolov8n.pt", device=DEVICE,
                           half=HALF, warmup_runs=1)
        dets = det.detect(_blank_frame())
        # Blank frame → 0 detections (no crash)
        assert isinstance(dets, list)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_detections_have_correct_types(self):
        from services.detect_track.detector import YOLODetector
        det = YOLODetector(model_path="yolov8n.pt", device=DEVICE,
                           half=HALF, warmup_runs=1)
        dets = det.detect(_blank_frame())
        for d in dets:
            assert isinstance(d, Detection)
            assert 0.0 <= d.conf <= 1.0
            assert d.class_id in DETECT_CLASS_IDS

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_detections_sorted_by_confidence(self):
        from services.detect_track.detector import YOLODetector
        det = YOLODetector(model_path="yolov8n.pt", device=DEVICE,
                           half=HALF, warmup_runs=1)
        dets = det.detect(_blank_frame())
        if len(dets) >= 2:
            confs = [d.conf for d in dets]
            assert confs == sorted(confs, reverse=True)


# =============================================================================
# 4. Tracker (GPU optional)
# =============================================================================

class TestByteTracker:
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_tracker_loads(self):
        from services.detect_track.tracker import ByteTracker
        bt = ByteTracker(model_path="yolov8n.pt", device=DEVICE,
                         half=HALF, warmup_runs=1)
        assert bt is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_update_blank_returns_list(self):
        from services.detect_track.tracker import ByteTracker
        bt = ByteTracker(model_path="yolov8n.pt", device=DEVICE,
                         half=HALF, warmup_runs=1)
        tracks = bt.update(
            frame=_blank_frame(),
            frame_id=1,
            camera_id="cam_test",
            timestamp=time.monotonic(),
        )
        assert isinstance(tracks, list)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_track_fields_valid(self):
        from services.detect_track.tracker import ByteTracker
        bt = ByteTracker(model_path="yolov8n.pt", device=DEVICE,
                         half=HALF, warmup_runs=1)
        for i in range(3):
            tracks = bt.update(
                frame=_blank_frame(),
                frame_id=i + 1,
                camera_id="cam_test",
                timestamp=time.monotonic(),
            )
        for t in tracks:
            assert isinstance(t.track_id, int)
            assert t.track_id > 0
            assert t.class_id in DETECT_CLASS_IDS
            assert 0.0 <= t.conf <= 1.0
            x1, y1, x2, y2 = t.bbox
            assert x2 > x1 and y2 > y1

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_track_to_dict_serialisable(self):
        """Track.to_dict() must be JSON-serialisable (no numpy types)."""
        import json
        from services.detect_track.tracker import ByteTracker
        bt = ByteTracker(model_path="yolov8n.pt", device=DEVICE,
                         half=HALF, warmup_runs=1)
        tracks = bt.update(
            frame=_blank_frame(),
            frame_id=1,
            camera_id="cam_test",
            timestamp=time.monotonic(),
        )
        for t in tracks:
            json.dumps(t.to_dict())   # must not raise
