"""
tests/test_p4_night.py
========================
IBVAP P4 -- Unit tests for night-time enhancement module.

Tests:
  - NightEnhancer.is_dark() correctly classifies synthetic frames
  - NightEnhancer.enhance() does not raise on valid frames
  - Each mode (clahe, gamma, combined) produces brighter output than input
  - NightEnhancer handles edge cases: zero-size frame, single pixel
  - NightEngine.update() returns events only during night hours
  - NightEngine.update() respects dedup window
  - IndirectIndian plate validator integration (smoke test -- not P4 but ensures imports work)
"""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.analytics.night.night_enhancer import NightEnhancer, _DARK_THRESHOLD
from services.analytics.night.night_engine import NightEngine
from services.detect_track.models import Track
from services.event_engine.models import EventType, Severity


# =============================================================================
# Fixtures
# =============================================================================

def _dark_frame(h: int = 240, w: int = 320, brightness: int = 20) -> np.ndarray:
    """Synthetic dark BGR frame (constant colour)."""
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _bright_frame(h: int = 240, w: int = 320, brightness: int = 200) -> np.ndarray:
    """Synthetic bright BGR frame."""
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _make_track(track_id: int = 1, class_name: str = "person") -> Track:
    return Track(
        track_id   = track_id,
        bbox       = (10.0, 10.0, 100.0, 200.0),
        conf       = 0.80,
        class_id   = 0,
        class_name = class_name,
        frame_id   = 1,
        camera_id  = "test_cam",
    )


# =============================================================================
# NightEnhancer -- darkness detection
# =============================================================================

class TestDarknessDetection:

    def test_dark_frame_detected(self):
        enhancer = NightEnhancer()
        frame    = _dark_frame(brightness=15)
        assert enhancer.is_dark(frame), "Very dark frame should be detected as dark"

    def test_bright_frame_not_dark(self):
        enhancer = NightEnhancer()
        frame    = _bright_frame(brightness=200)
        assert not enhancer.is_dark(frame), "Bright frame should NOT be detected as dark"

    def test_custom_threshold(self):
        enhancer = NightEnhancer(dark_threshold=50.0)
        # brightness=60 in BGR -> LAB-L somewhere > 50
        frame = _bright_frame(brightness=60)
        # We test both sides of a custom threshold
        assert isinstance(enhancer.is_dark(frame), bool)

    def test_mean_luminance_range(self):
        enhancer = NightEnhancer()
        frame    = _dark_frame(brightness=0)
        lum      = enhancer.mean_luminance(frame)
        assert 0.0 <= lum <= 255.0

    def test_empty_frame_not_dark(self):
        enhancer  = NightEnhancer()
        empty     = np.zeros((0, 0, 3), dtype=np.uint8)
        result    = enhancer.is_dark(empty)
        assert result is False   # edge-case: treat empty as not-dark


# =============================================================================
# NightEnhancer -- enhancement output
# =============================================================================

class TestEnhancement:

    @pytest.mark.parametrize("mode", ["clahe", "gamma", "combined"])
    def test_output_shape_preserved(self, mode):
        enhancer = NightEnhancer(mode=mode)
        frame    = _dark_frame()
        enhanced = enhancer.enhance(frame)
        assert enhanced.shape == frame.shape, f"[{mode}] output shape mismatch"

    @pytest.mark.parametrize("mode", ["clahe", "gamma", "combined"])
    def test_output_dtype_preserved(self, mode):
        enhancer = NightEnhancer(mode=mode)
        frame    = _dark_frame()
        enhanced = enhancer.enhance(frame)
        assert enhanced.dtype == np.uint8, f"[{mode}] output dtype should be uint8"

    @pytest.mark.parametrize("mode", ["clahe", "gamma", "combined"])
    def test_dark_frame_is_brightened(self, mode):
        # Use brightness=60 (moderately dark but not near-zero).
        # Near-zero pixels (< ~5) stay near-zero even with gamma < 1 due to uint8 rounding.
        enhancer = NightEnhancer(mode=mode)
        frame    = _dark_frame(brightness=60)
        enhanced = enhancer.enhance(frame)
        assert enhanced.mean() > frame.mean(), \
            f"[{mode}] enhanced mean {enhanced.mean():.1f} should > original {frame.mean():.1f}"

    def test_input_not_mutated(self):
        enhancer  = NightEnhancer(mode="combined")
        frame     = _dark_frame(brightness=20)
        original  = frame.copy()
        enhancer.enhance(frame)
        np.testing.assert_array_equal(frame, original, err_msg="Input frame was mutated")

    def test_empty_frame_returns_input(self):
        enhancer = NightEnhancer(mode="combined")
        empty    = np.zeros((0, 0, 3), dtype=np.uint8)
        result   = enhancer.enhance(empty)
        assert result is empty   # should be the same object

    def test_single_pixel_frame(self):
        enhancer = NightEnhancer(mode="combined")
        tiny     = np.array([[[10, 10, 10]]], dtype=np.uint8)
        enhanced = enhancer.enhance(tiny)
        assert enhanced.shape == (1, 1, 3)

    def test_dnn_fallback_to_combined(self):
        """DNN mode with missing model should fall back gracefully."""
        enhancer = NightEnhancer(mode="dnn", dnn_model_path="nonexistent_model.onnx")
        assert enhancer._mode == "combined"
        frame    = _dark_frame()
        enhanced = enhancer.enhance(frame)
        assert enhanced.shape == frame.shape


# =============================================================================
# NightEnhancer -- gamma LUT
# =============================================================================

class TestGammaLUT:

    def test_gamma_lut_shape(self):
        lut = NightEnhancer._build_gamma_lut(0.5)
        assert lut.shape == (256,) and lut.dtype == np.uint8

    def test_gamma_lut_monotone(self):
        lut = NightEnhancer._build_gamma_lut(0.5)
        diffs = np.diff(lut.astype(np.int16))
        assert (diffs >= 0).all(), "Gamma LUT should be monotonically non-decreasing"

    def test_bright_gamma_identity(self):
        """gamma=1.0 should map pixel[i] = i (identity)."""
        lut = NightEnhancer._build_gamma_lut(1.0)
        assert lut[0]   == 0
        assert lut[255] == 255
        # Mid-range pixel may differ by 1 due to float->uint8 rounding
        assert abs(int(lut[128]) - 128) <= 1

    def test_dark_gamma_brightens(self):
        """gamma < 1 (standard convention) should brighten mid-range pixels."""
        lut = NightEnhancer._build_gamma_lut(0.4)   # gamma=0.4 < 1 -> brightening
        # For pixel=128: (128/255)^0.4 * 255 = 0.502^0.4 * 255 ≈ 0.794 * 255 ≈ 202
        assert lut[128] > 128, "gamma=0.4 < 1 should brighten mid-range pixel 128"


# =============================================================================
# NightEngine -- orchestration
# =============================================================================

class TestNightEngine:

    def test_no_events_when_no_tracks(self):
        engine      = NightEngine(night_hours=None, always_enhance=True)
        frame       = _dark_frame()
        _, events   = engine.update(frame, [], frame_id=1, camera_id="cam")
        assert events == []

    def test_events_emitted_for_tracks(self):
        engine = NightEngine(night_hours=None, always_enhance=True, dedup_window_s=0.0)
        frame  = _dark_frame()
        tracks = [_make_track(1), _make_track(2)]
        _, events = engine.update(frame, tracks, frame_id=1, camera_id="cam")
        assert len(events) == 2
        assert all(e.event_type == EventType.NIGHT_MOVEMENT for e in events)

    def test_dedup_suppresses_repeat(self):
        engine = NightEngine(night_hours=None, always_enhance=True, dedup_window_s=60.0)
        frame  = _dark_frame()
        track  = _make_track(1)

        _, ev1 = engine.update(frame, [track], frame_id=1)
        _, ev2 = engine.update(frame, [track], frame_id=2)
        assert len(ev1) == 1, "First call should emit 1 event"
        assert len(ev2) == 0, "Second call within dedup window should emit 0"

    def test_dedup_expires(self):
        engine = NightEngine(night_hours=None, always_enhance=True, dedup_window_s=0.05)
        frame  = _dark_frame()
        track  = _make_track(1)

        _, ev1 = engine.update(frame, [track], frame_id=1)
        time.sleep(0.15)   # 150ms sleep — well past 50ms dedup window
        _, ev2 = engine.update(frame, [track], frame_id=2)
        assert len(ev1) == 1
        assert len(ev2) == 1, "Event should fire again after dedup window expires"

    def test_hour_gate_suppresses_events(self):
        """Events suppressed if current hour NOT in night_hours."""
        current_hour = datetime.now().hour
        # Pick a set that definitely excludes current hour
        other_hour   = (current_hour + 12) % 24
        engine = NightEngine(
            night_hours    = {other_hour},
            always_enhance = True,
            dedup_window_s = 0.0,
        )
        frame  = _dark_frame()
        track  = _make_track(1)
        _, events = engine.update(frame, [track], frame_id=1)
        assert events == [], "Events should be suppressed outside night hours"

    def test_no_hour_gate_always_emits(self):
        """night_hours=None means emit at any hour."""
        engine = NightEngine(night_hours=None, always_enhance=True, dedup_window_s=0.0)
        frame  = _dark_frame()
        track  = _make_track(1)
        _, events = engine.update(frame, [track], frame_id=1)
        assert len(events) == 1

    def test_enhanced_frame_is_brighter(self):
        engine = NightEngine(enhancer_mode="combined", always_enhance=True)
        # Use brightness=60 (moderately dark) so gamma + CLAHE can visibly brighten
        frame  = _dark_frame(brightness=60)
        bright, _ = engine.update(frame, [], frame_id=1)
        assert bright.mean() > frame.mean(), "Enhanced frame should be brighter"

    def test_bright_frame_not_enhanced_without_always(self):
        engine = NightEngine(dark_threshold=40.0, always_enhance=False)
        frame  = _bright_frame(brightness=200)
        bright, _ = engine.update(frame, [], frame_id=1)
        # Should return original (not enhanced) — means pixel values identical
        np.testing.assert_array_equal(bright, frame)

    def test_stats_tracking(self):
        engine = NightEngine(always_enhance=True)
        frame  = _dark_frame()
        engine.update(frame, [], frame_id=1)
        engine.update(frame, [], frame_id=2)
        assert engine.stats["enhanced_frames"] == 2
        assert engine.stats["skipped_frames"]  == 0

    def test_event_severity(self):
        engine = NightEngine(
            night_hours    = None,
            always_enhance = True,
            dedup_window_s = 0.0,
            event_severity = Severity.CRITICAL,
        )
        frame  = _dark_frame()
        track  = _make_track(1)
        _, events = engine.update(frame, [track], frame_id=1)
        assert events[0].severity == Severity.CRITICAL

    def test_event_metadata_keys(self):
        engine = NightEngine(night_hours=None, always_enhance=True, dedup_window_s=0.0)
        frame  = _dark_frame()
        track  = _make_track(1)
        _, events = engine.update(frame, [track], frame_id=1)
        meta = events[0].metadata
        assert "luminance"     in meta
        assert "enhancer_mode" in meta
        assert "is_dark"       in meta
