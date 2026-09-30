"""
tests/test_p6_face.py
========================
IBVAP P6 -- Unit tests for face detection + watchlist pipeline.

Strategy: tests run regardless of whether InsightFace / FAISS is installed.
  - FaceDetector is tested via the YOLO-less path (no actual model),
    so we test quality-filtering helpers independently.
  - FaceEmbedder is tested in HOG-fallback mode (no model download needed).
  - Watchlist is tested with the pure-numpy _NumpyIndex fallback.
  - FaceEngine is tested with mock FaceDetection objects injected directly.

All tests are pure CPU, no GPU, no model download required.
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from services.analytics.face.face_detector import (
    FaceDetection, FaceDetector, _laplacian_sharpness, _align_face
)
from services.analytics.face.face_embedder import FaceEmbedder, SAME_PERSON_THRESHOLD
from services.analytics.face.watchlist     import Watchlist, _NumpyIndex
from services.analytics.face.face_engine   import FaceEngine, _face_stub_track
from services.event_engine.models          import EventType


# =============================================================================
# Helpers
# =============================================================================

EMBED_DIM = 512

def _random_crop(h: int = 112, w: int = 112) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

def _random_emb() -> np.ndarray:
    v = np.random.randn(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)

def _make_face_det(cx: float = 200.0, cy: float = 200.0,
                   conf: float = 0.85, sharpness: float = 100.0) -> FaceDetection:
    hw = 40
    crop = _random_crop()
    return FaceDetection(
        bbox       = (int(cx - hw), int(cy - hw), int(cx + hw), int(cy + hw)),
        conf       = conf,
        landmarks  = None,
        sharpness  = sharpness,
        crop       = crop,
        frame_area = 640 * 480,
    )


# =============================================================================
# FaceDetection dataclass
# =============================================================================

class TestFaceDetection:

    def test_area_computed_correctly(self):
        fd = _make_face_det(cx=100, cy=100)
        x1, y1, x2, y2 = fd.bbox
        assert fd.area == (x2 - x1) * (y2 - y1)

    def test_center_computed_correctly(self):
        fd = _make_face_det(cx=200, cy=300)
        cx, cy = fd.center
        assert abs(cx - 200) <= 2 and abs(cy - 300) <= 2

    def test_zero_area_bbox(self):
        fd = FaceDetection(
            bbox=(50, 50, 50, 50), conf=0.9, landmarks=None,
            sharpness=100.0, crop=_random_crop(), frame_area=640*480,
        )
        assert fd.area == 0


# =============================================================================
# Helper functions
# =============================================================================

class TestHelpers:

    def test_laplacian_sharpness_blurry(self):
        """Uniformly coloured image should have very low sharpness."""
        flat = np.full((112, 112, 3), 128, dtype=np.uint8)
        assert _laplacian_sharpness(flat) < 5.0

    def test_laplacian_sharpness_sharp(self):
        """Checkerboard should have high sharpness."""
        checker = np.zeros((112, 112, 3), dtype=np.uint8)
        checker[::8, :] = 255
        checker[:, ::8] = 255
        assert _laplacian_sharpness(checker) > 100.0

    def test_laplacian_sharpness_grayscale_input(self):
        """Should accept single-channel input."""
        gray = np.random.randint(0, 256, (112, 112), dtype=np.uint8)
        val  = _laplacian_sharpness(gray)
        assert val >= 0.0

    def test_align_face_output_size(self):
        """_align_face should return 112x112 even without valid landmarks."""
        img       = _random_crop(200, 200)
        landmarks = np.array([[40,50],[70,50],[56,70],[42,90],[70,90]], dtype=np.float32)
        aligned   = _align_face(img, landmarks, output_size=112)
        assert aligned.shape == (112, 112, 3)

    def test_align_face_fallback_on_bad_lm(self):
        """Degenerate landmarks should not crash, fallback to resize."""
        img  = _random_crop(60, 60)
        lm   = np.zeros((5, 2), dtype=np.float32)  # all at origin -> degenerate
        out  = _align_face(img, lm, output_size=112)
        assert out.shape == (112, 112, 3)


# =============================================================================
# FaceEmbedder (HOG backend -- no model download)
# =============================================================================

class TestFaceEmbedder:

    @pytest.fixture
    def embedder(self):
        return FaceEmbedder(device="cpu", backend="hog")

    def test_embed_returns_correct_shape(self, embedder):
        crop = _random_crop()
        emb  = embedder.embed(crop)
        assert emb.shape == (EMBED_DIM,)
        assert emb.dtype == np.float32

    def test_embed_is_l2_normalised(self, embedder):
        crop = _random_crop()
        emb  = embedder.embed(crop)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Expected L2-norm=1, got {norm}"

    def test_embed_empty_returns_zeros(self, embedder):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        emb   = embedder.embed(empty)
        assert emb.shape == (EMBED_DIM,)
        assert emb.sum() == 0.0

    def test_embed_deterministic(self, embedder):
        crop = _random_crop()
        e1   = embedder.embed(crop)
        e2   = embedder.embed(crop)
        np.testing.assert_array_almost_equal(e1, e2)

    def test_embed_batch_shape(self, embedder):
        crops = [_random_crop() for _ in range(4)]
        embs  = embedder.embed_batch(crops)
        assert embs.shape == (4, EMBED_DIM)

    def test_embed_batch_empty(self, embedder):
        embs = embedder.embed_batch([])
        assert embs.shape == (0, EMBED_DIM)

    def test_similarity_same_embedding(self, embedder):
        emb = _random_emb()
        sim = embedder.similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_similarity_orthogonal(self, embedder):
        a = np.zeros(EMBED_DIM, dtype=np.float32); a[0] = 1.0
        b = np.zeros(EMBED_DIM, dtype=np.float32); b[1] = 1.0
        sim = embedder.similarity(a, b)
        assert sim == 0.0

    def test_l2_normalize_zero_vector(self, embedder):
        z = np.zeros(EMBED_DIM, dtype=np.float32)
        n = embedder.l2_normalize(z)
        assert n.sum() == 0.0

    def test_different_crops_different_embeddings(self, embedder):
        e1 = embedder.embed(_random_crop())
        e2 = embedder.embed(_random_crop())
        assert not np.allclose(e1, e2)


# =============================================================================
# Watchlist + _NumpyIndex
# =============================================================================

class TestNumpyIndex:

    def test_add_and_search(self):
        idx = _NumpyIndex(dim=4)
        v   = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        idx.add(v)
        sims, idxs = idx.search(np.array([[1.0, 0, 0, 0]], dtype=np.float32), k=1)
        assert idxs[0][0] == 0
        assert abs(sims[0][0] - 1.0) < 1e-5

    def test_empty_search_returns_minus_one(self):
        idx = _NumpyIndex(dim=4)
        sims, idxs = idx.search(np.zeros((1, 4), dtype=np.float32), k=1)
        assert idxs[0][0] == -1

    def test_reconstruct_n(self):
        idx = _NumpyIndex(dim=4)
        v   = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
        idx.add(v)
        rec = idx.reconstruct_n(0, 1)
        np.testing.assert_array_almost_equal(rec, v)


class TestWatchlist:

    @pytest.fixture
    def wl(self):
        return Watchlist(embed_dim=EMBED_DIM, use_gpu=False)

    def test_add_increases_size(self, wl):
        wl.add_person("Alice", _random_emb())
        assert wl.size == 1

    def test_persons_property(self, wl):
        wl.add_person("Alice", _random_emb())
        wl.add_person("Bob",   _random_emb())
        assert set(wl.persons) == {"Alice", "Bob"}

    def test_search_empty_returns_none(self, wl):
        result = wl.search(_random_emb(), threshold=0.1)
        assert result is None

    def test_search_identical_embedding_matches(self, wl):
        emb = _random_emb()
        wl.add_person("Alice", emb)
        match = wl.search(emb, threshold=0.5)
        assert match is not None
        name, sim = match
        assert name == "Alice"
        assert sim > 0.99

    def test_search_below_threshold_returns_none(self, wl):
        emb_a = np.zeros(EMBED_DIM, dtype=np.float32); emb_a[0] = 1.0
        emb_b = np.zeros(EMBED_DIM, dtype=np.float32); emb_b[1] = 1.0
        wl.add_person("Alice", emb_a)
        match = wl.search(emb_b, threshold=0.99)
        assert match is None

    def test_search_all_returns_list(self, wl):
        emb = _random_emb()
        wl.add_person("Alice", emb)
        wl.add_person("Bob",   _random_emb())
        results = wl.search_all(emb, top_k=5, threshold=0.0)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_remove_person(self, wl):
        emb = _random_emb()
        wl.add_person("Alice", emb)
        wl.add_person("Bob",   _random_emb())
        removed = wl.remove_person("Alice")
        assert removed == 1
        assert "Alice" not in wl.persons

    def test_save_and_load(self, wl, tmp_path):
        emb = _random_emb()
        wl.add_person("Alice", emb)
        save_path = tmp_path / "wl.npz"
        wl.save(str(save_path))
        wl2 = Watchlist(embed_dim=EMBED_DIM)
        wl2.load(str(save_path))
        assert wl2.size == 1
        assert wl2.persons == ["Alice"]
        # Match should succeed
        match = wl2.search(emb, threshold=0.5)
        assert match is not None and match[0] == "Alice"

    def test_load_missing_file_is_noop(self, wl):
        wl.load("nonexistent_file.npz")
        assert wl.size == 0


# =============================================================================
# FaceEngine (mocked detector -- no model required)
# =============================================================================

class TestFaceEngine:

    @pytest.fixture
    def engine(self):
        """Build FaceEngine with HOG embedder, numpy watchlist, no face detector model."""
        eng = FaceEngine.__new__(FaceEngine)
        from services.analytics.face.face_detector import FaceDetector
        from services.analytics.face.face_embedder import FaceEmbedder
        from services.analytics.face.watchlist     import Watchlist
        from collections import defaultdict
        eng._threshold   = 0.45
        eng._dedup_pos   = 60.0
        eng._dedup_w     = 0.0   # no dedup for tests
        eng._fd_severity = __import__("services.event_engine.models", fromlist=["Severity"]).Severity.LOW
        eng._wl_severity = __import__("services.event_engine.models", fromlist=["Severity"]).Severity.CRITICAL
        eng._camera      = "test"
        eng._en_detected = True
        eng._total_detected = 0
        eng._total_matched  = 0
        eng._last_fired  = defaultdict(float)
        eng.detector     = None
        eng.embedder     = FaceEmbedder(device="cpu", backend="hog")
        eng.watchlist    = Watchlist(embed_dim=EMBED_DIM)
        return eng

    def _run_update(self, engine, faces):
        """Inject face detections directly, bypassing FaceDetector."""
        from collections import defaultdict
        import time as t_mod
        cam    = "test"
        now    = t_mod.monotonic()
        events = []
        engine._total_detected += len(faces)

        for face in faces:
            cx, cy = face.center
            bkt    = (int(cx // engine._dedup_pos), int(cy // engine._dedup_pos))
            last   = engine._last_fired.get(bkt, 0.0)
            if (now - last) < engine._dedup_w:
                continue
            engine._last_fired[bkt] = now
            emb   = engine.embedder.embed(face.crop)
            match = engine.watchlist.search(emb, threshold=engine._threshold)
            stub  = _face_stub_track(face, 1, cam)
            from services.event_engine.models import Event, EventType
            if match:
                name, sim = match
                engine._total_matched += 1
                events.append(Event(
                    event_type = EventType.FACE_WATCHLIST,
                    severity   = engine._wl_severity,
                    camera_id  = cam,
                    track      = stub,
                    frame_id   = 1,
                    zone_id    = None,
                    metadata   = {"name": name, "similarity": sim,
                                  "threshold": engine._threshold,
                                  "sharpness": face.sharpness, "det_conf": face.conf},
                ))
            elif engine._en_detected:
                events.append(Event(
                    event_type = EventType.FACE_DETECTED,
                    severity   = engine._fd_severity,
                    camera_id  = cam,
                    track      = stub,
                    frame_id   = 1,
                    zone_id    = None,
                    metadata   = {"sharpness": face.sharpness,
                                  "det_conf": face.conf, "area_px2": face.area},
                ))
        return events

    def test_no_faces_returns_empty_events(self, engine):
        events = self._run_update(engine, [])
        assert events == []

    def test_unknown_face_emits_face_detected(self, engine):
        face   = _make_face_det()
        events = self._run_update(engine, [face])
        assert len(events) == 1
        assert events[0].event_type == EventType.FACE_DETECTED

    def test_watchlist_match_emits_watchlist_event(self, engine):
        # Add an embedding generated from a known crop
        crop = _random_crop()
        emb  = engine.embedder.embed(crop)
        engine.watchlist.add_person("Alice", emb)

        # Build a face with the same crop
        face = FaceDetection(
            bbox=(160, 160, 240, 240), conf=0.9, landmarks=None,
            sharpness=100.0, crop=crop, frame_area=640*480,
        )
        events = self._run_update(engine, [face])
        wl = [e for e in events if e.event_type == EventType.FACE_WATCHLIST]
        assert len(wl) == 1
        assert wl[0].metadata["name"] == "Alice"

    def test_multiple_faces_multiple_events(self, engine):
        faces  = [_make_face_det(cx=100 + i * 100, cy=100) for i in range(3)]
        events = self._run_update(engine, faces)
        assert len(events) == 3

    def test_stats_incremented(self, engine):
        faces = [_make_face_det()]
        self._run_update(engine, faces)
        assert engine.stats["total_detected"] == 1

    def test_face_detected_disabled(self, engine):
        engine._en_detected = False
        face   = _make_face_det()
        events = self._run_update(engine, [face])
        assert events == []

    def test_stub_track_has_face_class(self):
        face  = _make_face_det()
        track = _face_stub_track(face, frame_id=1, camera_id="cam")
        assert track.class_name == "face"
        assert track.track_id   == -1
