"""
services/analytics/face/face_embedder.py
==========================================
IBVAP P6 -- Face embedding extractor.

Extracts 512-dim ArcFace embeddings from aligned 112x112 BGR face crops.

Backend: InsightFace's ArcFace model (buffalo_l pack includes it).
If InsightFace is not available, falls back to a simple HOG descriptor
that still allows functional testing (lower accuracy, CPU-only).

The embedding is L2-normalised so cosine similarity = dot product,
allowing direct use with FAISS IndexFlatIP (inner product = cosine).

Usage
-----
    embedder = FaceEmbedder(device="cuda:0")
    emb = embedder.embed(face_crop_112x112_bgr)   # np.ndarray (512,)
    sim = embedder.similarity(emb_a, emb_b)        # float in [0, 1]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Literal

import cv2
import numpy as np
from loguru import logger


# Cosine similarity threshold defaults
SAME_PERSON_THRESHOLD = 0.45   # sim > this -> same person


class FaceEmbedder:
    """
    Extracts L2-normalised 512-d ArcFace embeddings from face crops.

    Parameters
    ----------
    device      : "cuda:0" | "cpu"
    backend     : "insightface" | "hog" (fallback, for testing)
    model_pack  : InsightFace model pack (must match FaceDetector's pack)
    """

    EMBED_DIM = 512

    def __init__(
        self,
        device:     str = "cuda:0",
        backend:    Literal["insightface", "hog"] = "insightface",
        model_pack: str = "buffalo_l",
    ) -> None:
        self._device  = device
        self._backend = backend
        self._model   = None
        self._ctx_id  = 0 if "cuda" in device else -1

        if backend == "insightface":
            self._init_insightface(model_pack)

        if self._model is None and backend == "insightface":
            logger.warning("InsightFace embedder unavailable — falling back to HOG.")
            self._backend = "hog"

        logger.info(f"FaceEmbedder ready — backend={self._backend}, device={device}")

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Compute L2-normalised embedding from an aligned BGR face crop.

        Parameters
        ----------
        crop : (112, 112, 3) uint8 BGR image

        Returns
        -------
        np.ndarray shape (512,) float32, L2-normalised.
        """
        if crop is None or crop.size == 0:
            return np.zeros(self.EMBED_DIM, dtype=np.float32)

        if self._backend == "insightface" and self._model is not None:
            return self._embed_insightface(crop)
        else:
            return self._embed_hog(crop)

    def embed_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Embed multiple crops at once.

        Returns
        -------
        np.ndarray shape (N, 512) float32.
        """
        if not crops:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)
        return np.stack([self.embed(c) for c in crops])

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised embeddings.
        Returns a float in [0, 1] (1 = identical, 0 = orthogonal).
        """
        return float(np.clip(np.dot(a, b), 0.0, 1.0))

    @staticmethod
    def l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return np.zeros_like(v)
        # pyrefly: ignore [no-any-return-implicit]
        return v / norm

    @property
    def is_ready(self) -> bool:
        return self._backend == "hog" or self._model is not None

    # ── Backends ──────────────────────────────────────────────────────────────

    def _init_insightface(self, model_pack: str) -> None:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name       = model_pack,
                providers  = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                              if "cuda" in self._device
                              else ["CPUExecutionProvider"]),
            )
            app.prepare(ctx_id=self._ctx_id, det_size=(112, 112))
            # Get just the recognition model from the loaded models
            self._model = app.models.get("recognition")
            if self._model is None:
                # Try any key containing 'rec'
                for k, v in app.models.items():
                    if "rec" in k.lower():
                        self._model = v
                        break
            if self._model is not None:
                logger.success(f"InsightFace ArcFace recognition model loaded")
            else:
                logger.warning("ArcFace recognition model not found in pack")
        except Exception as exc:
            logger.warning(f"InsightFace embedder init failed: {exc}")
            self._model = None

    def _embed_insightface(self, crop: np.ndarray) -> np.ndarray:
        """Use InsightFace's ArcFace model to embed a 112x112 face crop."""
        try:
            # Ensure 112x112
            if crop.shape[:2] != (112, 112):
                crop = cv2.resize(crop, (112, 112))
            # pyrefly: ignore [missing-attribute]
            emb = self._model.get_feat(crop)   # returns (512,) or (1, 512)
            if emb.ndim == 2:
                emb = emb[0]
            return self.l2_normalize(emb.astype(np.float32))
        except Exception as exc:
            logger.warning(f"InsightFace embed failed: {exc}")
            return self._embed_hog(crop)

    def _embed_hog(self, crop: np.ndarray) -> np.ndarray:
        """
        Pure-numpy pseudo-embedding (CPU fallback, no cv2.HOGDescriptor needed).

        Pipeline:
          1. Resize to 32x32 greyscale
          2. Compute pixel-wise gradient magnitudes & orientations (numpy)
          3. Histogram over 4x4 spatial cells, 8 orientation bins -> 256 dims
          4. Flatten, pad to 512, L2-normalise
        """
        if crop.shape[:2] != (112, 112):
            crop = cv2.resize(crop, (112, 112))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 32x32 for speed
        small = cv2.resize(gray, (32, 32))

        # Gradients
        gx   = np.diff(small, axis=1, prepend=small[:, :1])
        gy   = np.diff(small, axis=0, prepend=small[:1, :])
        mag  = np.sqrt(gx ** 2 + gy ** 2)
        ori  = (np.degrees(np.arctan2(gy, gx + 1e-6)) % 180)

        # Cell histogram: 4x4 grid of 8x8 pixels, 8 orientation bins
        n_cells = 4
        cell_px = 8    # 4*8 = 32
        n_bins  = 8
        hist    = np.zeros((n_cells, n_cells, n_bins), dtype=np.float32)

        for r in range(n_cells):
            for c in range(n_cells):
                m_cell = mag[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
                o_cell = ori[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
                hist[r, c, :] = np.histogram(
                    o_cell.flatten(), bins=n_bins,
                    range=(0, 180), weights=m_cell.flatten()
                )[0]

        desc = hist.flatten()   # 4*4*8 = 128 dims

        # Also include raw pixel intensities as extra features (flattened 32x32 = 1024)
        # Subsample to add 384 more dims -> 512 total
        pixels = small.flatten()
        step   = max(1, len(pixels) // (self.EMBED_DIM - len(desc)))
        extra  = pixels[::step][:self.EMBED_DIM - len(desc)]
        desc   = np.concatenate([desc, extra.astype(np.float32)])

        # Pad or truncate to EMBED_DIM
        if len(desc) >= self.EMBED_DIM:
            vec = desc[:self.EMBED_DIM]
        else:
            vec = np.pad(desc, (0, self.EMBED_DIM - len(desc)))

        return self.l2_normalize(vec.astype(np.float32))

