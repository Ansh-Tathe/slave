"""
services/analytics/night/night_enhancer.py
===========================================
IBVAP P4 — Low-light / night-time frame enhancer.

Four backends (selected by NightEnhancer(mode=...)):

  "clahe"     (always available, zero extra dependencies)
      ─ CLAHE on the L channel of LAB colour space.
        Fast (~0.5 ms/frame on CPU).

  "gamma"     (CPU-only, good for deep-dark frames)
      ─ Adaptive gamma correction: brightness is estimated from the
        mean LAB-L value, then a matching gamma LUT is applied.

  "combined"  (default — best quality on CPU)
      ─ gamma → CLAHE in sequence.

  "dnn"       (highest quality, requires Zero-DCE ONNX weights)
      ─ Loaded via cv2.dnn. Falls back to "combined" if model missing.

Darkness detection
------------------
is_dark(frame) → True when mean LAB-L < dark_threshold.
mean_luminance(frame) → float LAB-L mean (0–255).

Usage
-----
    enhancer = NightEnhancer(mode="combined")
    if enhancer.is_dark(frame):
        bright = enhancer.enhance(frame)
    else:
        bright = frame
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


# ── Tunable constants ─────────────────────────────────────────────────────────

_DARK_THRESHOLD: float      = 80.0    # mean LAB-L < this → "dark frame"
_GAMMA_MIN: float           = 0.35    # aggressive boost for very dark frames
_GAMMA_MAX: float           = 0.85    # mild boost for moderately dark frames
_CLAHE_CLIP_LIMIT: float    = 2.5
_CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)


# ── NightEnhancer ─────────────────────────────────────────────────────────────

class NightEnhancer:
    """
    Low-light image enhancement for IBVAP camera feeds.

    Parameters
    ----------
    mode          : "clahe" | "gamma" | "combined" | "dnn"
    dark_threshold: mean LAB-L below this → frame is considered dark (0-255)
    dnn_model_path: path to Zero-DCE ONNX/DNN model  (mode="dnn" only)
    clahe_clip    : CLAHE clip limit (default 2.5)
    clahe_grid    : CLAHE tile grid size (default (8,8))
    """

    def __init__(
        self,
        mode:           Literal["clahe", "gamma", "combined", "dnn"] = "combined",
        dark_threshold: float = _DARK_THRESHOLD,
        dnn_model_path: Optional[str] = None,
        clahe_clip:     float = _CLAHE_CLIP_LIMIT,
        clahe_grid:     Tuple[int, int] = _CLAHE_GRID_SIZE,
    ) -> None:
        self._dark_thresh = dark_threshold
        self._mode = mode

        # CLAHE object (created once, thread-safe for reads)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_grid,
        )

        # DNN backend
        self._dnn_net = None
        if mode == "dnn":
            self._dnn_net = self._load_dnn(dnn_model_path)
            if self._dnn_net is None:
                logger.warning(
                    "DNN model not loaded — falling back to 'combined' mode."
                )
                self._mode = "combined"

        logger.info(
            f"NightEnhancer ready — mode={self._mode}, "
            f"dark_threshold={self._dark_thresh:.0f}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_dark(self, frame: np.ndarray) -> bool:
        """
        Return True when the frame's mean luminance is below dark_threshold.

        Samples at 1/4 resolution for speed (~0.1 ms on CPU).
        """
        if frame is None or frame.size == 0:
            return False
        return self.mean_luminance(frame) < self._dark_thresh

    def mean_luminance(self, frame: np.ndarray) -> float:
        """
        Return the mean LAB-L channel value (0–255).
        Sampled at 1/4 resolution for speed (min 1x1 px).
        """
        h, w = frame.shape[:2]
        new_w = max(1, w // 4)
        new_h = max(1, h // 4)
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        lab   = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        return float(lab[:, :, 0].mean())

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Return an enhanced copy of the dark BGR frame.
        Input is never modified.
        """
        if frame is None or frame.size == 0:
            return frame

        if self._mode == "clahe":
            return self._apply_clahe(frame)
        elif self._mode == "gamma":
            return self._apply_gamma(frame)
        elif self._mode == "dnn" and self._dnn_net is not None:
            return self._apply_dnn(frame)
        else:  # combined (default)
            return self._apply_combined(frame)

    # ── Backends ──────────────────────────────────────────────────────────────

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE on L channel (LAB space) → colour-balanced brightness boost."""
        lab         = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b     = cv2.split(lab)
        l_eq        = self._clahe.apply(l)
        merged      = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Adaptive gamma correction — gamma chosen from frame brightness."""
        mean_l = self.mean_luminance(frame)
        gamma  = self._adaptive_gamma(mean_l)
        lut    = self._build_gamma_lut(gamma)
        return cv2.LUT(frame, lut)

    def _apply_combined(self, frame: np.ndarray) -> np.ndarray:
        """Gamma correction followed by CLAHE for best CPU-side quality."""
        return self._apply_clahe(self._apply_gamma(frame))

    def _apply_dnn(self, frame: np.ndarray) -> np.ndarray:
        """
        Zero-DCE (or any compatible network) via cv2.dnn.
        Network must accept (1, 3, H, W) float32 [0-1] and output the same.
        Falls back to combined on any error.
        """
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                frame, scalefactor=1.0 / 255.0,
                size=(512, 512),
                swapRB=True, crop=False,
            )
            self._dnn_net.setInput(blob)
            out = self._dnn_net.forward()    # (1, 3, 512, 512)
            out = out[0].transpose(1, 2, 0)  # (512, 512, 3) RGB 0-1
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception as exc:
            logger.warning(f"DNN enhancement failed ({exc}); using combined.")
            return self._apply_combined(frame)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _adaptive_gamma(self, mean_l: float) -> float:
        """
        Linearly map mean luminance to a gamma value.
          mean_l = 0             → _GAMMA_MIN  (brightest boost)
          mean_l = dark_threshold→ _GAMMA_MAX  (mildest boost)
        """
        clamped = max(0.0, min(mean_l, self._dark_thresh))
        frac    = clamped / self._dark_thresh
        return _GAMMA_MIN + frac * (_GAMMA_MAX - _GAMMA_MIN)

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """
        Standard gamma LUT: output = (input / 255)^gamma * 255.
        gamma < 1.0  → brightens  (used for dark frames)
        gamma = 1.0  → identity
        gamma > 1.0  → darkens
        """
        g = max(gamma, 1e-6)
        return np.array(
            [((i / 255.0) ** g) * 255 for i in range(256)],
            dtype=np.uint8,
        )

    @staticmethod
    def _load_dnn(model_path: Optional[str]) -> Optional[cv2.dnn.Net]:
        if not model_path:
            logger.warning("DNN mode: no model_path provided.")
            return None
        p = Path(model_path)
        if not p.exists():
            logger.warning(
                f"DNN model not found: {p}. "
                "Download Zero-DCE ONNX weights and pass via --dnn-model."
            )
            return None
        try:
            net = cv2.dnn.readNet(str(p))
            logger.success(f"DNN night model loaded: {p}")
            return net
        except Exception as exc:
            logger.error(f"Failed to load DNN model '{p}': {exc}")
            return None
