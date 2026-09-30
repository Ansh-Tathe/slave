"""
services/analytics/anpr/ocr_engine.py
========================================
IBVAP P3 — OCR engine for license plate images using EasyOCR.

Pipeline per plate crop:
  1. Resize to standard width (320 px) — keeps OCR model in its sweet spot
  2. CLAHE (Contrast Limited Adaptive Histogram Equalisation) — helps on
     plates with glare, shadows, or faded paint
  3. Optional denoise (cv2.fastNlMeansDenoising)
  4. EasyOCR  (GPU-accelerated, English alphabet)
  5. Post-process: filter short results, merge multi-line reads

Performance on RTX 4050:
  ~ 8–15 ms per plate crop (GPU, batch=1)

Limitations:
  - Hindi/Devanagari characters on old plates may not read correctly
  - Very blurry or rotated plates (>20°) will have poor accuracy
  - Night plates without IR illumination will fail — P4 will pre-enhance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from services.analytics.anpr.plate_detector import PlateCandidate


# ── OCR result ────────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    """Raw output from the OCR step for one plate crop."""
    text:       str             # cleaned concatenated text
    confidence: float           # average char-level confidence (0–1)
    candidate:  PlateCandidate  # reference back to the plate region


# ── Image preprocessing ───────────────────────────────────────────────────────

_TARGET_WIDTH  = 320   # resize all plate crops to this width before OCR
_MIN_CONF      = 0.35  # discard OCR boxes below this confidence


def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """
    Enhance a plate crop for OCR.

    1. Resize to _TARGET_WIDTH (preserve aspect ratio)
    2. Convert to grayscale
    3. CLAHE for local contrast enhancement
    4. Mild Gaussian blur to reduce JPEG artefacts
    Returns a grayscale uint8 image.
    """
    if crop is None or crop.size == 0:
        return np.zeros((_TARGET_WIDTH // 4, _TARGET_WIDTH), dtype=np.uint8)

    h, w = crop.shape[:2]
    if w == 0:
        return np.zeros((_TARGET_WIDTH // 4, _TARGET_WIDTH), dtype=np.uint8)

    # Resize
    scale = _TARGET_WIDTH / w
    new_h = max(1, int(h * scale))
    resized = cv2.resize(crop, (_TARGET_WIDTH, new_h), interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # CLAHE — good for plates with uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Mild blur to reduce pixel noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return blurred


# ── OCR Engine ────────────────────────────────────────────────────────────────

class PlateOCR:
    """
    EasyOCR-based OCR for license plate images.

    Parameters
    ----------
    use_gpu : bool  — use GPU for inference (default True)
    languages : list[str] — EasyOCR language codes (default ["en"])
    min_confidence : float — discard OCR boxes below this threshold
    """

    def __init__(
        self,
        use_gpu:        bool = True,
        languages:      List[str] = None,
        min_confidence: float = _MIN_CONF,
    ) -> None:
        import easyocr
        langs = languages or ["en"]
        logger.info(
            f"Loading EasyOCR (gpu={use_gpu}, lang={langs}) "
            "— first run downloads ~100 MB models …"
        )
        self._reader     = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
        self._min_conf   = min_confidence
        logger.success("EasyOCR ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def read_candidates(
        self,
        candidates: List[PlateCandidate],
    ) -> List[OCRResult]:
        """
        Run OCR on a list of plate crop candidates.

        Returns one OCRResult per candidate that produced a readable string.
        Empty / low-confidence reads are filtered out.
        """
        results: List[OCRResult] = []

        for cand in candidates:
            result = self._read_one(cand)
            if result is not None:
                results.append(result)

        return results

    def read_image(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Low-level: run OCR on a single (pre-processed or raw) image.

        Returns (text, confidence).
        """
        try:
            raw = self._reader.readtext(img, detail=1, paragraph=False)
        except Exception as e:
            logger.warning(f"EasyOCR error: {e}")
            return "", 0.0

        if not raw:
            return "", 0.0

        texts  = []
        confs  = []
        for (_bbox, text, conf) in raw:
            if conf >= self._min_conf and text.strip():
                texts.append(text.strip())
                confs.append(conf)

        if not texts:
            return "", 0.0

        merged_text = " ".join(texts)
        avg_conf    = sum(confs) / len(confs)
        return merged_text, avg_conf

    # ── Private ───────────────────────────────────────────────────────────────

    def _read_one(self, cand: PlateCandidate) -> Optional[OCRResult]:
        """Pre-process then OCR one candidate."""
        processed = preprocess_plate(cand.crop)
        text, conf = self.read_image(processed)

        if not text or len(text.replace(" ", "")) < 4:
            return None   # too short to be a valid plate

        return OCRResult(
            text       = text,
            confidence = conf,
            candidate  = cand,
        )
