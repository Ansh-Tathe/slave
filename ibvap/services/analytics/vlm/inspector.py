"""
services/analytics/vlm/inspector.py
===================================
IBVAP — Vision-Language Model (VLM) & Visual Attribute Inspector.
Analyzes snapshot crops to automatically describe clothing colors,
accessories, carried items, and suspicious behavior for surveillance logs.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
from loguru import logger


@dataclass
class VLMInspectionResult:
    description: str
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    accessories: List[str] = field(default_factory=list)
    provider: str = "cv_heuristic"
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VLMInspector:
    """
    Multi-provider visual attribute analyzer.
    Supports local Ollama VLMs (Moondream, LLaVA, Qwen2-VL) with automatic
    high-speed CV color histogram & silhouette fallback.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        preferred_model: str = "moondream",
    ) -> None:
        self.ollama_url = ollama_url
        self.preferred_model = preferred_model

    def inspect_image(self, image: np.ndarray | str | Path) -> VLMInspectionResult:
        """
        Inspect an image or file path and return visual descriptions.
        """
        t0 = time.perf_counter()

        # Load image if path provided
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                return VLMInspectionResult(
                    description="Image crop not found on disk",
                    provider="error",
                    latency_ms=0.0,
                )
            img = cv2.imread(str(p))
            if img is None:
                return VLMInspectionResult(
                    description="Failed to decode image file",
                    provider="error",
                    latency_ms=0.0,
                )
        else:
            img = image

        # 1. Try local Ollama VLM if available
        ollama_res = self._try_ollama_inspection(img)
        if ollama_res is not None:
            ollama_res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return ollama_res

        # 2. High-speed CV attribute heuristic fallback
        cv_res = self._analyze_cv_attributes(img)
        cv_res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return cv_res

    def _try_ollama_inspection(self, img: np.ndarray) -> Optional[VLMInspectionResult]:
        """Attempt to query local Ollama vision endpoint."""
        try:
            # Check if Ollama is running
            with httpx.Client(timeout=1.0) as client:
                tags_res = client.get(f"{self.ollama_url}/api/tags")
                if tags_res.status_code != 200:
                    return None

                models = [m.get("name", "") for m in tags_res.json().get("models", [])]
                # Look for vision model match
                selected_model = None
                for m in models:
                    if any(vm in m.lower() for vm in ["moondream", "llava", "qwen2-vl", "vision"]):
                        selected_model = m
                        break

                if not selected_model:
                    return None

                # Encode image to JPEG base64
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64_img = base64.b64encode(buf).decode("utf-8")

                prompt = (
                    "You are a military border security AI. Describe this surveillance target in 1-2 concise sentences. "
                    "Specify: target type (person/vehicle), clothing colors (upper and lower body), and any items carried."
                )

                payload = {
                    "model": selected_model,
                    "prompt": prompt,
                    "images": [b64_img],
                    "stream": False,
                }

                gen_res = client.post(f"{self.ollama_url}/api/generate", json=payload, timeout=12.0)
                if gen_res.status_code == 200:
                    text = gen_res.json().get("response", "").strip()
                    if text:
                        return VLMInspectionResult(
                            description=text,
                            provider=f"ollama:{selected_model}",
                        )
        except Exception:
            pass  # Fall back to CV heuristic

        return None

    def _analyze_cv_attributes(self, img: np.ndarray) -> VLMInspectionResult:
        """
        Fast classical computer vision attribute extractor:
        Segments upper and lower body to determine clothing colors and aspect ratio.
        """
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return VLMInspectionResult(
                description="Target too small or distant for detailed visual attribute analysis.",
                provider="cv_heuristic",
            )

        # Upper body (top 15% to 50%)
        upper_crop = img[int(h * 0.15) : int(h * 0.50), int(w * 0.1) : int(w * 0.9)]
        # Lower body (bottom 50% to 90%)
        lower_crop = img[int(h * 0.50) : int(h * 0.90), int(w * 0.1) : int(w * 0.9)]

        upper_color = self._get_dominant_color_name(upper_crop) if upper_crop.size > 0 else "neutral"
        lower_color = self._get_dominant_color_name(lower_crop) if lower_crop.size > 0 else "neutral"

        aspect_ratio = round(h / float(w), 2)
        accessories = []

        # Check for potential backpack / baggage if crop has lateral width expansion
        if aspect_ratio < 1.8:
            accessories.append("possible baggage/backpack")

        desc = (
            f"Detected individual wearing {upper_color} upper clothing and {lower_color} lower garment. "
            f"Aspect ratio: {aspect_ratio} (silhouette posture normal)."
        )
        if accessories:
            desc += f" Carrying {', '.join(accessories)}."

        return VLMInspectionResult(
            description=desc,
            upper_color=upper_color,
            lower_color=lower_color,
            accessories=accessories,
            provider="cv_heuristic",
        )

    def _get_dominant_color_name(self, bgr_crop: np.ndarray) -> str:
        """Identify dominant color name from BGR crop using HSV color quantization."""
        if bgr_crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Mean saturation and value
        mean_s = np.mean(s)
        mean_v = np.mean(v)

        if mean_v < 45:
            return "black / dark"
        if mean_v > 210 and mean_s < 35:
            return "white / light"
        if mean_s < 35:
            return "gray / neutral"

        # Hue analysis (H in 0..179)
        mean_h = np.median(h)
        if mean_h < 12 or mean_h >= 170:
            return "red"
        elif 12 <= mean_h < 25:
            return "orange"
        elif 25 <= mean_h < 35:
            return "yellow / beige"
        elif 35 <= mean_h < 85:
            return "green / olive"
        elif 85 <= mean_h < 130:
            return "blue / navy"
        elif 130 <= mean_h < 170:
            return "purple / violet"

        return "dark neutral"
