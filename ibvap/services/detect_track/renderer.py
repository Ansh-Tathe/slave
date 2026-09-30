"""
services/detect_track/renderer.py
===================================
IBVAP P1 — Annotates frames with bounding boxes, track IDs, class labels,
confidence scores, FPS counter, and GPU memory usage.

Design:
  - Each COCO class has a unique BGR colour (defined in models.py)
  - Track ID is shown above the box in the same colour
  - Overlay (top-left): frame_id, FPS, GPU mem
  - All drawing is in-place on a copy of the input frame (never mutates input)
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from services.detect_track.models import CLASS_COLORS, DEFAULT_COLOR, Track


# ── Font config ───────────────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 2
_PAD        = 4   # pixels of padding around text background


class Renderer:
    """
    Stateful renderer: tracks per-frame FPS via a rolling window.

    Usage:
        renderer = Renderer()
        annotated = renderer.draw(frame, tracks, frame_id=42)
        cv2.imshow("IBVAP", annotated)
    """

    def __init__(self, fps_window: int = 30) -> None:
        self._ts: deque[float] = deque(maxlen=fps_window)

    def draw(
        self,
        frame:      np.ndarray,
        tracks:     List[Track],
        frame_id:   int = 0,
        extra_info: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw all tracks onto a copy of *frame*.

        Parameters
        ----------
        frame     : BGR uint8 ndarray
        tracks    : active tracks from ByteTracker
        frame_id  : current frame counter
        extra_info: optional one-line string appended to the HUD

        Returns
        -------
        Annotated BGR uint8 ndarray (same size as input).
        """
        out = frame.copy()
        self._ts.append(time.monotonic())

        for t in tracks:
            self._draw_track(out, t)

        self._draw_hud(out, frame_id, len(tracks), extra_info)
        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _draw_track(img: np.ndarray, t: Track) -> None:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        color = CLASS_COLORS.get(t.class_id, DEFAULT_COLOR)

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, _THICKNESS)

        # Label: "ID:42  person  0.91"
        label = f"#{t.track_id}  {t.class_name}  {t.conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)

        # Background pill above the box
        by1 = max(0, y1 - th - 2 * _PAD)
        by2 = y1
        bx2 = min(img.shape[1], x1 + tw + 2 * _PAD)
        cv2.rectangle(img, (x1, by1), (bx2, by2), color, cv2.FILLED)

        # Text (white on coloured background)
        cv2.putText(
            img, label,
            (x1 + _PAD, by2 - _PAD),
            _FONT, _FONT_SCALE,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Small filled circle at centroid
        cx, cy = int(t.center[0]), int(t.center[1])
        cv2.circle(img, (cx, cy), 4, color, cv2.FILLED)

    def _draw_hud(
        self,
        img:        np.ndarray,
        frame_id:   int,
        n_tracks:   int,
        extra_info: Optional[str],
    ) -> None:
        """Top-left overlay: FPS | frame | tracks | GPU mem."""
        fps = self._calc_fps()

        try:
            import torch
            free_b, total_b = torch.cuda.mem_get_info(0)
            used_gb  = (total_b - free_b) / 1024**3
            total_gb = total_b / 1024**3
            gpu_str  = f"GPU {used_gb:.1f}/{total_gb:.1f}GB"
        except Exception:
            gpu_str = ""

        lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {frame_id}",
            f"Tracks: {n_tracks}",
        ]
        if gpu_str:
            lines.append(gpu_str)
        if extra_info:
            lines.append(extra_info)

        y = 20
        for line in lines:
            # Dark background for readability
            (tw, th), _ = cv2.getTextSize(line, _FONT, _FONT_SCALE, 1)
            cv2.rectangle(img, (8, y - th - 2), (8 + tw + 4, y + 2),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(img, line, (10, y), _FONT, _FONT_SCALE,
                        (0, 255, 120), 1, cv2.LINE_AA)
            y += th + 8

    def _calc_fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        elapsed = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0
