"""
services/analytics/fence/zone_renderer.py
===========================================
IBVAP P2 — Draws fence zones and tripwires as overlays on frames.

Polygon zones  → semi-transparent filled polygon + border + name label
Tripwire lines → coloured line + arrow + name label
Alert flash    → red border flash when a CRITICAL/HIGH event fires
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from services.analytics.fence.fence_engine import FenceEngine
from services.event_engine.models          import Event, Severity


class ZoneRenderer:
    """
    Draws fence overlays onto frames.

    Parameters
    ----------
    fence : FenceEngine — provides zone/wire config and pixel dimensions.
    alpha : polygon fill opacity (0 = transparent, 1 = opaque).
    """

    # Colour palette for polygon zones (BGR)
    _ZONE_COLORS = [
        (0,   180, 255),   # orange
        (255, 100,   0),   # blue
        (0,   220,  50),   # green
        (180,   0, 255),   # purple
        (0,   220, 220),   # yellow
    ]
    _WIRE_COLOR  = (0, 0, 220)      # red for tripwires
    _ALERT_COLOR = (0, 0, 255)      # bright red for flash
    _ALERT_DUR   = 0.5              # seconds to flash border

    def __init__(self, fence: FenceEngine, alpha: float = 0.25) -> None:
        self._fence    = fence
        self._alpha    = alpha
        self._fw       = fence.frame_w
        self._fh       = fence.frame_h
        self._last_alert_t: float = 0.0   # monotonic time of last alert

    def trigger_alert(self) -> None:
        """Call this when a CRITICAL or HIGH event fires to trigger the flash."""
        self._last_alert_t = time.monotonic()

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all zones and wires onto a copy of *frame*.
        Returns annotated frame (same shape, uint8 BGR).
        """
        out = frame.copy()

        self._draw_polygons(out)
        self._draw_tripwires(out)
        self._draw_alert_flash(out)

        return out

    # ── Private ───────────────────────────────────────────────────────────────

    def _norm_to_px(self, x: float, y: float) -> tuple[int, int]:
        return (int(x * self._fw), int(y * self._fh))

    def _draw_polygons(self, img: np.ndarray) -> None:
        overlay = img.copy()
        for i, zone in enumerate(self._fence.zone_cfgs):
            color = self._ZONE_COLORS[i % len(self._ZONE_COLORS)]
            pts   = np.array(
                [self._norm_to_px(x, y) for x, y in zone.polygon],
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            # Semi-transparent fill
            cv2.fillPoly(overlay, [pts], color)

            # Solid border
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            # Name label at centroid
            cx = int(np.mean([p[0][0] for p in pts]))
            cy = int(np.mean([p[0][1] for p in pts]))
            self._label(img, zone.name, cx, cy, color)

        cv2.addWeighted(overlay, self._alpha, img, 1 - self._alpha, 0, img)

    def _draw_tripwires(self, img: np.ndarray) -> None:
        for wire in self._fence.wire_cfgs:
            p1 = self._norm_to_px(*wire.start)
            p2 = self._norm_to_px(*wire.end)
            color = self._WIRE_COLOR

            # Main line
            cv2.line(img, p1, p2, color, 3)

            # Arrow head at midpoint
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.arrowedLine(img, p1, p2, color, 2, tipLength=0.03)

            # Name label
            self._label(img, wire.name, mid[0], mid[1] - 12, color)

    def _draw_alert_flash(self, img: np.ndarray) -> None:
        elapsed = time.monotonic() - self._last_alert_t
        if elapsed < self._ALERT_DUR:
            # Pulsing red border
            thickness = max(4, int(12 * (1 - elapsed / self._ALERT_DUR)))
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1),
                          self._ALERT_COLOR, thickness)

    @staticmethod
    def _label(
        img:   np.ndarray,
        text:  str,
        cx:    int,
        cy:    int,
        color: tuple,
    ) -> None:
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        x = max(0, cx - tw // 2)
        y = max(th, cy)
        cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + 2),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(img, text, (x, y), font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
