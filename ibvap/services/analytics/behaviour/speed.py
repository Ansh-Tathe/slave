"""
services/analytics/behaviour/speed.py
=======================================
IBVAP P5 -- Speed / velocity estimator.

Emits:
  RUNNING      -- track velocity exceeds `run_threshold_px_per_s`
  WRONG_WAY    -- vehicle moving opposite to configured allowed_direction
                  (only checked if allowed_direction != "any")

Algorithm
---------
Velocity is estimated as the Euclidean distance between successive
centroids divided by the elapsed time.  A short Exponential Moving
Average (EMA) smooths out per-frame jitter.

  v_smooth[t] = alpha * v_raw[t] + (1 - alpha) * v_smooth[t-1]

Direction (for WRONG_WAY) is the angle (degrees) of the velocity
vector measured from the positive X-axis.  A track is "wrong way"
if its smoothed direction differs from `allowed_direction` by more
than `direction_tolerance_deg`.

allowed_direction values: "left", "right", "up", "down", "any"
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from services.detect_track.models import Track
from services.event_engine.models  import Event, EventType, Severity


# Direction angles (degrees, 0=right, 90=up, 180=left, 270=down)
_DIR_ANGLE = {
    "right": 0.0,
    "up":    90.0,
    "left":  180.0,
    "down":  270.0,
}


@dataclass
class _SpeedState:
    prev_cx:      float    # previous centroid X
    prev_cy:      float    # previous centroid Y
    prev_time:    float    # monotonic timestamp of previous frame
    v_smooth:     float = 0.0   # EMA-smoothed speed (px/s)
    dir_smooth:   float = 0.0   # EMA-smoothed direction (degrees)
    run_alerted:  bool  = False
    run_alert_t:  float = 0.0   # last RUNNING alert time
    ww_alerted:   bool  = False
    ww_alert_t:   float = 0.0   # last WRONG_WAY alert time


class SpeedDetector:
    """
    Estimates per-track velocity and emits RUNNING / WRONG_WAY events.

    Parameters
    ----------
    run_threshold_px_per_s   : speed above this -> RUNNING
    allowed_direction        : "any" | "left" | "right" | "up" | "down"
    direction_tolerance_deg  : max deviation (degrees) from allowed direction
    wrong_way_classes        : class names to check for wrong-way (vehicles)
    running_classes          : class names to check for running (persons)
    ema_alpha                : EMA smoothing factor (0=no smoothing, 1=no memory)
    min_frames               : frames before speed is trusted
    dedup_window_s           : suppress repeat events per track
    run_severity             : severity for RUNNING
    wrong_way_severity       : severity for WRONG_WAY
    camera_id                : used in emitted events
    """

    def __init__(
        self,
        run_threshold_px_per_s:  float = 150.0,
        allowed_direction:       str = "any",
        direction_tolerance_deg: float = 60.0,
        wrong_way_classes:       Optional[List[str]] = None,
        running_classes:         Optional[List[str]] = None,
        ema_alpha:               float = 0.4,
        min_frames:              int = 3,
        dedup_window_s:          float = 10.0,
        run_severity:            Severity = Severity.MEDIUM,
        wrong_way_severity:      Severity = Severity.HIGH,
        camera_id:               str = "unknown",
    ) -> None:
        self._run_thr     = run_threshold_px_per_s
        self._allowed_dir = allowed_direction.lower()
        self._dir_tol     = direction_tolerance_deg
        self._ww_classes  = set(wrong_way_classes or ["car", "truck", "bus", "motorcycle"])
        self._run_classes = set(running_classes or ["person"])
        self._alpha       = ema_alpha
        self._min_frames  = min_frames
        self._dedup_w     = dedup_window_s
        self._run_sev     = run_severity
        self._ww_sev      = wrong_way_severity
        self._camera      = camera_id
        self._state: Dict[int, _SpeedState] = {}
        self._frame_count: Dict[int, int]   = {}

        logger.info(
            f"SpeedDetector ready — run_threshold={run_threshold_px_per_s}px/s, "
            f"allowed_dir={allowed_direction}, ww_classes={self._ww_classes}"
        )

    def update(
        self,
        tracks:    List[Track],
        frame_id:  int,
        camera_id: Optional[str] = None,
    ) -> List[Event]:
        cam    = camera_id or self._camera
        now    = time.monotonic()
        events: List[Event] = []

        active_ids: Set[int] = {t.track_id for t in tracks}

        # Prune disappeared tracks
        for tid in list(self._state.keys()):
            if tid not in active_ids:
                del self._state[tid]
                self._frame_count.pop(tid, None)

        for track in tracks:
            tid   = track.track_id
            cx, cy = track.center

            if tid not in self._state:
                self._state[tid] = _SpeedState(
                    prev_cx=cx, prev_cy=cy, prev_time=now
                )
                self._frame_count[tid] = 0
                continue

            s  = self._state[tid]
            dt = now - s.prev_time
            if dt < 1e-6:
                continue   # same timestamp, skip

            # Raw speed
            dx       = cx - s.prev_cx
            dy       = cy - s.prev_cy
            dist     = math.sqrt(dx * dx + dy * dy)
            v_raw    = dist / dt

            # EMA smooth
            s.v_smooth  = self._alpha * v_raw + (1 - self._alpha) * s.v_smooth

            # Direction (atan2 with y-axis inverted for image coords)
            if dist > 0.5:   # only update direction if actually moving
                angle_raw   = math.degrees(math.atan2(-dy, dx)) % 360
                s.dir_smooth = self._alpha * angle_raw + (1 - self._alpha) * s.dir_smooth

            # Update prev
            s.prev_cx   = cx
            s.prev_cy   = cy
            s.prev_time = now
            self._frame_count[tid] = self._frame_count.get(tid, 0) + 1

            # Don't trust speed until we have min_frames
            if self._frame_count[tid] < self._min_frames:
                continue

            # ── RUNNING ──────────────────────────────────────────────────────
            if (track.class_name in self._run_classes
                    and s.v_smooth >= self._run_thr):
                if (now - s.run_alert_t) >= self._dedup_w:
                    s.run_alert_t = now
                    events.append(Event(
                        event_type = EventType.RUNNING,
                        severity   = self._run_sev,
                        camera_id  = cam,
                        track      = track,
                        frame_id   = frame_id,
                        zone_id    = None,
                        metadata   = {
                            "speed_px_per_s": round(s.v_smooth, 1),
                            "direction_deg":  round(s.dir_smooth, 1),
                        },
                    ))
                    logger.info(
                        f"[{cam}] RUNNING  track={tid}  "
                        f"speed={s.v_smooth:.1f}px/s"
                    )

            # ── WRONG_WAY ────────────────────────────────────────────────────
            if (self._allowed_dir != "any"
                    and track.class_name in self._ww_classes
                    and s.v_smooth > 5.0):   # must be actually moving
                allowed_angle = _DIR_ANGLE.get(self._allowed_dir, 0.0)
                diff = abs(s.dir_smooth - allowed_angle)
                diff = min(diff, 360.0 - diff)   # wrap around
                if diff > self._dir_tol:
                    if (now - s.ww_alert_t) >= self._dedup_w:
                        s.ww_alert_t = now
                        events.append(Event(
                            event_type = EventType.WRONG_WAY,
                            severity   = self._ww_sev,
                            camera_id  = cam,
                            track      = track,
                            frame_id   = frame_id,
                            zone_id    = None,
                            metadata   = {
                                "speed_px_per_s":  round(s.v_smooth, 1),
                                "direction_deg":   round(s.dir_smooth, 1),
                                "allowed_dir":     self._allowed_dir,
                                "allowed_angle":   allowed_angle,
                                "deviation_deg":   round(diff, 1),
                            },
                        ))
                        logger.info(
                            f"[{cam}] WRONG_WAY  track={tid}  "
                            f"dir={s.dir_smooth:.0f}deg  allowed={self._allowed_dir}"
                        )

        return events

    def get_speed(self, track_id: int) -> float:
        """Return smoothed speed (px/s) for a track, or 0.0 if unknown."""
        return self._state[track_id].v_smooth if track_id in self._state else 0.0

    def reset(self) -> None:
        self._state.clear()
        self._frame_count.clear()
