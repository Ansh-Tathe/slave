"""
services/analytics/behaviour/behaviour_engine.py
=================================================
IBVAP P5 -- Behaviour analytics orchestrator.

Combines all P5 rule modules into a single update() call:
  - LoiteringDetector   -> LOITERING
  - SpeedDetector       -> RUNNING, WRONG_WAY
  - GatheringDetector   -> GATHERING
  - AbandonedObjectDetector -> ABANDONED_OBJ

Also annotates each Track with velocity information (vx, vy, speed)
so downstream renderers can display speed overlays.

Usage
-----
    engine = BehaviourEngine.from_config("configs/rules.yaml", camera_id="cam_01")
    events = engine.update(tracks, frame_id=42)
    # tracks now have .velocity set (px/s)

    # Or with defaults:
    engine = BehaviourEngine(camera_id="cam_01")
    events = engine.update(tracks, frame_id=42)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger

from services.analytics.behaviour.abandoned  import AbandonedObjectDetector
from services.analytics.behaviour.gathering  import GatheringDetector
from services.analytics.behaviour.loitering  import LoiteringDetector
from services.analytics.behaviour.speed      import SpeedDetector
from services.detect_track.models            import Track
from services.event_engine.models            import Event, Severity


class BehaviourEngine:
    """
    Single entry-point for all P5 behaviour analytics.

    Parameters
    ----------
    camera_id               : camera identifier for all sub-detectors
    loiter_dwell_s          : loitering dwell threshold (seconds)
    loiter_move_px          : movement threshold to reset loiter clock
    run_threshold_px_per_s  : speed above this -> RUNNING
    allowed_direction       : "any" | "left" | "right" | "up" | "down"
    gather_min_count        : min persons in cluster for GATHERING
    gather_proximity_px     : cluster proximity radius
    gather_duration_s       : cluster duration before GATHERING fires
    abandoned_stationary_s  : object stationary time for ABANDONED_OBJ
    abandoned_owner_radius_px: person within this radius suppresses alert
    enable_loitering        : toggle individual rules on/off
    enable_running          : ...
    enable_gathering        : ...
    enable_abandoned        : ...
    """

    def __init__(
        self,
        camera_id:               str = "unknown",
        loiter_dwell_s:          float = 60.0,
        loiter_move_px:          float = 30.0,
        run_threshold_px_per_s:  float = 150.0,
        allowed_direction:       str = "any",
        gather_min_count:        int = 5,
        gather_proximity_px:     float = 200.0,
        gather_duration_s:       float = 10.0,
        abandoned_stationary_s:  float = 120.0,
        abandoned_owner_radius_px: float = 150.0,
        enable_loitering:        bool = True,
        enable_running:          bool = True,
        enable_gathering:        bool = True,
        enable_abandoned:        bool = True,
    ) -> None:
        self._camera = camera_id

        self._loitering: Optional[LoiteringDetector] = (
            LoiteringDetector(
                dwell_seconds         = loiter_dwell_s,
                movement_threshold_px = loiter_move_px,
                camera_id             = camera_id,
            ) if enable_loitering else None
        )

        self._speed: Optional[SpeedDetector] = (
            SpeedDetector(
                run_threshold_px_per_s = run_threshold_px_per_s,
                allowed_direction      = allowed_direction,
                camera_id              = camera_id,
            ) if enable_running else None
        )

        self._gathering: Optional[GatheringDetector] = (
            GatheringDetector(
                min_count     = gather_min_count,
                proximity_px  = gather_proximity_px,
                duration_s    = gather_duration_s,
                camera_id     = camera_id,
            ) if enable_gathering else None
        )

        self._abandoned: Optional[AbandonedObjectDetector] = (
            AbandonedObjectDetector(
                stationary_seconds     = abandoned_stationary_s,
                owner_radius_px        = abandoned_owner_radius_px,
                camera_id              = camera_id,
            ) if enable_abandoned else None
        )

        enabled = [
            name for name, flag in [
                ("loitering", enable_loitering),
                ("running/wrong_way", enable_running),
                ("gathering", enable_gathering),
                ("abandoned_obj", enable_abandoned),
            ] if flag
        ]
        logger.info(f"BehaviourEngine ready — camera={camera_id}, rules={enabled}")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        tracks:    List[Track],
        frame_id:  int,
        camera_id: Optional[str] = None,
    ) -> List[Event]:
        """
        Run all enabled behaviour rules on the current track list.

        Side-effect: sets track.velocity = (vx, vy) for each track
        where speed data is available (used by Renderer for speed overlay).

        Returns
        -------
        Flat list of Events from all rules (may be empty).
        """
        cam    = camera_id or self._camera
        events: List[Event] = []

        # Speed first -- so we can annotate tracks with velocity
        if self._speed:
            events.extend(self._speed.update(tracks, frame_id, cam))
            # Annotate track.velocity with current speed vector
            for track in tracks:
                s = self._speed._state.get(track.track_id)
                if s is not None:
                    spd = s.v_smooth
                    import math
                    ang = math.radians(s.dir_smooth)
                    track.velocity = (spd * math.cos(ang), -spd * math.sin(ang))

        if self._loitering:
            events.extend(self._loitering.update(tracks, frame_id, cam))

        if self._gathering:
            events.extend(self._gathering.update(tracks, frame_id, cam))

        if self._abandoned:
            events.extend(self._abandoned.update(tracks, frame_id, cam))

        return events

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        rules_yaml: str | Path,
        camera_id:  str = "unknown",
    ) -> "BehaviourEngine":
        """
        Build a BehaviourEngine from rules.yaml.
        Falls back to defaults if the file is missing or keys are absent.
        """
        import yaml
        path = Path(rules_yaml)
        if not path.exists():
            logger.warning(f"rules.yaml not found: {path} — using defaults")
            return cls(camera_id=camera_id)

        with path.open() as f:
            raw = yaml.safe_load(f)

        models   = raw.get("models", {})
        rules    = {r["id"]: r for r in raw.get("rules", [])}

        # Extract per-rule params from rules.yaml
        loiter_r = rules.get("rule_loitering_person", {})
        run_r    = rules.get("rule_running", {})
        gather_r = rules.get("rule_group_gathering", {})
        aband_r  = rules.get("rule_abandoned_object", {})

        return cls(
            camera_id              = camera_id,
            loiter_dwell_s         = loiter_r.get("dwell_seconds", 60.0),
            run_threshold_px_per_s = run_r.get("speed_threshold_px_per_s", 150.0),
            gather_min_count       = gather_r.get("min_count", 5),
            gather_proximity_px    = gather_r.get("proximity_px", 200.0),
            gather_duration_s      = gather_r.get("duration_s", 10.0),
            abandoned_stationary_s = aband_r.get("stationary_seconds", 120.0),
            enable_loitering       = loiter_r.get("enabled", True),
            enable_running         = run_r.get("enabled", True),
            enable_gathering       = gather_r.get("enabled", True),
            enable_abandoned       = aband_r.get("enabled", True),
        )

    def reset(self) -> None:
        """Reset all sub-detector state (call on stream restart)."""
        for det in [self._loitering, self._speed, self._gathering, self._abandoned]:
            if det is not None:
                det.reset()
