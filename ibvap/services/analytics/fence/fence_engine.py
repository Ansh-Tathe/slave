"""
services/analytics/fence/fence_engine.py
==========================================
IBVAP P2 — Fence engine: loads zones.yaml and orchestrates
ZoneChecker + TripwireChecker for one camera.

Usage:
    engine = FenceEngine.from_config("configs/zones.yaml", "cam_01",
                                     frame_w=1280, frame_h=720)
    events = engine.update(tracks, frame_id=42)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from loguru import logger

from services.analytics.fence.tripwire    import TripwireChecker, TripwireConfig
from services.analytics.fence.zone_checker import ZoneChecker, ZoneConfig
from services.detect_track.models          import Track
from services.event_engine.models          import Event, Severity


# ── Helper: parse severity string → enum ─────────────────────────────────────

def _sev(s: str) -> Severity:
    try:
        return Severity[s.upper()]
    except KeyError:
        return Severity.MEDIUM


# ── Main class ────────────────────────────────────────────────────────────────

class FenceEngine:
    """
    Combines ZoneChecker and TripwireChecker for a single camera.

    Parameters
    ----------
    zone_cfgs   : polygon zone definitions
    wire_cfgs   : tripwire line definitions
    frame_w/h   : frame resolution (for normalised → pixel mapping)
    camera_id   : camera identifier
    """

    def __init__(
        self,
        zone_cfgs:  List[ZoneConfig],
        wire_cfgs:  List[TripwireConfig],
        frame_w:    int,
        frame_h:    int,
        camera_id:  str,
    ) -> None:
        self._zone_checker = ZoneChecker(
            zone_cfgs, frame_w, frame_h, camera_id
        )
        self._wire_checker = TripwireChecker(
            wire_cfgs, frame_w, frame_h, camera_id
        )
        self.zone_cfgs  = zone_cfgs
        self.wire_cfgs  = wire_cfgs
        self.camera_id  = camera_id
        self.frame_w    = frame_w
        self.frame_h    = frame_h

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        tracks:   List[Track],
        frame_id: int,
    ) -> List[Event]:
        """
        Run zone + tripwire checks for the current frame's tracks.

        Returns combined list of Events (may be empty).
        """
        events: List[Event] = []
        events.extend(self._zone_checker.update(tracks, frame_id))
        events.extend(self._wire_checker.update(tracks, frame_id))
        return events

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        zones_yaml:    str | Path,
        camera_id:     str,
        frame_w:       int = 1280,
        frame_h:       int = 720,
        zones_key:     Optional[str] = None,  # override auto-derive
    ) -> "FenceEngine":
        """
        Load zones.yaml and build a FenceEngine for *camera_id*.

        The zones_key defaults to  f"{camera_id}_zones"  which matches
        the convention in configs/zones.yaml.
        """
        path = Path(zones_yaml)
        if not path.exists():
            logger.warning(f"zones.yaml not found: {path} — fence disabled")
            return cls([], [], frame_w, frame_h, camera_id)

        with path.open() as f:
            raw = yaml.safe_load(f)

        key = zones_key or f"{camera_id}_zones"
        cam_zones = raw.get("zones", {}).get(key, {})

        if not cam_zones:
            logger.warning(
                f"No zones defined for key '{key}' in {path} — fence disabled"
            )
            return cls([], [], frame_w, frame_h, camera_id)

        # ── Parse polygons ────────────────────────────────────────────────
        zone_cfgs: List[ZoneConfig] = []
        for p in cam_zones.get("polygons", []):
            zone_cfgs.append(ZoneConfig(
                zone_id       = p["id"],
                name          = p.get("name", p["id"]),
                polygon       = [tuple(pt) for pt in p["points"]],
                on_enter      = p.get("triggers", {}).get("on_enter", True),
                on_exit       = p.get("triggers", {}).get("on_exit", False),
                dwell_seconds = p.get("triggers", {}).get("dwell_seconds", 0.0),
                classes       = p.get("triggers", {}).get("classes", ["person"]),
                severity      = _sev(p.get("severity", "HIGH")),
                enabled       = p.get("enabled", True),
            ))

        # ── Parse tripwires ───────────────────────────────────────────────
        wire_cfgs: List[TripwireConfig] = []
        for w in cam_zones.get("tripwires", []):
            wire_cfgs.append(TripwireConfig(
                wire_id   = w["id"],
                name      = w.get("name", w["id"]),
                start     = tuple(w["start"]),
                end       = tuple(w["end"]),
                direction = w.get("direction", "both"),
                classes   = w.get("classes", ["person"]),
                severity  = _sev(w.get("severity", "CRITICAL")),
                enabled   = w.get("enabled", True),
            ))

        logger.info(
            f"[{camera_id}] FenceEngine loaded: "
            f"{len(zone_cfgs)} polygon(s), {len(wire_cfgs)} tripwire(s)"
        )
        return cls(zone_cfgs, wire_cfgs, frame_w, frame_h, camera_id)
