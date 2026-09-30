"""
services/event_engine/models.py
================================
IBVAP P2 — Shared Event dataclass.

Every analytics module (fence, ANPR, behaviour, face, night) emits Events.
The EventEngine consumes them, deduplicates, attaches snapshots, and dispatches.

JSON representation is used for logging, webhooks, and the dashboard API.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from services.detect_track.models import Track


# ── Enums ─────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    # Fence / intrusion
    ZONE_ENTER      = "ZONE_ENTER"
    ZONE_EXIT       = "ZONE_EXIT"
    DWELL           = "DWELL"           # loitering inside a polygon
    TRIPWIRE_CROSS  = "TRIPWIRE_CROSS"

    # Behaviour (P5)
    LOITERING       = "LOITERING"
    RUNNING         = "RUNNING"
    GATHERING       = "GATHERING"
    ABANDONED_OBJ   = "ABANDONED_OBJ"
    WRONG_WAY       = "WRONG_WAY"

    # ANPR (P3)
    ANPR_READ       = "ANPR_READ"
    ANPR_WATCHLIST  = "ANPR_WATCHLIST"

    # Night (P4)
    NIGHT_MOVEMENT  = "NIGHT_MOVEMENT"

    # Face (P6)
    FACE_DETECTED   = "FACE_DETECTED"
    FACE_WATCHLIST  = "FACE_WATCHLIST"

    # Threat & Weapons (Harmful objects)
    WEAPON_DETECTED = "WEAPON_DETECTED"
    ARMED_PERSON    = "ARMED_PERSON"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"

    def color_bgr(self) -> tuple[int, int, int]:
        return {
            "CRITICAL": (0,   0, 255),
            "HIGH":     (0, 100, 255),
            "MEDIUM":   (0, 200, 255),
            "LOW":      (200, 200, 0),
        }[self.value]

    def __lt__(self, other: "Severity") -> bool:
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return order.index(self.value) < order.index(other.value)


# ── Event dataclass ───────────────────────────────────────────────────────────

@dataclass
class Event:
    """
    A single alert event produced by any analytics module.

    Fields
    ------
    event_id        UUID, auto-generated.
    event_type      What happened (ZONE_ENTER, TRIPWIRE_CROSS, …).
    severity        CRITICAL / HIGH / MEDIUM / LOW.
    camera_id       Source camera.
    track           The Track that triggered this event.
    zone_id         Zone or tripwire ID (None for non-spatial events).
    timestamp       Wall-clock time of the event (time.time()).
    frame_id        Frame number from FrameReader.
    snapshot_path   Path to saved JPEG crop (None until EventEngine processes it).
    clip_path       Path to saved video clip (None until clip is complete).
    confirmed       None = pending, True = operator confirmed, False = rejected.
    metadata        Extra key-value pairs (dwell_s, direction, plate_text, …).
    """
    event_type:     EventType
    severity:       Severity
    camera_id:      str
    track:          Track
    frame_id:       int

    zone_id:        Optional[str]           = None
    timestamp:      float                   = field(default_factory=time.time)
    event_id:       str                     = field(default_factory=lambda: str(uuid.uuid4()))
    snapshot_path:  Optional[str]           = None
    clip_path:      Optional[str]           = None
    confirmed:      Optional[bool]          = None   # None = awaiting review
    metadata:       Dict[str, Any]          = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id":     self.event_id,
            "event_type":   self.event_type.value,
            "severity":     self.severity.value,
            "camera_id":    self.camera_id,
            "zone_id":      self.zone_id,
            "track_id":     self.track.track_id,
            "class_name":   self.track.class_name,
            "class_id":     self.track.class_id,
            "confidence":   round(self.track.conf, 4),
            "bbox":         [round(v, 1) for v in self.track.bbox],
            "center":       [round(v, 1) for v in self.track.center],
            "frame_id":     self.frame_id,
            "timestamp":    round(self.timestamp, 6),
            "snapshot_path": self.snapshot_path,
            "clip_path":    self.clip_path,
            "confirmed":    self.confirmed,
            "metadata":     self.metadata,
        }
