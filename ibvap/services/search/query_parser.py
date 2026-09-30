"""
services/search/query_parser.py
===============================
IBVAP P9 — Natural Language Query Parser for Surveillance Search.
Extracts structured intent, temporal ranges, event types, severities,
object classes, and cameras from plain-English queries.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class ParsedQuery:
    raw_query: str
    event_types: List[str] = field(default_factory=list)
    severities: List[str] = field(default_factory=list)
    camera_ids: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    plate: Optional[str] = None
    confirmed: Optional[bool] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "event_types": self.event_types,
            "severities": self.severities,
            "camera_ids": self.camera_ids,
            "classes": self.classes,
            "plate": self.plate,
            "confirmed": self.confirmed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "keywords": self.keywords,
        }


class NLQueryParser:
    """Parses plain English surveillance queries into structured search criteria."""

    # Event type synonym mapping
    EVENT_SYNONYMS = {
        "TRIPWIRE_CROSS": ["tripwire", "crossed", "cross", "boundary", "line", "intrusion", "fence line"],
        "ZONE_ENTER": ["zone enter", "entered zone", "restricted area", "perimeter breach", "enter"],
        "LOITERING": ["loiter", "loitering", "loiterer", "dwelling", "standing", "waiting around"],
        "RUNNING": ["running", "runner", "speeding", "sprint", "fleeing", "fast"],
        "GATHERING": ["gathering", "crowd", "group", "mob", "assembly", "congregation"],
        "ABANDONED_OBJ": ["abandoned", "unattended", "left behind", "luggage", "backpack", "package", "bag"],
        "NIGHT_MOVEMENT": ["night", "nighttime", "after hours", "nocturnal", "dark"],
        "ANPR_WATCHLIST": ["stolen car", "wanted vehicle", "plate watchlist", "blacklisted car"],
        "ANPR_READ": ["plate", "license plate", "number plate", "vehicle read"],
        "FACE_WATCHLIST": ["suspect", "wanted person", "face watchlist", "target", "blacklist person"],
        "FACE_DETECTED": ["face", "facial", "person recognized"],
    }

    SEVERITY_SYNONYMS = {
        "CRITICAL": ["critical", "emergency", "urgent", "danger", "severe"],
        "HIGH": ["high", "major", "serious"],
        "MEDIUM": ["medium", "moderate"],
        "LOW": ["low", "minor", "info", "routine"],
    }

    CLASS_SYNONYMS = {
        "person": ["person", "people", "man", "woman", "pedestrian", "someone", "individual", "human", "intruder"],
        "car": ["car", "cars", "automobile", "sedan", "suv", "vehicle", "vehicles"],
        "truck": ["truck", "trucks", "lorry", "container", "heavy vehicle"],
        "motorcycle": ["motorcycle", "bike", "motorbike", "scooter"],
        "bus": ["bus", "buses", "coach"],
    }

    LOCATION_MAP = {
        "cam_01": ["cam_01", "main gate", "north gate", "gate", "perimeter north"],
        "cam_test_file": ["cam_test_file", "lab", "test camera", "test cam"],
    }

    def parse(self, query: str, ref_time: Optional[float] = None) -> ParsedQuery:
        now = ref_time if ref_time is not None else time.time()
        q_lower = query.lower().strip()
        parsed = ParsedQuery(raw_query=query)

        # 1. Extract Severities
        for sev, syns in self.SEVERITY_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(s)}\b", q_lower) for s in syns):
                parsed.severities.append(sev)

        # 2. Extract Event Types
        for etype, syns in self.EVENT_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(s)}\b", q_lower) for s in syns):
                if etype not in parsed.event_types:
                    parsed.event_types.append(etype)

        # 3. Extract Classes
        for cls_name, syns in self.CLASS_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(s)}\b", q_lower) for s in syns):
                if cls_name not in parsed.classes:
                    parsed.classes.append(cls_name)

        # 4. Extract Cameras / Locations
        for cam_id, names in self.LOCATION_MAP.items():
            if any(re.search(rf"\b{re.escape(n)}\b", q_lower) for n in names):
                if cam_id not in parsed.camera_ids:
                    parsed.camera_ids.append(cam_id)

        # Direct camera id match (e.g. cam_02, camera 1)
        cam_match = re.search(r"\b(?:cam|camera)[_\s]*0?(\d+)\b", q_lower)
        if cam_match:
            cam_str = f"cam_{int(cam_match.group(1)):02d}"
            if cam_str not in parsed.camera_ids:
                parsed.camera_ids.append(cam_str)

        # 5. Extract License Plate Pattern (e.g. KA01AB1234, DL-04-C-1234, MH12DE5678)
        plate_match = re.search(r"\b([A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[A-Z]{1,3}[-\s]?[0-9]{4})\b", query.upper())
        if plate_match:
            parsed.plate = re.sub(r"[-\s]", "", plate_match.group(1))

        # 6. Extract Confirmation Status
        if re.search(r"\b(confirmed|verified|approved)\b", q_lower):
            parsed.confirmed = True
        elif re.search(r"\b(rejected|false alarm|dismissed)\b", q_lower):
            parsed.confirmed = False
        elif re.search(r"\b(pending|unconfirmed|unreviewed|needs review)\b", q_lower):
            parsed.confirmed = None  # Explicit unreviewed search indicator

        # 7. Extract Temporal Expressions
        self._parse_time_range(q_lower, now, parsed)

        # 8. Remaining Keywords
        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", q_lower)
        stop_words = {
            "show", "find", "list", "get", "all", "the", "and", "for", "with", "near", "from",
            "detected", "alert", "alerts", "events", "camera", "last", "past", "yesterday", "today"
        }
        parsed.keywords = [w for w in words if w not in stop_words]

        return parsed

    def _parse_time_range(self, q: str, now: float, parsed: ParsedQuery) -> None:
        # "last N minutes"
        m_min = re.search(r"\blast\s+(\d+)\s+min(?:ute)?s?\b", q)
        if m_min:
            parsed.start_time = now - (int(m_min.group(1)) * 60)
            parsed.end_time = now
            return

        # "last N hours"
        m_hr = re.search(r"\blast\s+(\d+)\s+hour?s?\b", q)
        if m_hr:
            parsed.start_time = now - (int(m_hr.group(1)) * 3600)
            parsed.end_time = now
            return

        # "last N days"
        m_day = re.search(r"\blast\s+(\d+)\s+days?\b", q)
        if m_day:
            parsed.start_time = now - (int(m_day.group(1)) * 86400)
            parsed.end_time = now
            return

        # "today"
        if "today" in q:
            dt_today = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
            parsed.start_time = dt_today.timestamp()
            parsed.end_time = now
            return

        # "yesterday"
        if "yesterday" in q:
            dt_today = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
            dt_yesterday = dt_today - timedelta(days=1)
            parsed.start_time = dt_yesterday.timestamp()
            parsed.end_time = dt_today.timestamp()
            return
