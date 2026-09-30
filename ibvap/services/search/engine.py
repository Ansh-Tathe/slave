"""
services/search/engine.py
=========================
IBVAP P9 — Natural Language Surveillance Search Engine.
Matches parsed query filters, scores relevance, and returns ranked events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from services.api.models import EventResponse
from services.search.query_parser import NLQueryParser, ParsedQuery


@dataclass
class ScoredEvent:
    event: EventResponse
    score: float
    match_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "match_reasons": self.match_reasons,
            "event": self.event.model_dump(),
        }


class NLSearchEngine:
    """Evaluates plain English search queries across surveillance events."""

    def __init__(self, parser: Optional[NLQueryParser] = None) -> None:
        self.parser = parser or NLQueryParser()

    def search(
        self,
        query: str,
        events: List[EventResponse],
        limit: int = 50,
        offset: int = 0,
        min_score: float = 1.0,
    ) -> tuple[ParsedQuery, List[ScoredEvent], int]:
        """
        Execute natural language search across a list of events.

        Returns
        -------
        parsed_query : ParsedQuery
        results      : List[ScoredEvent] paginated
        total        : Total number of matching events
        """
        parsed = self.parser.parse(query)
        scored: List[ScoredEvent] = []

        for ev in events:
            score = 0.0
            reasons = []

            # 1. Event Type Match
            if parsed.event_types:
                if ev.event_type in parsed.event_types:
                    score += 15.0
                    reasons.append(f"Event type '{ev.event_type}' matched")
                else:
                    # If query specifically wanted an event type and didn't match, reduce score
                    score -= 5.0

            # 2. Severity Match
            if parsed.severities:
                if ev.severity.upper() in [s.upper() for s in parsed.severities]:
                    score += 10.0
                    reasons.append(f"Severity '{ev.severity}' matched")
                else:
                    score -= 3.0

            # 3. Camera / Location Match
            if parsed.camera_ids:
                if ev.camera_id in parsed.camera_ids:
                    score += 12.0
                    reasons.append(f"Camera '{ev.camera_id}' matched")
                else:
                    score -= 4.0

            # 4. Class Name Match
            if parsed.classes:
                if ev.class_name.lower() in [c.lower() for c in parsed.classes]:
                    score += 10.0
                    reasons.append(f"Class '{ev.class_name}' matched")
                else:
                    score -= 3.0

            # 5. Time Range Match
            if parsed.start_time is not None and parsed.end_time is not None:
                if parsed.start_time <= ev.timestamp <= parsed.end_time:
                    score += 15.0
                    reasons.append("Within specified time window")
                else:
                    continue  # Hard filter out-of-time events

            # 6. License Plate Match
            if parsed.plate:
                meta_plate = str(ev.metadata.get("plate", "")).upper().replace("-", "").replace(" ", "")
                if parsed.plate in meta_plate:
                    score += 30.0
                    reasons.append(f"License plate '{parsed.plate}' matched")
                else:
                    score -= 10.0

            # 7. Confirmation Status Match
            if parsed.confirmed is not None:
                if ev.confirmed == parsed.confirmed:
                    score += 8.0
                    reasons.append(f"Confirmation status matched ({parsed.confirmed})")

            # 8. Keyword / Free-text Matches in Metadata
            if parsed.keywords:
                ev_str = f"{ev.event_type} {ev.camera_id} {ev.zone_id or ''} {ev.class_name} {json.dumps(ev.metadata)}".lower()
                for kw in parsed.keywords:
                    if kw in ev_str:
                        score += 5.0
                        reasons.append(f"Keyword '{kw}' found in metadata")

            if score >= min_score:
                scored.append(ScoredEvent(event=ev, score=score, match_reasons=reasons))

        # Sort by score descending, then by timestamp descending
        scored.sort(key=lambda x: (x.score, x.event.timestamp), reverse=True)

        total = len(scored)
        paginated = scored[offset : offset + limit]
        return parsed, paginated, total
