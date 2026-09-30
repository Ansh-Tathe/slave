"""
services/api/routers/search.py
==============================
IBVAP P9 — Natural Language Search REST Endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from services.api.models import EventResponse
from services.api.state import state
from services.search.engine import NLSearchEngine, ScoredEvent
from services.search.query_parser import ParsedQuery

router = APIRouter(prefix="/search", tags=["Search"])
search_engine = NLSearchEngine()


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class SearchResponse(BaseModel):
    query: str
    parsed: Dict[str, Any]
    total: int
    limit: int
    offset: int
    results: List[Dict[str, Any]]


@router.post("/nl", response_model=SearchResponse)
async def search_natural_language_post(body: SearchRequest) -> SearchResponse:
    """Execute plain English surveillance query (POST)."""
    all_events = list(state.event_store._events_list)
    parsed, results, total = search_engine.search(
        query=body.query,
        events=all_events,
        limit=body.limit,
        offset=body.offset,
    )
    return SearchResponse(
        query=body.query,
        parsed=parsed.to_dict(),
        total=total,
        limit=body.limit,
        offset=body.offset,
        results=[r.to_dict() for r in results],
    )


@router.get("", response_model=SearchResponse)
async def search_natural_language_get(
    q: str = Query(..., description="Natural language search query"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> SearchResponse:
    """Execute plain English surveillance query (GET)."""
    all_events = list(state.event_store._events_list)
    parsed, results, total = search_engine.search(
        query=q,
        events=all_events,
        limit=limit,
        offset=offset,
    )
    return SearchResponse(
        query=q,
        parsed=parsed.to_dict(),
        total=total,
        limit=limit,
        offset=offset,
        results=[r.to_dict() for r in results],
    )
