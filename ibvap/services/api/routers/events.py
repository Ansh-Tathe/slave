"""
services/api/routers/events.py
==============================
IBVAP P7 — Surveillance events endpoints: query, pagination, confirmation,
snapshot serving, and real-time SSE stream.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

from services.api.auth import get_current_user, require_role
from services.api.models import (
    EventConfirmRequest,
    EventCreate,
    EventListResponse,
    EventResponse,
)
from services.api.state import state
from services.api.webhooks.dispatcher import dispatcher

router = APIRouter(prefix="/events", tags=["Events"])


@router.get("", response_model=EventListResponse)
async def list_events(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type (e.g. TRIPWIRE_CROSS)"),
    severity: Optional[str] = Query(None, description="Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)"),
    confirmed: Optional[bool] = Query(None, description="Filter by confirmed status (true, false, null)"),
    since_s: Optional[float] = Query(None, description="Filter events within last N seconds"),
    limit: int = Query(50, ge=1, le=500, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> EventListResponse:
    """Query surveillance events with multi-field filtering and pagination."""
    items, total = state.event_store.query(
        camera_id=camera_id,
        event_type=event_type,
        severity=severity,
        confirmed=confirmed,
        since_s=since_s,
        limit=limit,
        offset=offset,
    )
    return EventListResponse(total=total, limit=limit, offset=offset, events=items)


@router.get("/stream")
async def stream_events():
    """
    Server-Sent Events (SSE) endpoint for real-time live alert stream.
    Connect with EventSource in browser.
    """
    async def event_generator():
        q = await state.sse.subscribe()
        try:
            # Send initial keepalive / ping
            yield ": ibvap stream connected\n\n"
            while True:
                try:
                    # Timeout after 15s to send keepalive comment
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            state.sse.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: str) -> EventResponse:
    """Retrieve details for a specific event by UUID."""
    ev = state.event_store.get(event_id)
    if not ev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    return ev


@router.post("", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def create_event(payload: EventCreate) -> EventResponse:
    """
    Ingest a new alert event (called by detection pipeline or external camera edge).
    Persists to event store, broadcasts to SSE dashboard, and triggers C2 webhooks.
    """
    event_id = str(uuid.uuid4())
    event_resp = EventResponse(
        event_id=event_id,
        event_type=payload.event_type,
        severity=payload.severity.upper(),
        camera_id=payload.camera_id,
        zone_id=payload.zone_id,
        track_id=payload.track_id,
        class_name=payload.class_name,
        class_id=payload.class_id,
        confidence=payload.confidence,
        bbox=payload.bbox,
        center=payload.center,
        frame_id=payload.frame_id,
        timestamp=payload.timestamp,
        snapshot_path=payload.snapshot_path,
        clip_path=payload.clip_path,
        confirmed=payload.confirmed,
        metadata=payload.metadata,
    )

    # 1. Add to store
    state.event_store.add(event_resp)

    # 2. Update camera metrics
    state.camera_registry.record_event(payload.camera_id)

    # 3. Broadcast to real-time SSE clients
    await state.sse.broadcast(event_resp.model_dump())

    # 4. Asynchronously forward to C2 webhooks
    await dispatcher.dispatch_event(event_resp)

    return event_resp


@router.patch("/{event_id}/confirm", response_model=EventResponse)
async def confirm_event(
    event_id: str,
    body: EventConfirmRequest,
    current_user: dict = Depends(require_role("operator")),
) -> EventResponse:
    """
    Operator action: confirm or reject an alert.
    Requires at least 'operator' role.
    """
    notes = body.notes or f"Reviewed by {current_user['username']}"
    updated = state.event_store.update_confirmation(event_id, body.confirmed, notes=notes)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )

    # Broadcast updated event to SSE so operators see state sync immediately
    await state.sse.broadcast(updated.model_dump())
    return updated


@router.get("/{event_id}/snapshot")
async def get_snapshot(event_id: str):
    """Serve the JPEG snapshot crop associated with an event."""
    ev = state.event_store.get(event_id)
    if not ev or not ev.snapshot_path:
        raise HTTPException(status_code=404, detail="Snapshot not available for this event")

    snap_file = Path(ev.snapshot_path)
    if not snap_file.exists():
        raise HTTPException(status_code=404, detail="Snapshot file missing on disk")

    return FileResponse(snap_file, media_type="image/jpeg")
