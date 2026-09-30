"""
services/api/routers/vlm.py
===========================
IBVAP — Vision-Language Model Inspection REST Endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.analytics.vlm.inspector import VLMInspectionResult, VLMInspector
from services.api.models import EventResponse
from services.api.state import state

router = APIRouter(prefix="/vlm", tags=["Vision-Language Models"])
vlm_inspector = VLMInspector()


class VLMInspectRequest(BaseModel):
    event_id: Optional[str] = None
    image_path: Optional[str] = None


class VLMInspectResponse(BaseModel):
    event_id: Optional[str] = None
    description: str
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    accessories: List[str] = Field(default_factory=list)
    provider: str
    latency_ms: float


@router.post("/inspect", response_model=VLMInspectResponse)
async def inspect_target(body: VLMInspectRequest) -> VLMInspectResponse:
    """Analyze a detected person or vehicle using Vision-Language Models."""
    target_img_path = None
    ev = None

    if body.event_id:
        ev = state.event_store.get(body.event_id)
        if not ev:
            raise HTTPException(status_code=404, detail=f"Event {body.event_id} not found")
        target_img_path = ev.snapshot_path
    elif body.image_path:
        target_img_path = body.image_path

    if not target_img_path:
        raise HTTPException(status_code=400, detail="Must provide either event_id with snapshot or image_path")

    # Run inspection
    res: VLMInspectionResult = vlm_inspector.inspect_image(target_img_path)

    # If associated with an event, enrich event metadata and broadcast update
    if ev:
        ev.metadata["vlm_description"] = res.description
        ev.metadata["vlm_upper_color"] = res.upper_color
        ev.metadata["vlm_lower_color"] = res.lower_color
        ev.metadata["vlm_provider"] = res.provider
        # Broadcast updated event over SSE
        await state.sse.broadcast(ev.model_dump())

    return VLMInspectResponse(
        event_id=body.event_id,
        description=res.description,
        upper_color=res.upper_color,
        lower_color=res.lower_color,
        accessories=res.accessories,
        provider=res.provider,
        latency_ms=res.latency_ms,
    )
