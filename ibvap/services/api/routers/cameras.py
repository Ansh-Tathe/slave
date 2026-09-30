"""
services/api/routers/cameras.py
===============================
IBVAP P7 — Camera management and runtime telemetry endpoints.
"""

from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.api.auth import require_role
from services.api.models import CameraConfigUpdate, CameraHeartbeat, CameraStatus
from services.api.state import state

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.get("", response_model=List[CameraStatus])
async def list_cameras() -> List[CameraStatus]:
    """List all registered surveillance cameras and their operational statuses."""
    return state.camera_registry.get_all()


@router.get("/{camera_id}", response_model=CameraStatus)
async def get_camera(camera_id: str) -> CameraStatus:
    """Get status and telemetry for a specific camera."""
    cam = state.camera_registry.get(camera_id)
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )
    return cam


@router.post("/{camera_id}/heartbeat", response_model=CameraStatus)
async def update_heartbeat(camera_id: str, heartbeat: CameraHeartbeat) -> CameraStatus:
    """Ingest pipeline heartbeat: updates FPS, frame count, and last seen timestamp."""
    return state.camera_registry.record_heartbeat(
        cam_id=camera_id,
        fps=heartbeat.fps,
        frame_id=heartbeat.frame_id,
        status=heartbeat.status,
    )


@router.patch("/{camera_id}", response_model=CameraStatus)
async def update_camera(
    camera_id: str,
    update: CameraConfigUpdate,
    _user: dict = Depends(require_role("operator")),
) -> CameraStatus:
    """Update camera configuration (enable/disable, name, location). Requires operator role."""
    cam = state.camera_registry.get(camera_id)
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )
    if update.enabled is not None:
        cam.enabled = update.enabled
        cam.status = "ONLINE" if update.enabled else "OFFLINE"
    if update.name is not None:
        cam.name = update.name
    if update.location is not None:
        cam.location = update.location

    return cam


# ── Live Video Streaming & Ingest ─────────────────────────────────────────────

class ZoneUpdateRequest(BaseModel):
    polygon: List[List[float]]


@router.post("/{camera_id}/frame")
async def upload_frame(camera_id: str, request: Request) -> dict:
    """Ingest a processed surveillance frame from run_webcam or camera reader."""
    body = await request.body()
    if body:
        state.frame_buffer.set_frame(camera_id, body)
    return {"status": "ok", "bytes": len(body)}


@router.get("/{camera_id}/frame")
async def get_latest_frame(camera_id: str):
    """Retrieve the latest captured frame as JPEG."""
    frame = state.frame_buffer.get_frame(camera_id)
    if not frame:
        raise HTTPException(status_code=404, detail="No frame available for this camera yet")
    return Response(content=frame, media_type="image/jpeg")


@router.get("/{camera_id}/stream")
async def mjpeg_stream(camera_id: str):
    """Real-time multipart/x-mixed-replace MJPEG video stream."""
    async def generator():
        while True:
            frame = state.frame_buffer.get_frame(camera_id)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(0.05)  # up to 20 FPS

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/{camera_id}/zone")
async def update_camera_zone(camera_id: str, body: ZoneUpdateRequest):
    """Dynamically update restriction zone coordinates from the frontend."""
    from pathlib import Path
    import yaml
    p = Path("configs/zones.yaml")
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            key = f"{camera_id}_zones"
            if "zones" not in cfg:
                cfg["zones"] = {}
            if key not in cfg["zones"]:
                cfg["zones"][key] = {"polygons": [], "tripwires": []}
            polys = cfg["zones"][key].get("polygons", [])
            if polys:
                polys[0]["points"] = body.polygon
            else:
                polys.append({
                    "id": "restricted_custom",
                    "name": "Custom Restricted Zone",
                    "enabled": True,
                    "points": body.polygon,
                    "triggers": {"on_enter": True, "dwell_seconds": 3},
                    "severity": "HIGH",
                })
            cfg["zones"][key]["polygons"] = polys
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        except Exception:
            pass
    return {"status": "updated", "camera_id": camera_id, "polygon": body.polygon}

