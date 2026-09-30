"""
services/api/routers/cameras.py
===============================
IBVAP P7 — Camera management and runtime telemetry endpoints.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

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
