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


# ── Direct Browser AI Detection Endpoint ──────────────────────────────────────

_global_yolo = None
_global_threat = None
_global_mask = None


def get_detectors():
    global _global_yolo, _global_threat, _global_mask
    from ultralytics import YOLO
    from pathlib import Path
    if _global_yolo is None:
        try:
            _global_yolo = YOLO("yolov8n.pt")
        except Exception:
            pass
    if _global_threat is None:
        p = Path("models/threat_yolov8n.pt")
        if p.exists():
            try:
                _global_threat = YOLO(str(p))
            except Exception:
                pass
    if _global_mask is None:
        p = Path("models/mask_yolov8n.pt")
        if p.exists():
            try:
                _global_mask = YOLO(str(p))
            except Exception:
                pass
    return _global_yolo, _global_threat, _global_mask


def is_point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray-casting algorithm to test if point (x, y) is inside polygon."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


@router.post("/{camera_id}/detect_frame")
async def detect_frame(camera_id: str, request: Request) -> dict:
    """Run real-time person, weapon, mask threat detection and zone analysis on uploaded browser frame."""
    import cv2
    import numpy as np
    from pathlib import Path
    import yaml
    import time
    from services.api.models import EventResponse

    body = await request.body()
    if not body:
        return {"tracks": [], "weapons": [], "alerts": []}

    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"tracks": [], "weapons": [], "alerts": []}

    h, w = frame.shape[:2]
    yolo, threat, mask_model = get_detectors()

    # Save latest frame in streaming buffer
    state.frame_buffer.set_frame(camera_id, body)

    # Load active zone polygon
    zone_poly = [[0.05, 0.15], [0.48, 0.15], [0.48, 0.85], [0.05, 0.85]]
    try:
        p_cfg = Path("configs/zones.yaml")
        if p_cfg.exists():
            with open(p_cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            key = f"{camera_id}_zones"
            polys = cfg.get("zones", {}).get(key, {}).get("polygons", [])
            if polys and "points" in polys[0]:
                zone_poly = polys[0]["points"]
    except Exception:
        pass

    tracks = []
    weapons = []
    alerts = []

    # 1. Person & Vehicle Detection
    if yolo is not None:
        r_y = yolo(frame, conf=0.25, verbose=False)[0]
        for idx, b in enumerate(r_y.boxes):
            c_name = r_y.names[int(b.cls[0])]
            conf = float(b.conf[0].item())
            xyxy = [float(v) for v in b.xyxy[0].cpu().numpy()]
            x1, y1, x2, y2 = xyxy

            # Zone test: bottom center (feet) or center of person
            foot_nx = ((x1 + x2) / 2.0) / w
            foot_ny = y2 / h
            center_ny = ((y1 + y2) / 2.0) / h
            in_zone = is_point_in_polygon(foot_nx, foot_ny, zone_poly) or is_point_in_polygon(foot_nx, center_ny, zone_poly)

            # Masked face check on upper torso / head crop
            is_masked = False
            if c_name == "person" and mask_model is not None:
                head_crop = frame[max(0, int(y1)):min(h, int(y1 + (y2 - y1) * 0.45)), max(0, int(x1)):min(w, int(x2))]
                if head_crop.size > 0:
                    try:
                        r_m = mask_model(head_crop, conf=0.15, verbose=False)[0]
                        for mb in r_m.boxes:
                            m_cls = r_m.names[int(mb.cls[0])]
                            if "bermasker" in m_cls.lower() and "tidak" not in m_cls.lower():
                                is_masked = True
                                alerts.append({
                                    "type": "MASKED_PERSON",
                                    "severity": "HIGH",
                                    "message": f"Masked / Concealed Individual (Target #{idx+1})",
                                })
                                break
                    except Exception:
                        pass

            if in_zone and c_name == "person":
                alerts.append({
                    "type": "ZONE_INTRUSION",
                    "severity": "HIGH",
                    "message": f"Restricted zone intrusion detected (Target #{idx+1})",
                })

            tracks.append({
                "track_id": idx + 1,
                "class_name": c_name,
                "conf": round(conf, 3),
                "bbox": [round(c, 1) for c in xyxy],
                "is_masked": is_masked,
                "in_zone": in_zone,
                "is_armed": False,
                "weapon": None,
            })

    # 2. Weapon & Threat Detection
    if threat is not None:
        r_t = threat(frame, conf=0.18, verbose=False)[0]
        for b in r_t.boxes:
            w_name = r_t.names[int(b.cls[0])]
            conf = float(b.conf[0].item())
            xyxy = [float(v) for v in b.xyxy[0].cpu().numpy()]
            wx1, wy1, wx2, wy2 = xyxy
            wcx, wcy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0

            weapons.append({
                "class_name": w_name,
                "conf": round(conf, 3),
                "bbox": [round(c, 1) for c in xyxy],
            })

            # Associate weapon with person
            armed_target_id = None
            for t in tracks:
                if t["class_name"] == "person":
                    px1, py1, px2, py2 = t["bbox"]
                    # Expanded bounding box check (hands/belt area)
                    margin_x = (px2 - px1) * 0.35
                    margin_y = (py2 - py1) * 0.25
                    if (px1 - margin_x <= wcx <= px2 + margin_x) and (py1 - margin_y <= wcy <= py2 + margin_y):
                        t["is_armed"] = True
                        t["weapon"] = w_name
                        armed_target_id = t["track_id"]
                        break

            if armed_target_id:
                alerts.append({
                    "type": "ARMED_PERSON",
                    "severity": "CRITICAL",
                    "message": f"CRITICAL: Target #{armed_target_id} armed with {w_name.upper()} ({conf*100:.0f}%)",
                })
            else:
                alerts.append({
                    "type": "WEAPON_DETECTED",
                    "severity": "HIGH",
                    "message": f"Dangerous threat spotted: {w_name.upper()} ({conf*100:.0f}%)",
                })

    # Push top alert to SSE broadcaster and event store
    if alerts:
        top_alert = max(alerts, key=lambda a: 2 if a["severity"] == "CRITICAL" else 1)
        ref_track = tracks[0] if tracks else None
        t_bbox = ref_track["bbox"] if ref_track else [0.0, 0.0, float(w), float(h)]
        t_center = [(t_bbox[0] + t_bbox[2]) / 2.0, (t_bbox[1] + t_bbox[3]) / 2.0]
        ev_resp = EventResponse(
            event_id=f"ev_web_{int(time.time()*1000)}",
            camera_id=camera_id,
            timestamp=time.time(),
            event_type=top_alert["type"],
            severity=top_alert["severity"],
            zone_id="restricted_zone",
            track_id=ref_track["track_id"] if ref_track else 1,
            frame_id=0,
            class_name=ref_track["class_name"] if ref_track else "threat",
            class_id=0,
            confidence=ref_track["conf"] if ref_track else 0.85,
            bbox=t_bbox,
            center=t_center,
            metadata={"source": "browser_webcam", "summary": top_alert["message"]},
        )
        state.event_store.add(ev_resp)
        asyncio.create_task(state.sse.broadcast(ev_resp.dict()))

    return {
        "camera_id": camera_id,
        "width": w,
        "height": h,
        "tracks": tracks,
        "weapons": weapons,
        "alerts": alerts,
        "zone": zone_poly,
    }


