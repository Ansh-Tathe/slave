"""
services/api/models.py
======================
IBVAP P7 — Pydantic models for REST API requests and responses.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Auth Models ───────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    role: str = "operator"
    username: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    username: str
    role: str
    is_active: bool = True


# ── Event Models ──────────────────────────────────────────────────────────────

class EventCreate(BaseModel):
    event_type: str
    severity: str = "MEDIUM"
    camera_id: str
    track_id: int
    class_name: str = "person"
    class_id: int = 0
    confidence: float = 0.85
    bbox: List[float] = Field(default_factory=lambda: [0.0, 0.0, 50.0, 50.0])
    center: List[float] = Field(default_factory=lambda: [25.0, 25.0])
    frame_id: int = 0
    zone_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    confirmed: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventResponse(BaseModel):
    event_id: str
    event_type: str
    severity: str
    camera_id: str
    zone_id: Optional[str] = None
    track_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]
    center: List[float]
    frame_id: int
    timestamp: float
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    confirmed: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventConfirmRequest(BaseModel):
    confirmed: bool
    notes: Optional[str] = None


class EventListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    events: List[EventResponse]


# ── Camera Models ─────────────────────────────────────────────────────────────

class CameraHeartbeat(BaseModel):
    fps: float = 0.0
    status: str = "ONLINE"
    frame_id: int = 0
    message: Optional[str] = None


class CameraConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    name: Optional[str] = None
    location: Optional[str] = None


class CameraStatus(BaseModel):
    id: str
    name: str
    enabled: bool = True
    status: str = "OFFLINE"  # ONLINE, OFFLINE, DEGRADED
    source_type: str = "rtsp"
    fps: float = 0.0
    total_frames: int = 0
    total_events: int = 0
    location: Optional[str] = None
    last_seen: Optional[float] = None


# ── Watchlist Models ──────────────────────────────────────────────────────────

class WatchlistPlateCreate(BaseModel):
    plate: str
    notes: Optional[str] = None
    severity: str = "CRITICAL"


class WatchlistPlateResponse(BaseModel):
    plate: str
    notes: Optional[str] = None
    severity: str = "CRITICAL"
    created_at: float = Field(default_factory=time.time)


class WatchlistPersonCreate(BaseModel):
    name: str
    notes: Optional[str] = None
    severity: str = "CRITICAL"
    embedding: Optional[List[float]] = None


class WatchlistPersonResponse(BaseModel):
    name: str
    num_embeddings: int = 1
    notes: Optional[str] = None
    severity: str = "CRITICAL"
    created_at: float = Field(default_factory=time.time)


# ── Webhook Models ────────────────────────────────────────────────────────────

class WebhookCreate(BaseModel):
    id: str
    name: str
    url: str
    method: str = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    events: List[str] = Field(default_factory=list)
    enabled: bool = True
    retry_attempts: int = 3
    retry_delay_s: float = 2.0


class WebhookResponse(BaseModel):
    id: str
    name: str
    url: str
    method: str
    headers: Dict[str, str]
    events: List[str]
    enabled: bool
    retry_attempts: int
    retry_delay_s: float
    success_count: int = 0
    failure_count: int = 0
    last_status: Optional[int] = None
    last_attempt_at: Optional[float] = None


# ── System Health & Metrics ───────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    status: str = "OK"
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "OK"
    uptime_s: float
    version: str = "0.7.0"
    timestamp: float = Field(default_factory=time.time)
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    uptime_s: float
    total_events: int
    events_by_severity: Dict[str, int]
    events_by_type: Dict[str, int]
    active_cameras: int
    total_cameras: int
    connected_sse_clients: int
    webhook_deliveries_total: int
    webhook_failures_total: int
