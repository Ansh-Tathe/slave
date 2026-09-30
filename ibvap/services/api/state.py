"""
services/api/state.py
=====================
IBVAP P7 — Central in-memory state manager: Event Store, Camera Registry,
Watchlist Store, Metrics, and SSE Pub/Sub.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from loguru import logger

from services.api.models import (
    CameraStatus,
    ComponentHealth,
    EventCreate,
    EventResponse,
    HealthResponse,
    MetricsResponse,
    WatchlistPersonResponse,
    WatchlistPlateResponse,
    WebhookResponse,
)


class EventStore:
    """Thread-safe ring buffer and index for surveillance events."""

    def __init__(self, max_events: int = 10000, log_path: str = "data/logs/events.jsonl") -> None:
        self.max_events = max_events
        self.log_path = Path(log_path)
        self._events_list: deque[EventResponse] = deque(maxlen=max_events)
        self._events_by_id: Dict[str, EventResponse] = {}

    def load_from_disk(self) -> int:
        """Load historical events from JSONL log if present."""
        if not self.log_path.exists():
            return 0
        loaded = 0
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        event = EventResponse(**data)
                        self._events_list.append(event)
                        self._events_by_id[event.event_id] = event
                        loaded += 1
                    except Exception:
                        continue
            logger.info(f"Loaded {loaded} historical events from {self.log_path}")
        except Exception as e:
            logger.warning(f"Could not load historical events: {e}")
        return loaded

    def add(self, event: EventResponse) -> None:
        self._events_list.append(event)
        self._events_by_id[event.event_id] = event

    def get(self, event_id: str) -> Optional[EventResponse]:
        return self._events_by_id.get(event_id)

    def query(
        self,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        confirmed: Optional[bool] = None,
        since_s: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[EventResponse], int]:
        now = time.time()
        filtered: List[EventResponse] = []

        # Iterate in reverse chronological order (newest first)
        for ev in reversed(self._events_list):
            if camera_id and ev.camera_id != camera_id:
                continue
            if event_type and ev.event_type != event_type:
                continue
            if severity and ev.severity.upper() != severity.upper():
                continue
            if confirmed is not None and ev.confirmed != confirmed:
                continue
            if since_s is not None and (now - ev.timestamp) > since_s:
                continue
            filtered.append(ev)

        total = len(filtered)
        paginated = filtered[offset : offset + limit]
        return paginated, total

    def update_confirmation(self, event_id: str, confirmed: bool, notes: Optional[str] = None) -> Optional[EventResponse]:
        ev = self._events_by_id.get(event_id)
        if not ev:
            return None
        ev.confirmed = confirmed
        if notes:
            ev.metadata["operator_notes"] = notes
            ev.metadata["confirmed_at"] = time.time()
        return ev

    def count(self) -> int:
        return len(self._events_list)


class CameraRegistry:
    """Maintains active status and metrics for all configured cameras."""

    def __init__(self, config_path: str = "configs/cameras.yaml") -> None:
        self.config_path = Path(config_path)
        self.cameras: Dict[str, CameraStatus] = {}
        self.load_from_config()

    def load_from_config(self) -> None:
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for item in data.get("cameras", []):
                cam_id = item.get("id")
                if not cam_id:
                    continue
                meta = item.get("meta", {})
                source = item.get("source", {})
                self.cameras[cam_id] = CameraStatus(
                    id=cam_id,
                    name=item.get("name", cam_id),
                    enabled=item.get("enabled", True),
                    status="ONLINE" if item.get("enabled", True) else "OFFLINE",
                    source_type=source.get("type", "rtsp"),
                    fps=float(item.get("capture", {}).get("target_fps", 0.0)),
                    location=meta.get("location"),
                    last_seen=time.time() if item.get("enabled", True) else None,
                )
        except Exception as e:
            logger.warning(f"Error loading cameras from {self.config_path}: {e}")

    def get_all(self) -> List[CameraStatus]:
        return list(self.cameras.values())

    def get(self, cam_id: str) -> Optional[CameraStatus]:
        return self.cameras.get(cam_id)

    def record_heartbeat(self, cam_id: str, fps: float, frame_id: int, status: str = "ONLINE") -> CameraStatus:
        if cam_id not in self.cameras:
            self.cameras[cam_id] = CameraStatus(
                id=cam_id,
                name=cam_id,
                enabled=True,
                status=status,
                fps=fps,
                total_frames=frame_id,
                last_seen=time.time(),
            )
        else:
            cam = self.cameras[cam_id]
            cam.fps = fps
            cam.total_frames = max(cam.total_frames, frame_id)
            cam.status = status
            cam.last_seen = time.time()
        return self.cameras[cam_id]

    def record_event(self, cam_id: str) -> None:
        if cam_id in self.cameras:
            self.cameras[cam_id].total_events += 1


class WatchlistStore:
    """In-memory watchlists for plates and persons."""

    def __init__(self) -> None:
        self.plates: Dict[str, WatchlistPlateResponse] = {
            "KA01AB1234": WatchlistPlateResponse(
                plate="KA01AB1234",
                notes="Suspect vehicle - P3 test",
                severity="CRITICAL",
                created_at=time.time(),
            ),
            "MH12DE5678": WatchlistPlateResponse(
                plate="MH12DE5678",
                notes="Stolen vehicle report",
                severity="CRITICAL",
                created_at=time.time(),
            ),
        }
        self.persons: Dict[str, WatchlistPersonResponse] = {
            "Suspect-Alpha": WatchlistPersonResponse(
                name="Suspect-Alpha",
                num_embeddings=2,
                notes="Person of interest",
                severity="CRITICAL",
                created_at=time.time(),
            )
        }

    def add_plate(self, plate: str, notes: Optional[str] = None, severity: str = "CRITICAL") -> WatchlistPlateResponse:
        entry = WatchlistPlateResponse(
            plate=plate.upper().strip(),
            notes=notes,
            severity=severity,
            created_at=time.time(),
        )
        self.plates[entry.plate] = entry
        return entry

    def remove_plate(self, plate: str) -> bool:
        norm = plate.upper().strip()
        if norm in self.plates:
            del self.plates[norm]
            return True
        return False

    def add_person(self, name: str, notes: Optional[str] = None, severity: str = "CRITICAL") -> WatchlistPersonResponse:
        entry = WatchlistPersonResponse(
            name=name.strip(),
            num_embeddings=1,
            notes=notes,
            severity=severity,
            created_at=time.time(),
        )
        self.persons[entry.name] = entry
        return entry

    def remove_person(self, name: str) -> bool:
        if name in self.persons:
            del self.persons[name]
            return True
        return False


class SSEBroadcaster:
    """Async pub/sub message broadcaster for Server-Sent Events (SSE)."""

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def broadcast(self, data: dict) -> None:
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                dead.append(q)
            except Exception:
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    @property
    def client_count(self) -> int:
        return len(self._subscribers)


class AppState:
    """Global singleton state for IBVAP API."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.event_store = EventStore()
        self.camera_registry = CameraRegistry()
        self.watchlist_store = WatchlistStore()
        self.sse = SSEBroadcaster()
        self.webhooks: Dict[str, WebhookResponse] = {}
        self.webhook_deliveries_total: int = 0
        self.webhook_failures_total: int = 0

    def init(self) -> None:
        self.event_store.load_from_disk()

    @property
    def uptime_s(self) -> float:
        return time.time() - self.start_time

    def get_metrics(self) -> MetricsResponse:
        sev_counts: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        type_counts: Dict[str, int] = {}

        for ev in self.event_store._events_list:
            s = ev.severity.upper()
            sev_counts[s] = sev_counts.get(s, 0) + 1
            t = ev.event_type
            type_counts[t] = type_counts.get(t, 0) + 1

        active_cams = sum(1 for c in self.camera_registry.cameras.values() if c.status == "ONLINE")

        return MetricsResponse(
            uptime_s=round(self.uptime_s, 2),
            total_events=self.event_store.count(),
            events_by_severity=sev_counts,
            events_by_type=type_counts,
            active_cameras=active_cams,
            total_cameras=len(self.camera_registry.cameras),
            connected_sse_clients=self.sse.client_count,
            webhook_deliveries_total=self.webhook_deliveries_total,
            webhook_failures_total=self.webhook_failures_total,
        )

    def get_health(self) -> HealthResponse:
        components = {
            "event_store": ComponentHealth(status="OK", details=f"{self.event_store.count()} events in buffer"),
            "cameras": ComponentHealth(
                status="OK",
                details=f"{len(self.camera_registry.cameras)} configured",
            ),
            "watchlist": ComponentHealth(
                status="OK",
                details=f"{len(self.watchlist_store.plates)} plates, {len(self.watchlist_store.persons)} persons",
            ),
            "webhooks": ComponentHealth(
                status="OK" if self.webhook_failures_total == 0 else "WARN",
                details=f"{self.webhook_deliveries_total} sent, {self.webhook_failures_total} failed",
            ),
        }
        return HealthResponse(
            status="OK",
            uptime_s=round(self.uptime_s, 2),
            components=components,
        )


# Global singleton instance
state = AppState()
