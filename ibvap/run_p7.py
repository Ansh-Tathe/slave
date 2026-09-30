#!/usr/bin/env python3
"""
run_p7.py
=========
IBVAP P7 — FastAPI REST API + Command Dashboard + C2 Webhooks Launcher.

Usage:
  python run_p7.py
  python run_p7.py --port 8080 --simulate-events
  python run_p7.py --host 0.0.0.0 --port 8000

Dashboard UI: http://localhost:8000/
API Docs:     http://localhost:8000/docs
Health:       http://localhost:8000/health
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import threading
import time
from pathlib import Path

import uvicorn
from loguru import logger

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.api.models import EventCreate
from services.api.state import state
from services.api.webhooks.dispatcher import dispatcher

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)


def _event_simulator_worker():
    """Background worker that pushes simulated alert events every few seconds."""
    logger.info("Event simulator started: generating synthetic alerts every 3-5 seconds...")
    sample_cameras = ["cam_01", "cam_test_file"]
    sample_classes = ["person", "car", "truck", "motorcycle"]
    event_types = [
        ("TRIPWIRE_CROSS", "HIGH"),
        ("ZONE_ENTER", "MEDIUM"),
        ("LOITERING", "HIGH"),
        ("ANPR_WATCHLIST_HIT", "CRITICAL"),
        ("FACE_WATCHLIST_HIT", "CRITICAL"),
        ("GATHERING", "HIGH"),
        ("NIGHT_MOVEMENT", "HIGH"),
        ("ANPR_READ", "LOW"),
    ]

    time.sleep(2.0)  # Wait for uvicorn to initialize
    while True:
        try:
            ev_type, default_sev = random.choice(event_types)
            cam = random.choice(sample_cameras)
            cls_name = "car" if "ANPR" in ev_type else random.choice(sample_classes)
            track_id = random.randint(10, 800)

            # Build event payload
            from services.api.models import EventResponse
            import uuid

            ev = EventResponse(
                event_id=str(uuid.uuid4()),
                event_type=ev_type,
                severity=default_sev,
                camera_id=cam,
                zone_id="cam_01_zones" if cam == "cam_01" else None,
                track_id=track_id,
                class_name=cls_name,
                class_id=0 if cls_name == "person" else 2,
                confidence=round(random.uniform(0.82, 0.98), 2),
                bbox=[100.0, 150.0, 220.0, 380.0],
                center=[160.0, 265.0],
                frame_id=random.randint(100, 5000),
                timestamp=time.time(),
                metadata={
                    "simulated": True,
                    "speed_kmh": round(random.uniform(5.0, 65.0), 1) if cls_name in ["car", "truck"] else None,
                    "plate": "KA01AB1234" if "ANPR" in ev_type else None,
                    "identity": "Suspect-Alpha" if "FACE" in ev_type else None,
                },
            )

            state.event_store.add(ev)
            state.camera_registry.record_event(cam)

            # Fire SSE broadcast
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(state.sse.broadcast(ev.model_dump()), loop)
                    asyncio.run_coroutine_threadsafe(dispatcher.dispatch_event(ev), loop)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Simulator loop error: {e}")

        time.sleep(random.uniform(3.0, 5.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="IBVAP P7 — REST API & Live Surveillance Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    parser.add_argument("--simulate-events", action="store_true", help="Periodically inject simulated surveillance alerts")
    args = parser.parse_args()

    print("=" * 72)
    print(" IBVAP P7 — Surveillance REST API & Operations Console")
    print("=" * 72)
    print(f" * Live Operations Dashboard : http://{args.host}:{args.port}/")
    print(f" * Swagger OpenAPI Explorer  : http://{args.host}:{args.port}/docs")
    print(f" * ReDoc API Reference       : http://{args.host}:{args.port}/redoc")
    print(f" * Real-Time Alert SSE Stream: http://{args.host}:{args.port}/api/v1/events/stream")
    print(f" * System Health Probe       : http://{args.host}:{args.port}/health")
    print("=" * 72)

    if args.simulate_events:
        sim_thread = threading.Thread(target=_event_simulator_worker, daemon=True)
        sim_thread.start()

    uvicorn.run(
        "services.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
