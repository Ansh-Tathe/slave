"""
services/api/main.py
====================
IBVAP P7 — FastAPI Application Factory.
Wires together routers, CORS, real-time SSE, webhook dispatcher, and operations dashboard.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger

from services.api.routers import (
    auth,
    cameras,
    events,
    health,
    reports,
    search,
    vlm,
    watchlist,
    webhooks,
)
from services.api.state import state
from services.api.webhooks.dispatcher import dispatcher

DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: starts up background resources and handles graceful shutdown."""
    logger.info("Initializing IBVAP API and state...")
    state.init()
    await dispatcher.start()
    yield
    logger.info("Shutting down IBVAP API...")
    await dispatcher.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="IBVAP — Border Video Analytics Platform API",
        description="RESTful API for real-time video surveillance events, cameras, watchlists, NL search, VLM, and C2 Command integration.",
        version="0.9.1",
        lifespan=lifespan,
    )

    # ── CORS Middleware ───────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Mount API v1 Routers ──────────────────────────────────────────────────
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(events.router, prefix="/api/v1")
    app.include_router(cameras.router, prefix="/api/v1")
    app.include_router(watchlist.router, prefix="/api/v1")
    app.include_router(webhooks.router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(reports.router, prefix="/api/v1")
    app.include_router(vlm.router, prefix="/api/v1")

    # Also mount health/metrics at root for standard monitoring endpoints
    app.include_router(health.router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def serve_dashboard():
        if DASHBOARD_FILE.exists():
            return FileResponse(DASHBOARD_FILE, media_type="text/html")
        return HTMLResponse(
            "<h1>IBVAP Command Center</h1><p>Dashboard UI file not found on server.</p>",
            status_code=404,
        )

    @app.get("/camera", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/live", response_class=HTMLResponse, include_in_schema=False)
    async def serve_camera_console():
        cam_file = Path(__file__).resolve().parent / "dashboard" / "camera.html"
        if cam_file.exists():
            return FileResponse(cam_file, media_type="text/html")
        return HTMLResponse(
            "<h1>IBVAP Camera Console</h1><p>Camera UI file not found on server.</p>",
            status_code=404,
        )

    return app


app = create_app()
