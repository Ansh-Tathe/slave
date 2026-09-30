"""
services/api/routers/health.py
==============================
IBVAP P7 — Health check and Prometheus-compatible metrics endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter

from services.api.models import HealthResponse, MetricsResponse
from services.api.state import state

router = APIRouter(tags=["Health & Telemetry"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Comprehensive health check for IBVAP system components."""
    return state.get_health()


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Telemetry metrics: event counts by severity/type, camera statuses, and queue depths."""
    return state.get_metrics()
