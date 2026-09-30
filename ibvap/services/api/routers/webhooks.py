"""
services/api/routers/webhooks.py
================================
IBVAP P7 — C2 Webhook management and test dispatch endpoints.
"""

from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from services.api.auth import require_role
from services.api.models import WebhookCreate, WebhookResponse
from services.api.state import state
from services.api.webhooks.dispatcher import dispatcher

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


@router.get("", response_model=List[WebhookResponse])
async def list_webhooks() -> List[WebhookResponse]:
    """List all registered C2 webhook destinations with runtime delivery stats."""
    return list(state.webhooks.values())


@router.post("", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def register_webhook(
    body: WebhookCreate,
    _user: dict = Depends(require_role("admin")),
) -> WebhookResponse:
    """Register a new C2 webhook destination. Requires admin role."""
    return dispatcher.register(body)


@router.delete("/{webhook_id}", status_code=status.HTTP_200_OK)
async def delete_webhook(
    webhook_id: str,
    _user: dict = Depends(require_role("admin")),
):
    """Unregister a webhook destination."""
    if not dispatcher.unregister(webhook_id):
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found")
    return {"message": f"Webhook {webhook_id} deleted"}


@router.post("/{webhook_id}/test")
async def test_webhook(
    webhook_id: str,
    _user: dict = Depends(require_role("operator")),
):
    """Send a synthetic test event to a configured webhook destination."""
    wh = state.webhooks.get(webhook_id)
    if not wh:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found")

    test_payload = {
        "event_id": "test-ping-00000000",
        "event_type": "TRIPWIRE_CROSS",
        "severity": "LOW",
        "camera_id": "cam_test",
        "track_id": 999,
        "class_name": "person",
        "confidence": 0.99,
        "timestamp": time.time(),
        "metadata": {"test": True, "message": "Manual C2 connection test from IBVAP dashboard"},
    }

    success, code, body = await dispatcher.test_dispatch(wh, test_payload)
    return {
        "success": success,
        "status_code": code,
        "response": body,
        "webhook_id": webhook_id,
        "url": wh.url,
    }
