"""
services/api/webhooks/dispatcher.py
===================================
IBVAP P7 — Asynchronous C2 Command & Control Webhook Dispatcher.
Supports event filtering, exponential backoff retries, and delivery telemetry.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from loguru import logger

from services.api.models import EventResponse, WebhookCreate, WebhookResponse
from services.api.state import state


class WebhookDispatcher:
    """Dispatches surveillance alerts to C2 endpoints asynchronously."""

    def __init__(self, config_path: str = "configs/rules.yaml") -> None:
        self.config_path = Path(config_path)
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        """Initialize HTTP client and load configurations."""
        self._client = httpx.AsyncClient(timeout=5.0)
        self.load_from_config()

    async def stop(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def load_from_config(self) -> None:
        """Read webhooks from rules.yaml."""
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for item in data.get("webhooks", []):
                wh_id = item.get("id")
                if not wh_id:
                    continue
                # Replace ${VAR} in headers
                headers: Dict[str, str] = {}
                for k, v in item.get("headers", {}).items():
                    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                        env_var = v[2:-1]
                        headers[k] = os.getenv(env_var, "default-c2-key")
                    else:
                        headers[k] = str(v)

                model = WebhookResponse(
                    id=wh_id,
                    name=item.get("name", wh_id),
                    url=item.get("url", ""),
                    method=item.get("method", "POST"),
                    headers=headers,
                    events=item.get("events", []),
                    enabled=item.get("enabled", True),
                    retry_attempts=item.get("retry_attempts", 3),
                    retry_delay_s=item.get("retry_delay_s", 2.0),
                )
                state.webhooks[wh_id] = model
            logger.info(f"Loaded {len(state.webhooks)} webhook destinations from {self.config_path}")
        except Exception as e:
            logger.warning(f"Error loading webhooks from {self.config_path}: {e}")

    def register(self, wh: WebhookCreate) -> WebhookResponse:
        model = WebhookResponse(
            id=wh.id,
            name=wh.name,
            url=wh.url,
            method=wh.method,
            headers=wh.headers,
            events=wh.events,
            enabled=wh.enabled,
            retry_attempts=wh.retry_attempts,
            retry_delay_s=wh.retry_delay_s,
        )
        state.webhooks[wh.id] = model
        return model

    def unregister(self, wh_id: str) -> bool:
        if wh_id in state.webhooks:
            del state.webhooks[wh_id]
            return True
        return False

    async def dispatch_event(self, event: EventResponse) -> None:
        """Dispatch event to all applicable active webhook endpoints in the background."""
        for wh in list(state.webhooks.values()):
            if not wh.enabled:
                continue
            # Event type filtering
            if wh.events and event.event_type not in wh.events:
                continue

            # Schedule async post task
            asyncio.create_task(self._deliver(wh, event.model_dump()))

    async def test_dispatch(self, wh: WebhookResponse, payload: Dict[str, Any]) -> tuple[bool, int, str]:
        """Synchronously deliver a payload to test a webhook."""
        client = self._client or httpx.AsyncClient(timeout=5.0)
        try:
            resp = await client.request(
                method=wh.method,
                url=wh.url,
                json=payload,
                headers=wh.headers,
            )
            success = 200 <= resp.status_code < 300
            return success, resp.status_code, resp.text[:200]
        except Exception as e:
            return False, 0, str(e)

    async def _deliver(self, wh: WebhookResponse, payload: Dict[str, Any]) -> None:
        client = self._client or httpx.AsyncClient(timeout=5.0)
        wh.last_attempt_at = time.time()

        for attempt in range(1, wh.retry_attempts + 1):
            try:
                resp = await client.request(
                    method=wh.method,
                    url=wh.url,
                    json=payload,
                    headers=wh.headers,
                )
                wh.last_status = resp.status_code
                if 200 <= resp.status_code < 300:
                    wh.success_count += 1
                    state.webhook_deliveries_total += 1
                    logger.debug(f"Webhook {wh.id} delivered event {payload.get('event_id')} (status {resp.status_code})")
                    return
                else:
                    logger.warning(
                        f"Webhook {wh.id} returned status {resp.status_code} (attempt {attempt}/{wh.retry_attempts})"
                    )
            except Exception as exc:
                wh.last_status = 0
                logger.warning(
                    f"Webhook {wh.id} delivery attempt {attempt}/{wh.retry_attempts} failed: {exc}"
                )

            if attempt < wh.retry_attempts:
                await asyncio.sleep(wh.retry_delay_s * (2 ** (attempt - 1)))

        wh.failure_count += 1
        state.webhook_failures_total += 1
        logger.error(f"Webhook {wh.id} failed after {wh.retry_attempts} attempts")


# Global singleton dispatcher
dispatcher = WebhookDispatcher()
