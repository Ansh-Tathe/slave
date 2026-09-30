"""
services/hardening/watchdog.py
==============================
IBVAP P8 — Stream Ingest Watchdog and Auto-Recovery.
Monitors RTSP/video feeds for disconnects, frozen frames, or network lag,
and executes self-healing reconnection strategies.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable, Dict, Optional

from loguru import logger


class StreamState:
    ONLINE = "ONLINE"
    STALLED = "STALLED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class StreamWatchdog:
    """
    Monitors camera ingest health and automatically initiates reconnections.

    Parameters
    ----------
    camera_id       : Unique identifier of camera.
    timeout_s       : Max seconds between frames before declaring stream stalled.
    reconnect_fn    : Callback invoked to re-establish connection.
    max_retries     : Max consecutive reconnect attempts (-1 = infinite).
    base_delay_s    : Initial delay before reconnecting.
    max_delay_s     : Cap on exponential backoff delay.
    """

    def __init__(
        self,
        camera_id: str,
        timeout_s: float = 5.0,
        reconnect_fn: Optional[Callable[[], bool]] = None,
        max_retries: int = -1,
        base_delay_s: float = 2.0,
        max_delay_s: float = 30.0,
    ) -> None:
        self.camera_id = camera_id
        self.timeout_s = timeout_s
        self.reconnect_fn = reconnect_fn
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s

        self.state: str = StreamState.ONLINE
        self.last_frame_time: float = time.monotonic()
        self.last_frame_id: int = 0
        self.total_reconnects: int = 0
        self.consecutive_failures: int = 0

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background watchdog monitoring thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self.last_frame_time = time.monotonic()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info(f"StreamWatchdog started for camera '{self.camera_id}' (timeout: {self.timeout_s}s)")

    def stop(self) -> None:
        """Stop the watchdog monitoring thread."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info(f"StreamWatchdog stopped for camera '{self.camera_id}'")

    def heartbeat(self, frame_id: int = 0) -> None:
        """Called by reader/decoder every time a valid frame is received."""
        with self._lock:
            self.last_frame_time = time.monotonic()
            self.last_frame_id = frame_id
            if self.state != StreamState.ONLINE:
                logger.info(f"Camera '{self.camera_id}' stream restored to ONLINE (frame #{frame_id})")
                self.state = StreamState.ONLINE
                self.consecutive_failures = 0

    def check_health(self) -> bool:
        """
        Check if the stream is healthy. Returns True if frame was received recently,
        False if stream has stalled.
        """
        with self._lock:
            elapsed = time.monotonic() - self.last_frame_time
            if elapsed > self.timeout_s:
                if self.state == StreamState.ONLINE:
                    self.state = StreamState.STALLED
                    logger.warning(
                        f"StreamWatchdog alert: Camera '{self.camera_id}' stalled. "
                        f"No frame for {elapsed:.1f}s (threshold: {self.timeout_s}s)"
                    )
                return False
            return True

    def attempt_reconnect(self) -> bool:
        """Execute a reconnection attempt with exponential backoff and jitter."""
        if not self.reconnect_fn:
            return False

        with self._lock:
            self.state = StreamState.RECONNECTING
            self.consecutive_failures += 1
            attempt = self.consecutive_failures

        if self.max_retries != -1 and attempt > self.max_retries:
            with self._lock:
                self.state = StreamState.FAILED
            logger.error(f"Camera '{self.camera_id}' exceeded max reconnect retries ({self.max_retries}). Marked FAILED.")
            return False

        # Calculate backoff with jitter
        delay = min(self.max_delay_s, self.base_delay_s * (2 ** (min(attempt - 1, 5))))
        jitter = random.uniform(0.8, 1.2)
        sleep_time = delay * jitter

        logger.info(
            f"Camera '{self.camera_id}' reconnecting (attempt {attempt}, waiting {sleep_time:.1f}s)..."
        )
        time.sleep(sleep_time)

        try:
            success = self.reconnect_fn()
            with self._lock:
                if success:
                    self.state = StreamState.ONLINE
                    self.last_frame_time = time.monotonic()
                    self.total_reconnects += 1
                    self.consecutive_failures = 0
                    logger.info(f"Camera '{self.camera_id}' reconnected successfully!")
                    return True
                else:
                    logger.warning(f"Camera '{self.camera_id}' reconnect attempt {attempt} failed.")
                    return False
        except Exception as e:
            logger.error(f"Camera '{self.camera_id}' reconnect callback raised exception: {e}")
            return False

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                healthy = self.check_health()
                if not healthy and self._running:
                    self.attempt_reconnect()
            except Exception as e:
                logger.error(f"Watchdog monitor error for {self.camera_id}: {e}")

            time.sleep(1.0)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "state": self.state,
                "seconds_since_last_frame": round(time.monotonic() - self.last_frame_time, 2),
                "last_frame_id": self.last_frame_id,
                "consecutive_failures": self.consecutive_failures,
                "total_reconnects": self.total_reconnects,
            }
