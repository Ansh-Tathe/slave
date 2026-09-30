"""
services/hardening/buffer.py
============================
IBVAP P8 — Bounded memory backpressure queue with drop policies.
Prevents out-of-memory (OOM) crashes when GPU inference cannot keep up with ingest FPS.
"""

from __future__ import annotations

import queue
import threading
import time
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


class DropPolicy(str, Enum):
    DROP_OLDEST = "DROP_OLDEST"  # Discard oldest frame to keep pipeline near real-time
    DROP_NEWEST = "DROP_NEWEST"  # Reject new incoming frame
    BLOCK = "BLOCK"              # Block producer until space is available


class BoundedFrameBuffer(Generic[T]):
    """
    Thread-safe bounded ring buffer with configurable overflow policies.
    Guarantees fixed memory bounds under heavy load or processing stalls.
    """

    def __init__(self, maxsize: int = 30, policy: DropPolicy = DropPolicy.DROP_OLDEST) -> None:
        self.maxsize = maxsize
        self.policy = policy
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._closed = False

        # Telemetry
        self.total_pushed: int = 0
        self.total_popped: int = 0
        self.total_dropped: int = 0
        self._last_drop_log: float = 0.0

    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        """
        Put an item into the buffer according to the DropPolicy.
        Returns True if item was added, False if dropped or buffer closed.
        """
        if self._closed:
            return False

        with self._lock:
            self.total_pushed += 1

            if self.policy == DropPolicy.BLOCK:
                try:
                    self._queue.put(item, block=True, timeout=timeout)
                    return True
                except queue.Full:
                    self.total_dropped += 1
                    return False

            if self._queue.full():
                self.total_dropped += 1
                now = time.monotonic()
                if now - self._last_drop_log > 2.0:
                    logger.warning(
                        f"FrameBuffer saturated (size {self.qsize()}/{self.maxsize}). "
                        f"Dropped {self.total_dropped} frames total (Policy: {self.policy.value})."
                    )
                    self._last_drop_log = now

                if self.policy == DropPolicy.DROP_NEWEST:
                    return False

                if self.policy == DropPolicy.DROP_OLDEST:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(item)
                        return True
                    except queue.Full:
                        return False

            # If not full, put directly
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self.total_dropped += 1
                return False

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[T]:
        """Retrieve next frame from buffer."""
        try:
            item = self._queue.get(block=block, timeout=timeout)
            with self._lock:
                self.total_popped += 1
            return item
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def close(self) -> None:
        """Close buffer and unblock pending operations."""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "maxsize": self.maxsize,
                "current_size": self.qsize(),
                "total_pushed": self.total_pushed,
                "total_popped": self.total_popped,
                "total_dropped": self.total_dropped,
                "drop_rate_pct": (
                    round((self.total_dropped / self.total_pushed) * 100, 2)
                    if self.total_pushed > 0
                    else 0.0
                ),
            }
