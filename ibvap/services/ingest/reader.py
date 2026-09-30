"""
services/ingest/reader.py
=========================
IBVAP P1 — Thread-safe video frame reader.

Supports:
  - Local video files  (type: file)
  - RTSP streams       (type: rtsp)
  - USB / webcam       (type: usb)

The reader runs in a background thread and fills a bounded queue.
The consumer pops frames at its own pace; stale frames are dropped
automatically when the queue is full (keeps latency low).

Interface:
    reader = FrameReader(config)
    reader.start()
    for frame_data in reader:            # FrameData namedtuple
        process(frame_data.frame)
    reader.stop()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Dict, Generator, Iterator, Optional

import cv2
import numpy as np
from loguru import logger


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class FrameData:
    """A single decoded video frame with metadata."""
    frame:      np.ndarray          # BGR, HxWx3 uint8
    frame_id:   int                 # monotonically increasing per reader
    timestamp:  float               # wall-clock time (time.monotonic())
    camera_id:  str                 # from config
    source_fps: float               # reported FPS of the source


@dataclass
class ReaderStats:
    """Cumulative statistics exposed for monitoring."""
    frames_read:    int   = 0
    frames_dropped: int   = 0   # dropped because queue was full
    reconnects:     int   = 0
    last_fps:       float = 0.0


# ── Main class ────────────────────────────────────────────────────────────────

class FrameReader:
    """
    Threaded video reader that produces FrameData objects.

    Parameters
    ----------
    camera_cfg : dict
        One entry from cameras.yaml  (already parsed by yaml.safe_load).
    queue_maxsize : int
        Maximum frames buffered between reader and consumer.
        Older frames are dropped when full to keep latency low.
    """

    # COCO class IDs we actually care about detecting
    _KEEP_CLASSES = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck

    def __init__(
        self,
        camera_cfg: Dict,
        queue_maxsize: int = 4,
    ) -> None:
        src  = camera_cfg["source"]
        cap  = camera_cfg.get("capture", {})

        self.camera_id:   str   = camera_cfg["id"]
        self.source_type: str   = src["type"]           # rtsp | file | usb
        self.target_fps:  float = cap.get("target_fps", 10)
        self.loop_file:   bool  = src.get("loop", False)

        # Build the OpenCV source string
        if src["type"] == "rtsp":
            self._source = src["url"]
            self._transport = src.get("transport", "tcp")
            self._reconnect_delay  = src.get("reconnect_delay_s", 5)
            self._reconnect_max    = src.get("reconnect_max_retries", -1)
        elif src["type"] == "file":
            self._source = src["path"]
            self._reconnect_delay  = 0
            self._reconnect_max    = 0   # no reconnect for files
        elif src["type"] == "usb":
            self._source = int(src.get("device_index", 0))
            self._reconnect_delay  = 2
            self._reconnect_max    = 3
        else:
            raise ValueError(f"Unknown source type: {src['type']}")

        self._queue:    Queue[FrameData] = Queue(maxsize=queue_maxsize)
        self._stop_evt: threading.Event  = threading.Event()
        self._thread:   Optional[threading.Thread] = None
        self._stats:    ReaderStats = ReaderStats()

        self._frame_id: int   = 0
        self._cap:      Optional[cv2.VideoCapture] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> "FrameReader":
        """Start the background capture thread."""
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"reader-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[{self.camera_id}] Reader started — source: {self._source}")
        return self

    def stop(self) -> None:
        """Signal the capture thread to stop and join it."""
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info(
            f"[{self.camera_id}] Reader stopped — "
            f"read={self._stats.frames_read}, "
            f"dropped={self._stats.frames_dropped}, "
            f"reconnects={self._stats.reconnects}"
        )

    def read(self, timeout: float = 2.0) -> Optional[FrameData]:
        """
        Pop the next frame from the queue.
        Returns None if no frame arrives within *timeout* seconds.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def __iter__(self) -> Iterator[FrameData]:
        """Iterate until stop() is called."""
        while not self._stop_evt.is_set():
            fd = self.read(timeout=1.0)
            if fd is not None:
                yield fd

    @property
    def stats(self) -> ReaderStats:
        return self._stats

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _open_capture(self) -> bool:
        """Open (or re-open) the cv2.VideoCapture."""
        if self._cap and self._cap.isOpened():
            self._cap.release()

        if self.source_type == "rtsp":
            # Prefer TCP transport; set environment hint before opening
            cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
            # Small buffer to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.source_type == "usb":
            import platform
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self._source, backend)
            # Request resolution from capture config if available
            w = self.target_fps  # just checking
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self._source)

        if not cap.isOpened():
            logger.warning(f"[{self.camera_id}] Could not open: {self._source}")
            return False

        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        self._source_fps = src_fps
        self._skip_n = max(1, round(src_fps / self.target_fps))
        self._cap = cap
        logger.success(
            f"[{self.camera_id}] Opened — "
            f"source_fps={src_fps:.1f}, "
            f"processing every {self._skip_n} frame(s) "
            f"→ ~{src_fps/self._skip_n:.1f} FPS"
        )
        return True

    def _capture_loop(self) -> None:
        """Main loop: read → maybe skip → push to queue."""
        retries = 0
        if not self._open_capture():
            logger.error(f"[{self.camera_id}] Initial open failed")
            return

        _raw_count = 0          # count raw decoded frames for skip logic
        _fps_t0    = time.monotonic()
        _fps_count = 0

        while not self._stop_evt.is_set():
            if self._cap is None or not self._cap.isOpened():
                # Reconnect logic
                if self._reconnect_max >= 0 and retries >= self._reconnect_max:
                    logger.error(
                        f"[{self.camera_id}] Max reconnect attempts reached ({retries})"
                    )
                    break
                logger.warning(
                    f"[{self.camera_id}] Stream lost — reconnecting in "
                    f"{self._reconnect_delay}s (attempt {retries+1})"
                )
                time.sleep(self._reconnect_delay)
                if self._open_capture():
                    self._stats.reconnects += 1
                    retries = 0
                else:
                    retries += 1
                continue

            ok, frame = self._cap.read()
            if not ok:
                # End of file
                if self.source_type == "file" and self.loop_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # For RTSP: treat as stream drop
                if self._cap:
                    self._cap.release()
                    self._cap = None  # type: ignore[assignment]
                continue

            _raw_count += 1
            self._stats.frames_read += 1

            # Frame-skip: only process every Nth raw frame
            if _raw_count % self._skip_n != 0:
                continue

            self._frame_id += 1
            _fps_count += 1

            # FPS estimation (rolling over last second)
            now = time.monotonic()
            elapsed = now - _fps_t0
            if elapsed >= 1.0:
                self._stats.last_fps = _fps_count / elapsed
                _fps_t0    = now
                _fps_count = 0

            fd = FrameData(
                frame=frame,
                frame_id=self._frame_id,
                timestamp=time.monotonic(),
                camera_id=self.camera_id,
                source_fps=getattr(self, "_source_fps", self.target_fps),
            )

            # Non-blocking put: drop frame if consumer is too slow
            try:
                self._queue.put_nowait(fd)
            except Full:
                self._stats.frames_dropped += 1

        if self._cap:
            self._cap.release()
