"""
services/event_engine/logger.py
================================
IBVAP P2 — Event logger: JSON-lines file + rich console output.

Every confirmed (post-dedup) event is:
  1. Appended as a single JSON line to  data/logs/events.jsonl
  2. Printed to stdout with colour-coded severity

The JSONL format allows:
  - grep / jq queries for incident review
  - Streaming import into PostgreSQL (P7)
  - Natural-language search (P9)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger

from services.event_engine.models import Event, Severity


# Severity → ANSI colour codes for terminal
_SEV_COLOR = {
    "CRITICAL": "\033[1;31m",   # bold red
    "HIGH":     "\033[0;31m",   # red
    "MEDIUM":   "\033[0;33m",   # yellow
    "LOW":      "\033[0;36m",   # cyan
}
_RESET = "\033[0m"


class EventLogger:
    """
    Appends events to a JSON-lines file and prints to console.

    Parameters
    ----------
    log_path    : path to the .jsonl log file.
    console     : if True, print each event to stdout.
    """

    def __init__(
        self,
        log_path:  str = "data/logs/events.jsonl",
        console:   bool = True,
    ) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._console = console
        self._session_count = 0

        logger.info(f"EventLogger writing to: {self._path}")

    def log(self, event: Event) -> None:
        """Write one event to file and optionally print to console."""
        d = event.to_dict()

        # Append JSON line
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

        self._session_count += 1

        if self._console:
            self._print_event(event)

    def log_many(self, events: List[Event]) -> None:
        for e in events:
            self.log(e)

    @property
    def session_count(self) -> int:
        return self._session_count

    def print_summary(self) -> None:
        logger.info(
            f"EventLogger session summary: "
            f"{self._session_count} events written to {self._path}"
        )

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_event(evt: Event) -> None:
        sev   = evt.severity.value
        color = _SEV_COLOR.get(sev, "")
        ts    = time.strftime("%H:%M:%S", time.localtime(evt.timestamp))

        meta_str = ""
        if "dwell_s" in evt.metadata:
            meta_str = f"  dwell={evt.metadata['dwell_s']:.1f}s"
        if "direction" in evt.metadata:
            meta_str += f"  dir={evt.metadata['direction']}"
        if "plate_text" in evt.metadata:
            meta_str += f"  plate={evt.metadata['plate_text']}"

        snap = ""
        if evt.snapshot_path:
            snap = f"  📷 {Path(evt.snapshot_path).name}"

        print(
            f"{color}[{ts}] {sev:<8} {evt.event_type.value:<18} "
            f"cam={evt.camera_id}  "
            f"track={evt.track.track_id}({evt.track.class_name})  "
            f"zone={evt.zone_id or '-'}"
            f"{meta_str}{snap}{_RESET}"
        )
