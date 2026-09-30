#!/usr/bin/env python3
"""
run_p2.py
=========
IBVAP P2 — Full pipeline demo:
  ingest → detect/track → fence (zones + tripwires) → event engine → alerts

Reads zone config from configs/zones_p2_demo.yaml (created automatically
if missing).  For a real camera, point --zones at your configs/zones.yaml.

Usage
-----
    python run_p2.py --source data/samples/general/test_clip_720p.mp4
    python run_p2.py --source rtsp://... --zones configs/zones.yaml --camera cam_01
    python run_p2.py --source video.mp4 --output output_p2.mp4 --no-display

Controls: q / ESC = quit  |  s = screenshot  |  p = pause
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.analytics.fence.fence_engine import FenceEngine
from services.analytics.fence.zone_renderer import ZoneRenderer
from services.detect_track.renderer import Renderer
from services.detect_track.tracker  import ByteTracker
from services.event_engine.engine   import EventEngine
from services.event_engine.logger   import EventLogger
from services.event_engine.models   import Severity

# ── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


# =============================================================================
# Demo zones config (auto-generated for testing)
# =============================================================================

DEMO_ZONES_YAML = "configs/zones_p2_demo.yaml"

_DEMO_CONFIG = {
    "zones": {
        "demo_zones": {
            "polygons": [
                {
                    "id": "zone_left",
                    "name": "Left Zone",
                    "enabled": True,
                    "points": [[0.0, 0.0], [0.45, 0.0],
                               [0.45, 1.0], [0.0, 1.0]],
                    "triggers": {
                        "on_enter": True,
                        "on_exit": False,
                        "dwell_seconds": 5,
                        "classes": ["person", "car", "truck",
                                    "motorcycle", "bus", "bicycle"],
                    },
                    "severity": "HIGH",
                },
                {
                    "id": "zone_right",
                    "name": "Right Zone",
                    "enabled": True,
                    "points": [[0.55, 0.0], [1.0, 0.0],
                               [1.0, 1.0], [0.55, 1.0]],
                    "triggers": {
                        "on_enter": True,
                        "on_exit": True,
                        "dwell_seconds": 8,
                        "classes": ["person", "car", "truck",
                                    "motorcycle", "bus", "bicycle"],
                    },
                    "severity": "MEDIUM",
                },
            ],
            "tripwires": [
                {
                    "id": "wire_centre",
                    "name": "Centre Line",
                    "enabled": True,
                    "start": [0.5, 0.0],
                    "end":   [0.5, 1.0],
                    "direction": "both",
                    "classes": ["person", "car", "truck",
                                "motorcycle", "bus", "bicycle"],
                    "severity": "CRITICAL",
                },
            ],
        }
    }
}


def _ensure_demo_zones() -> None:
    p = Path(DEMO_ZONES_YAML)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            yaml.dump(_DEMO_CONFIG, f, default_flow_style=False)
        logger.info(f"Created demo zones config: {p}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="IBVAP P2 — fence + event engine demo")
    parser.add_argument("--source",    required=True)
    parser.add_argument("--model",     default="yolov8n.pt")
    parser.add_argument("--device",    default="cuda:0")
    parser.add_argument("--no-half",   action="store_true")
    parser.add_argument("--conf",      type=float, default=0.35)
    parser.add_argument("--target-fps",type=float, default=10.0)
    parser.add_argument("--zones",     default=DEMO_ZONES_YAML,
                        help="zones.yaml path (default: demo config)")
    parser.add_argument("--zones-key", default="demo_zones",
                        help="key inside zones.yaml (default: demo_zones)")
    parser.add_argument("--camera",    default="demo",
                        help="camera ID (used in event logs)")
    parser.add_argument("--dedup",     type=float, default=10.0,
                        help="dedup window in seconds (default: 10)")
    parser.add_argument("--output",    default=None)
    parser.add_argument("--no-display",action="store_true")
    parser.add_argument("--max-frames",type=int, default=0)
    args = parser.parse_args()

    _ensure_demo_zones()

    # ── Open source ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logger.error(f"Cannot open: {args.source}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip_n  = max(1, round(src_fps / args.target_fps))
    logger.info(f"Source: {width}x{height} @ {src_fps:.1f} FPS | skip={skip_n}")

    # ── Build pipeline ───────────────────────────────────────────────────────
    tracker = ByteTracker(
        model_path=args.model,
        device=args.device,
        half=not args.no_half,
        conf=args.conf,
    )

    fence = FenceEngine.from_config(
        zones_yaml=args.zones,
        camera_id=args.camera,
        frame_w=width,
        frame_h=height,
        zones_key=args.zones_key,
    )

    ev_engine = EventEngine(
        dedup_window_s=args.dedup,
        snapshot_dir="data/snapshots",
    )
    ev_logger  = EventLogger(log_path="data/logs/events.jsonl")
    renderer   = Renderer(fps_window=30)
    z_renderer = ZoneRenderer(fence, alpha=0.20)

    # ── Optional writer ───────────────────────────────────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.target_fps,
                                 (width, height))

    # ── Window ───────────────────────────────────────────────────────────────
    WIN = "IBVAP P2 — fence + events  (q=quit)"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(width, 1280), min(height, 720))

    # ── Main loop ─────────────────────────────────────────────────────────────
    raw_count  = 0
    proc_count = 0
    total_events = 0
    paused     = False
    t_start    = time.monotonic()

    logger.info("Running — press 'q' or ESC to quit")

    while True:
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("p"), ord(" ")): paused = False
            elif key in (ord("q"), 27):     break
            continue

        ok, frame = cap.read()
        if not ok:
            logger.info("Stream ended.")
            break

        raw_count += 1
        if raw_count % skip_n != 0:
            continue
        proc_count += 1

        # ── Detect + track ────────────────────────────────────────────────
        tracks = tracker.update(
            frame=frame, frame_id=proc_count,
            camera_id=args.camera, timestamp=time.monotonic(),
        )

        # ── Fence analytics ────────────────────────────────────────────────
        raw_events = fence.update(tracks, proc_count)

        # ── Event engine ───────────────────────────────────────────────────
        fired_events = ev_engine.process(raw_events, frame)
        ev_logger.log_many(fired_events)
        total_events += len(fired_events)

        # Trigger visual flash for high-severity events
        for evt in fired_events:
            if evt.severity in (Severity.CRITICAL, Severity.HIGH):
                z_renderer.trigger_alert()

        # ── Render ────────────────────────────────────────────────────────
        annotated = z_renderer.draw(frame)               # zones overlay
        annotated = renderer.draw(                        # tracks + HUD
            annotated, tracks, frame_id=proc_count,
            extra_info=f"events={total_events}",
        )

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break
            elif key == ord("s"):
                path = f"screenshot_p2_{int(time.time())}.jpg"
                cv2.imwrite(path, annotated)
                logger.info(f"Screenshot: {path}")
            elif key in (ord("p"), ord(" ")):
                paused = True

        if proc_count % 30 == 0:
            elapsed = time.monotonic() - t_start
            logger.info(
                f"frame={proc_count:5d} | "
                f"fps={proc_count/elapsed:5.1f} | "
                f"tracks={len(tracks):3d} | "
                f"events_total={total_events}"
            )

        if args.max_frames and proc_count >= args.max_frames:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    ev_logger.print_summary()
    elapsed = time.monotonic() - t_start
    logger.info(f"Done — {proc_count} frames in {elapsed:.1f}s "
                f"({proc_count/elapsed:.1f} FPS avg)")


if __name__ == "__main__":
    main()
