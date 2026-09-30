#!/usr/bin/env python3
"""
run_webcam.py
=============
IBVAP — Real-Time Laptop Webcam Surveillance Pipeline.

Features:
  1. Live video capture from laptop webcam (DirectShow on Windows).
  2. YOLOv8n object detection & ByteTrack persistent identity tracking.
  3. Interactive Virtual Fence (Restricted Polygon Zone + Center Tripwire).
  4. Behaviour analytics (Loitering & Speed detection).
  5. Live integration with IBVAP Web Dashboard (dispatches alerts to http://localhost:8000).
  6. Real-time visual overlay with event banners and alert flash.

Usage:
  python run_webcam.py
  python run_webcam.py --camera-index 0 --conf 0.45
  python run_webcam.py --no-api   (standalone mode without web server sync)

Controls:
  q / ESC : Exit
  s       : Save high-res screenshot
  p       : Pause / resume
  f       : Toggle virtual fence overlay
"""

from __future__ import annotations

import argparse
import platform
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import httpx
import numpy as np
from loguru import logger

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.analytics.behaviour.behaviour_engine import BehaviourEngine
from services.analytics.fence.fence_engine import FenceEngine
from services.analytics.fence.tripwire import TripwireConfig
from services.analytics.fence.zone_checker import ZoneConfig
from services.analytics.fence.zone_renderer import ZoneRenderer
from services.detect_track.renderer import Renderer
from services.detect_track.tracker import ByteTracker
from services.event_engine.engine import EventEngine
from services.event_engine.logger import EventLogger
from services.event_engine.models import Event, Severity

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)


def dispatch_event_to_api(event: Event, api_url: str = "http://localhost:8000/api/v1/events") -> None:
    """Send event asynchronously to local FastAPI dashboard server."""
    payload = event.to_dict()

    def _worker():
        try:
            with httpx.Client(timeout=2.0) as client:
                client.post(api_url, json=payload)
        except Exception:
            pass  # Non-blocking if API is offline

    threading.Thread(target=_worker, daemon=True).start()


def send_heartbeat_to_api(
    camera_id: str = "cam_usb_0",
    fps: float = 0.0,
    frame_id: int = 0,
    api_url: str = "http://localhost:8000/api/v1/cameras",
) -> None:
    """Send camera heartbeat to API server."""
    url = f"{api_url}/{camera_id}/heartbeat"

    def _worker():
        try:
            with httpx.Client(timeout=1.5) as client:
                client.post(url, json={"fps": round(fps, 1), "frame_id": frame_id, "status": "ONLINE"})
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="IBVAP Live Laptop Webcam Surveillance Pipeline")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold (default: 0.45)")
    parser.add_argument("--target-fps", type=float, default=15.0, help="Target processing FPS (default: 15.0)")
    parser.add_argument("--no-api", action="store_true", help="Disable dispatching events to web dashboard")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path (default: yolov8n.pt)")
    args = parser.parse_args()

    print("=" * 72)
    print(" IBVAP — Real-Time Laptop Webcam Surveillance Console")
    print("=" * 72)
    print(f" * Webcam Device Index   : {args.camera_index}")
    print(f" * YOLO Detection Model  : {args.model} (conf: {args.conf})")
    print(f" * Dashboard Integration : {'Disabled' if args.no_api else 'Active (http://localhost:8000)'}")
    print(" * Controls              : [q] Quit  [s] Screenshot  [p] Pause  [f] Toggle Fence")
    print("=" * 72)

    # 1. Open Webcam
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    logger.info(f"Opening webcam at index {args.camera_index}...")
    cap = cv2.VideoCapture(args.camera_index, backend)
    if not cap.isOpened():
        logger.warning("Failed with primary backend, trying default cv2 backend...")
        cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        logger.error(f"Cannot access webcam at index {args.camera_index}. Please check device permissions.")
        sys.exit(1)

    # Query resolution
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    logger.success(f"Webcam active: {frame_w}x{frame_h} pixels")

    # 2. Initialize Analytics Subsystems
    logger.info("Loading YOLOv8 + ByteTrack tracking engine...")
    tracker = ByteTracker(
        model_path=args.model,
        conf=args.conf,
        device="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        warmup_runs=1,
    )

    # Define webcam fence zones:
    # Zone 1: "Restricted Zone" (left side of webcam feed)
    # Wire 1: "Perimeter Tripwire" (vertical line at x=55%)
    zone_cfgs = [
        ZoneConfig(
            zone_id="restricted_left",
            name="Restricted Zone",
            polygon=[(0.05, 0.15), (0.45, 0.15), (0.45, 0.85), (0.05, 0.85)],
            on_enter=True,
            dwell_seconds=3.0,
            classes=["person"],
            severity=Severity.HIGH,
        )
    ]
    wire_cfgs = [
        TripwireConfig(
            wire_id="tripwire_center",
            name="Tripwire Line",
            start=(0.55, 0.1),
            end=(0.55, 0.9),
            direction="both",
            classes=["person"],
            severity=Severity.CRITICAL,
        )
    ]

    fence = FenceEngine(zone_cfgs, wire_cfgs, frame_w=frame_w, frame_h=frame_h, camera_id="cam_usb_0")
    fence_renderer = ZoneRenderer(fence, alpha=0.22)
    renderer = Renderer()
    behaviour = BehaviourEngine(camera_id="cam_usb_0")
    event_engine = EventEngine(dedup_window_s=5.0, snapshot_dir="data/snapshots")
    event_logger = EventLogger(log_path="data/logs/events.jsonl")

    # Runtime state
    show_fence = True
    paused = False
    frame_id = 0
    fps_estimate = 0.0
    recent_alert_text = ""
    recent_alert_time = 0.0
    last_heartbeat_time = 0.0

    window_name = "IBVAP — Laptop Webcam Surveillance Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, frame_w * 2), min(720, frame_h * 2))

    t_prev = time.perf_counter()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Empty frame received from webcam, waiting...")
                    time.sleep(0.05)
                    continue

                frame_id += 1
                now = time.time()

                # Calculate smoothed FPS
                t_curr = time.perf_counter()
                dt = t_curr - t_prev
                t_prev = t_curr
                fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / dt) if dt > 0 else 0.0

                # ── Analytics Pipeline ────────────────────────────────────────
                # 1. Detection & Tracking
                tracks = tracker.update(
                    frame,
                    frame_id=frame_id,
                    camera_id="cam_usb_0",
                    timestamp=now,
                )

                # 2. Virtual Fence / Tripwires
                fence_events = fence.update(tracks, frame_id=frame_id)

                # 3. Behaviour Analytics (Speed & Loitering)
                behaviour_events = behaviour.update(tracks, frame_id=frame_id)

                # 4. Event Processing & Deduplication
                raw_events = fence_events + behaviour_events
                active_events = event_engine.process(raw_events, frame)

                # Log and dispatch alerts
                if active_events:
                    event_logger.log_many(active_events)
                    top_ev = max(active_events, key=lambda e: e.severity)
                    recent_alert_text = f"{top_ev.event_type.value} [{top_ev.severity.value}] - Track #{top_ev.track.track_id}"
                    recent_alert_time = time.monotonic()
                    fence_renderer.trigger_alert()

                    # Push to web dashboard
                    if not args.no_api:
                        for ev in active_events:
                            dispatch_event_to_api(ev)

                # Send periodic heartbeat every 2 seconds
                if not args.no_api and (time.monotonic() - last_heartbeat_time > 2.0):
                    send_heartbeat_to_api(
                        camera_id="cam_usb_0",
                        fps=fps_estimate,
                        frame_id=frame_id,
                    )
                    last_heartbeat_time = time.monotonic()

                # ── Rendering & Visual Overlays ───────────────────────────────
                vis_frame = frame.copy()

                # Draw fence zones and tripwires
                if show_fence:
                    vis_frame = fence_renderer.draw(vis_frame)

                # Draw tracked objects
                vis_frame = renderer.draw(vis_frame, tracks, frame_id=frame_id)

                # Draw Top Status Bar
                cv2.rectangle(vis_frame, (0, 0), (frame_w, 36), (15, 23, 42), -1)
                status_txt = f"CAM: LAPTOP WEBCAM 0 | FPS: {fps_estimate:4.1f} | TRACKS: {len(tracks)}"
                cv2.putText(vis_frame, status_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # Draw active alert banner if within last 3 seconds
                if time.monotonic() - recent_alert_time < 3.0:
                    cv2.rectangle(vis_frame, (0, 36), (frame_w, 70), (0, 0, 220), -1)
                    cv2.putText(
                        vis_frame,
                        f"ALERT: {recent_alert_text}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

            # Display frame
            cv2.imshow(window_name, vis_frame)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:  # q or ESC
                break
            elif key == ord("s"):
                snap_path = f"data/snapshots/webcam_manual_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, frame)
                logger.info(f"Manual snapshot saved: {snap_path}")
            elif key == ord("p"):
                paused = not paused
                logger.info("Paused" if paused else "Resumed")
            elif key == ord("f"):
                show_fence = not show_fence
                logger.info(f"Fence overlay: {'ON' if show_fence else 'OFF'}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam surveillance pipeline terminated cleanly.")


if __name__ == "__main__":
    main()
