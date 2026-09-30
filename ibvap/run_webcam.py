#!/usr/bin/env python3
"""
run_webcam.py
=============
IBVAP — Real-Time Laptop Webcam Surveillance Pipeline with Weapon Detection
and Interactive Custom Restriction Zones.

Features:
  1. Live video capture from webcam (DirectShow on Windows / Video4Linux on Linux).
  2. YOLOv8n object detection & ByteTrack persistent identity tracking.
  3. Harmful Object / Weapon Detection:
     - Detects firearms (Guns/Pistols/Rifles) and melee weapons (Knives/Blades).
     - Automatically verifies if the weapon is carried/contained by a tracked person.
     - Emits CRITICAL ARMED_PERSON or HIGH WEAPON_DETECTED alerts with red visual warnings.
  4. Dynamic Customizable Restriction Zone:
     - Mouse drag-and-drop: Press 'z' to enter Zone Edit Mode and drag a new zone on screen.
     - Number keys [1..4] for instant presets (Left, Right, Center, Full Perimeter).
     - CLI `--zone` argument (e.g. --zone "0.1,0.2,0.6,0.8" or --zone "center").
  5. Behaviour analytics (Loitering & Speed detection).
  6. Live integration with IBVAP Web Dashboard (dispatches alerts to http://localhost:8000).
  7. Real-time visual overlay with event banners and alert flash.

Usage:
  python run_webcam.py
  python run_webcam.py --camera-index 0 --zone center
  python run_webcam.py --zone "0.2,0.2,0.8,0.8" --weapon-conf 0.35
  python run_webcam.py --no-api

Controls:
  q / ESC : Exit
  z       : Toggle Interactive Mouse Zone Edit Mode (click & drag on video)
  1..4    : Instant Zone presets (1:Left, 2:Right, 3:Center, 4:Full)
  f       : Toggle Virtual Fence overlay
  w       : Toggle Weapon Detector
  s       : Save high-res screenshot
  p       : Pause / resume
"""

from __future__ import annotations

import argparse
import platform
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import httpx
import numpy as np
import yaml
from loguru import logger

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.analytics.behaviour.behaviour_engine import BehaviourEngine
from services.analytics.fence.fence_engine import FenceEngine
from services.analytics.fence.tripwire import TripwireConfig
from services.analytics.fence.zone_checker import ZoneConfig
from services.analytics.fence.zone_renderer import ZoneRenderer
from services.analytics.weapon.weapon_detector import WeaponDetector
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


def send_frame_to_api(
    frame: np.ndarray,
    camera_id: str = "cam_usb_0",
    api_url: str = "http://localhost:8000/api/v1/cameras",
) -> None:
    """Send compressed JPEG frame asynchronously to local FastAPI streaming buffer."""
    def _worker():
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with httpx.Client(timeout=0.6) as client:
                client.post(
                    f"{api_url}/{camera_id}/frame",
                    content=buf.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                )
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def create_zone_polygon(x1: float, y1: float, x2: float, y2: float) -> List[Tuple[float, float]]:
    """Helper to convert rectangular bounds into 4 clockwise polygon points."""
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def parse_zone_arg(zone_str: str) -> List[Tuple[float, float]]:
    """Parse zone preset name or comma-separated coords: 'x1,y1,x2,y2'."""
    z = zone_str.strip().lower()
    if z == "left":
        return create_zone_polygon(0.05, 0.15, 0.48, 0.85)
    elif z == "right":
        return create_zone_polygon(0.52, 0.15, 0.95, 0.85)
    elif z == "center":
        return create_zone_polygon(0.25, 0.20, 0.75, 0.80)
    elif z in ("full", "perimeter"):
        return create_zone_polygon(0.05, 0.05, 0.95, 0.95)

    try:
        parts = [float(v.strip()) for v in zone_str.split(",")]
        if len(parts) == 4:
            return create_zone_polygon(parts[0], parts[1], parts[2], parts[3])
    except Exception as e:
        logger.warning(f"Could not parse zone string '{zone_str}': {e}. Using default left zone.")

    return create_zone_polygon(0.05, 0.15, 0.45, 0.85)


def save_zone_to_config(polygon: List[Tuple[float, float]], config_path: str = "configs/zones.yaml") -> None:
    """Persist new restricted zone polygon to configs/zones.yaml under cam_usb_0_zones."""
    p = Path(config_path)
    if not p.exists():
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if "zones" not in cfg:
            cfg["zones"] = {}

        if "cam_usb_0_zones" not in cfg["zones"]:
            cfg["zones"]["cam_usb_0_zones"] = {"polygons": [], "tripwires": []}

        polys = cfg["zones"]["cam_usb_0_zones"].get("polygons", [])
        updated = False
        for poly in polys:
            if poly.get("id") in ("restricted_left", "restricted_custom"):
                poly["points"] = [[round(pt[0], 3), round(pt[1], 3)] for pt in polygon]
                updated = True
                break

        if not updated:
            polys.append({
                "id": "restricted_custom",
                "name": "Custom Restricted Zone",
                "enabled": True,
                "points": [[round(pt[0], 3), round(pt[1], 3)] for pt in polygon],
                "triggers": {"on_enter": True, "dwell_seconds": 3},
                "severity": "HIGH",
            })
            cfg["zones"]["cam_usb_0_zones"]["polygons"] = polys

        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        logger.info(f"Custom restriction zone saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save zone config: {e}")


def main():
    parser = argparse.ArgumentParser(description="IBVAP Live Laptop Webcam Surveillance Pipeline")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold (default: 0.45)")
    parser.add_argument("--target-fps", type=float, default=15.0, help="Target processing FPS (default: 15.0)")
    parser.add_argument("--no-api", action="store_true", help="Disable dispatching events to web dashboard")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--threat-model", default="models/threat_yolov8n.pt", help="Threat model path")
    parser.add_argument("--weapon-conf", type=float, default=0.20, help="Weapon confidence threshold (default: 0.20)")
    parser.add_argument("--no-weapons", action="store_true", help="Disable weapon detector")
    parser.add_argument("--zone", default="left", help="Restriction zone: 'left','right','center','full' or 'x1,y1,x2,y2'")
    parser.add_argument("--fullscreen", action="store_true", default=True, help="Open camera feed in full screen mode (default: True)")
    parser.add_argument("--windowed", dest="fullscreen", action="store_false", help="Open in standard windowed mode")
    args = parser.parse_args()

    print("=" * 76)
    print(" IBVAP — Real-Time Surveillance Console (Weapons & Dynamic Zones)")
    print("=" * 76)
    print(f" * Webcam Device Index   : {args.camera_index}")
    print(f" * YOLO Person/Vehicle   : {args.model} (conf: {args.conf})")
    print(f" * Threat/Weapon Engine  : {'Disabled' if args.no_weapons else args.threat_model + f' (conf: {args.weapon_conf})'}")
    print(f" * Initial Zone Setting  : {args.zone}")
    print(f" * Fullscreen Mode       : {'Active (F11/F to toggle)' if args.fullscreen else 'Windowed'}")
    print(f" * Dashboard Integration : {'Disabled' if args.no_api else 'Active (http://localhost:8000)'}")
    print(" * Hotkeys:")
    print("     [f] / [F11] : Toggle Fullscreen")
    print("     [z]         : Toggle Interactive Mouse Zone Editor (drag on video feed)")
    print("     [1..4]      : Zone Left | Right | Center | Full")
    print("     [w]         : Toggle Weapons | [v] : Toggle Fence | [s] : Snapshot | [q] : Quit")
    print("=" * 76)

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

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    logger.success(f"Webcam active: {frame_w}x{frame_h} pixels")

    device = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    # 2. Initialize Analytics Subsystems
    logger.info("Loading YOLOv8 + ByteTrack tracking engine...")
    tracker = ByteTracker(
        model_path=args.model,
        conf=args.conf,
        device=device,
        warmup_runs=1,
    )

    # Threat & Weapon Detector
    enable_weapons = not args.no_weapons
    weapon_detector = None
    if enable_weapons:
        weapon_detector = WeaponDetector(
            model_path=args.threat_model,
            conf_thresh=args.weapon_conf,
            device=device,
            camera_id="cam_usb_0",
        )

    # Masked Face Detector
    mask_detector = None
    mask_p = Path("models/mask_yolov8n.pt")
    if mask_p.exists():
        try:
            from ultralytics import YOLO
            mask_detector = YOLO(str(mask_p))
            logger.info("Masked individual detector loaded (models/mask_yolov8n.pt)")
        except Exception as e:
            logger.warning(f"Could not load mask detector: {e}")

    # Restriction Zone & Tripwires
    current_zone_polygon = parse_zone_arg(args.zone)
    zone_cfgs = [
        ZoneConfig(
            zone_id="restricted_zone",
            name="Restricted Zone",
            polygon=current_zone_polygon,
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
            start=(0.50, 0.1),
            end=(0.50, 0.9),
            direction="both",
            classes=["person"],
            severity=Severity.CRITICAL,
        )
    ]

    fence = FenceEngine(zone_cfgs, wire_cfgs, frame_w=frame_w, frame_h=frame_h, camera_id="cam_usb_0")
    fence_renderer = ZoneRenderer(fence, alpha=0.25)
    renderer = Renderer()
    behaviour = BehaviourEngine(camera_id="cam_usb_0")
    event_engine = EventEngine(dedup_window_s=4.0, snapshot_dir="data/snapshots")
    event_logger = EventLogger(log_path="data/logs/events.jsonl")

    # Interactive Zone Dragging State
    zone_edit_mode = False
    drag_start: Optional[Tuple[int, int]] = None
    drag_current: Optional[Tuple[int, int]] = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal zone_edit_mode, drag_start, drag_current, current_zone_polygon, fence, fence_renderer
        if not zone_edit_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            x1, y1 = drag_start
            x2, y2 = x, y
            drag_start = None
            drag_current = None

            # Enforce minimum dimension of 20 pixels
            if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                nx1 = max(0.0, min(1.0, min(x1, x2) / frame_w))
                ny1 = max(0.0, min(1.0, min(y1, y2) / frame_h))
                nx2 = max(0.0, min(1.0, max(x1, x2) / frame_w))
                ny2 = max(0.0, min(1.0, max(y1, y2) / frame_h))

                current_zone_polygon = create_zone_polygon(nx1, ny1, nx2, ny2)
                zone_cfgs[0].polygon = current_zone_polygon
                fence = FenceEngine(zone_cfgs, wire_cfgs, frame_w=frame_w, frame_h=frame_h, camera_id="cam_usb_0")
                fence_renderer = ZoneRenderer(fence, alpha=0.25)
                save_zone_to_config(current_zone_polygon)
                zone_edit_mode = False
                logger.success(f"New custom restriction zone set: [{nx1:.2f}, {ny1:.2f}, {nx2:.2f}, {ny2:.2f}]")

    def apply_preset_zone(preset_name: str):
        nonlocal current_zone_polygon, fence, fence_renderer
        current_zone_polygon = parse_zone_arg(preset_name)
        zone_cfgs[0].polygon = current_zone_polygon
        fence = FenceEngine(zone_cfgs, wire_cfgs, frame_w=frame_w, frame_h=frame_h, camera_id="cam_usb_0")
        fence_renderer = ZoneRenderer(fence, alpha=0.25)
        save_zone_to_config(current_zone_polygon)
        logger.info(f"Applied zone preset: {preset_name.upper()}")

    # Window setup
    window_name = "IBVAP — Laptop Webcam Surveillance Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = bool(args.fullscreen)
    if is_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, min(1280, frame_w * 2), min(720, frame_h * 2))
    cv2.setMouseCallback(window_name, mouse_callback)

    # Runtime state
    show_fence = True
    paused = False
    frame_id = 0
    fps_estimate = 0.0
    recent_alert_text = ""
    recent_alert_time = 0.0
    last_heartbeat_time = 0.0

    t_prev = time.perf_counter()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                frame_id += 1
                now = time.time()

                # FPS tracking
                t_now = time.perf_counter()
                dt = t_now - t_prev
                t_prev = t_now
                if dt > 0:
                    fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / dt)

                # 1. Detection + Tracking
                tracks = tracker.update(frame, frame_id=frame_id, camera_id="cam_usb_0", timestamp=now)

                # 2. Virtual Fence / Tripwires
                fence_events = fence.update(tracks, frame_id=frame_id)

                # 3. Behaviour Analytics
                behaviour_events = behaviour.update(tracks, frame_id=frame_id)

                # 4. Weapon / Threat Detection
                weapon_detections = []
                weapon_events = []
                if enable_weapons and weapon_detector is not None:
                    weapon_detections, weapon_events = weapon_detector.update(frame, tracks, frame_id=frame_id)

                # 5. Event Engine Processing & Deduplication
                raw_events = fence_events + behaviour_events + weapon_events
                active_events = event_engine.process(raw_events, frame)

                if active_events:
                    event_logger.log_many(active_events)
                    top_ev = max(active_events, key=lambda e: e.severity)
                    target_label = f"Track #{top_ev.track.track_id}" if top_ev.track else "Perimeter"
                    weapon_tag = f" ({top_ev.metadata.get('weapon', '').upper()})" if "weapon" in top_ev.metadata else ""
                    recent_alert_text = f"{top_ev.event_type.value}{weapon_tag} [{top_ev.severity.value}] - {target_label}"
                    recent_alert_time = time.monotonic()
                    fence_renderer.trigger_alert()

                    # Push to web dashboard
                    if not args.no_api:
                        for ev in active_events:
                            dispatch_event_to_api(ev)

                # Mask Detection on tracked persons
                masked_track_ids = set()
                if mask_detector is not None:
                    for t in tracks:
                        if t.class_name == "person":
                            x1, y1, x2, y2 = [int(v) for v in t.bbox]
                            head_crop = frame[max(0, y1):min(frame_h, y1 + int((y2 - y1) * 0.45)), max(0, x1):min(frame_w, x2)]
                            if head_crop.size > 0:
                                try:
                                    rm = mask_detector(head_crop, conf=0.15, verbose=False)[0]
                                    for mb in rm.boxes:
                                        m_cls = rm.names[int(mb.cls[0])]
                                        if "bermasker" in m_cls.lower() and "tidak" not in m_cls.lower():
                                            masked_track_ids.add(t.track_id)
                                            break
                                except Exception:
                                    pass

                # Periodic heartbeat to dashboard
                if not args.no_api and (time.monotonic() - last_heartbeat_time > 2.0):
                    send_heartbeat_to_api(camera_id="cam_usb_0", fps=fps_estimate, frame_id=frame_id)
                    last_heartbeat_time = time.monotonic()

                # ── Rendering & Visual Overlays ───────────────────────────────
                vis_frame = frame.copy()

                # Draw fence zones and tripwires
                if show_fence:
                    vis_frame = fence_renderer.draw(vis_frame)

                # Draw tracked objects
                vis_frame = renderer.draw(vis_frame, tracks, frame_id=frame_id)

                # Draw Mask Badges
                for t in tracks:
                    if t.track_id in masked_track_ids:
                        tx1, ty1, tx2, ty2 = [int(v) for v in t.bbox]
                        cv2.rectangle(vis_frame, (tx1, max(0, ty1 - 22)), (tx1 + 95, ty1), (0, 180, 255), -1)
                        cv2.putText(vis_frame, "MASKED", (tx1 + 5, max(15, ty1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)

                # Draw weapon / firearm detections
                if enable_weapons and weapon_detector is not None and weapon_detections:
                    vis_frame = weapon_detector.draw(vis_frame, weapon_detections)

                # Draw Live Drag Selection Rectangle if in Zone Edit Mode
                if zone_edit_mode and drag_start is not None and drag_current is not None:
                    px1, py1 = drag_start
                    px2, py2 = drag_current
                    cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    cv2.putText(
                        vis_frame,
                        "RELEASE TO SET RESTRICTED ZONE",
                        (min(px1, px2), max(20, min(py1, py2) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

                # Top Status HUD
                cv2.rectangle(vis_frame, (0, 0), (frame_w, 36), (15, 23, 42), -1)
                weap_status = "ON" if enable_weapons else "OFF"
                status_txt = f"CAM: 0 | FPS: {fps_estimate:4.1f} | TRACKS: {len(tracks)} | WEAPONS: {weap_status}"
                cv2.putText(vis_frame, status_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

                # Zone Edit Mode Notification Bar
                if zone_edit_mode:
                    cv2.rectangle(vis_frame, (0, 36), (frame_w, 68), (0, 180, 255), -1)
                    cv2.putText(
                        vis_frame,
                        "ZONE EDIT ACTIVE: Drag mouse on video to set Restriction Zone | [z] Cancel",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 0, 0),
                        2,
                    )
                # Active Alert Banner (Red Flash)
                elif time.monotonic() - recent_alert_time < 3.5:
                    alert_color = (0, 0, 220) if "ARMED" not in recent_alert_text else (0, 0, 255)
                    cv2.rectangle(vis_frame, (0, 36), (frame_w, 72), alert_color, -1)
                    cv2.putText(
                        vis_frame,
                        f"ALERT: {recent_alert_text}",
                        (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        2,
                    )

            # Stream frame to web dashboard (every 2 frames ~15 FPS)
            if not args.no_api and (frame_id % 2 == 0):
                send_frame_to_api(vis_frame)

            # Display frame
            cv2.imshow(window_name, vis_frame)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:  # q or ESC
                break
            elif key == ord("z"):
                zone_edit_mode = not zone_edit_mode
                drag_start = None
                drag_current = None
                logger.info(f"Zone Edit Mode: {'ACTIVATED (drag with mouse)' if zone_edit_mode else 'DEACTIVATED'}")
            elif key == ord("1"):
                apply_preset_zone("left")
            elif key == ord("2"):
                apply_preset_zone("right")
            elif key == ord("3"):
                apply_preset_zone("center")
            elif key == ord("4"):
                apply_preset_zone("full")
            elif key == ord("w"):
                enable_weapons = not enable_weapons
                logger.info(f"Threat & Weapon detection: {'ON' if enable_weapons else 'OFF'}")
            elif key == ord("s"):
                snap_path = f"data/snapshots/webcam_manual_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, frame)
                logger.info(f"Manual snapshot saved: {snap_path}")
            elif key == ord("p"):
                paused = not paused
                logger.info("Paused" if paused else "Resumed")
            elif key in [ord("f"), ord("F")]:
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                if not is_fullscreen:
                    cv2.resizeWindow(window_name, min(1280, frame_w * 2), min(720, frame_h * 2))
                logger.info(f"Fullscreen: {'ON' if is_fullscreen else 'OFF'}")
            elif key == ord("v"):
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
