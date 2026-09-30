#!/usr/bin/env python3
"""
run_p5.py
=========
IBVAP P5 -- Full pipeline demo:
  ingest -> detect/track -> fence (P2) -> behaviour (P5) -> event engine -> alerts

Suspicious-activity rules active:
  LOITERING      -- person stationary for > --loiter-s seconds
  RUNNING        -- person speed > --run-speed px/s
  GATHERING      -- >= --gather-n persons within --gather-px for > --gather-dur s
  ABANDONED_OBJ  -- vehicle/object stationary for > --abandoned-s without owner
  WRONG_WAY      -- vehicle moving against --allowed-dir (if set)

Usage
-----
  python run_p5.py --source data/samples/general/test_clip_720p.mp4
  python run_p5.py --source video.mp4 --loiter-s 10 --run-speed 80 --gather-n 3
  python run_p5.py --source video.mp4 --no-display --output output_p5.mp4

Controls: q/ESC=quit  s=screenshot  p=pause
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.analytics.behaviour.behaviour_engine import BehaviourEngine
from services.analytics.behaviour.speed            import SpeedDetector
from services.detect_track.models                  import DETECT_CLASS_IDS
from services.detect_track.renderer                import Renderer
from services.detect_track.tracker                 import ByteTracker
from services.event_engine.engine                  import EventEngine
from services.event_engine.logger                  import EventLogger
from services.event_engine.models                  import Severity

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


# =============================================================================
# Behaviour overlay renderer
# =============================================================================

def _draw_behaviour_overlay(
    img:     np.ndarray,
    track,
    speed:   float,
    is_run:  bool,
) -> None:
    """Draw speed label and running indicator above the track box."""
    if speed < 5.0:
        return
    x1, y1, x2, y2 = (int(v) for v in track.bbox)
    color = (0, 0, 255) if is_run else (0, 180, 255)
    label = f"{speed:.0f}px/s {'RUN!' if is_run else ''}"
    cv2.putText(img, label, (x1, max(0, y1 - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="IBVAP P5 -- behaviour analytics demo")
    p.add_argument("--source",       required=True)
    p.add_argument("--model",        default="yolov8n.pt")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--no-half",      action="store_true")
    p.add_argument("--conf",         type=float, default=0.35)
    p.add_argument("--target-fps",   type=float, default=10.0)
    # Loitering
    p.add_argument("--loiter-s",     type=float, default=15.0,
                   help="Loitering dwell threshold seconds (default 15 for demo)")
    p.add_argument("--loiter-px",    type=float, default=40.0,
                   help="Loitering movement reset threshold px (default 40)")
    # Running
    p.add_argument("--run-speed",    type=float, default=120.0,
                   help="Running speed threshold px/s (default 120)")
    p.add_argument("--allowed-dir",  default="any",
                   choices=["any","left","right","up","down"],
                   help="Allowed vehicle direction (wrong-way detection)")
    # Gathering
    p.add_argument("--gather-n",     type=int,   default=3,
                   help="Minimum persons for GATHERING (default 3 for demo)")
    p.add_argument("--gather-px",    type=float, default=200.0,
                   help="Gathering proximity radius px (default 200)")
    p.add_argument("--gather-dur",   type=float, default=8.0,
                   help="Gathering duration seconds (default 8 for demo)")
    # Abandoned
    p.add_argument("--abandoned-s",  type=float, default=30.0,
                   help="Abandoned object stationary seconds (default 30 for demo)")
    p.add_argument("--abandoned-px", type=float, default=150.0,
                   help="Abandoned owner radius px (default 150)")
    # Output
    p.add_argument("--output",       default=None)
    p.add_argument("--no-display",   action="store_true")
    p.add_argument("--max-frames",   type=int, default=0)
    args = p.parse_args()

    # -- Open source -----------------------------------------------------------
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        logger.error(f"Cannot open: {args.source}"); sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip_n  = max(1, round(src_fps / args.target_fps))
    logger.info(f"Source: {width}x{height} @ {src_fps:.1f} FPS | skip={skip_n}")

    # -- Build pipeline --------------------------------------------------------
    tracker  = ByteTracker(
        model_path=args.model, device=args.device,
        half=not args.no_half, conf=args.conf,
    )
    behav = BehaviourEngine(
        camera_id              = "demo",
        loiter_dwell_s         = args.loiter_s,
        loiter_move_px         = args.loiter_px,
        run_threshold_px_per_s = args.run_speed,
        allowed_direction      = args.allowed_dir,
        gather_min_count       = args.gather_n,
        gather_proximity_px    = args.gather_px,
        gather_duration_s      = args.gather_dur,
        abandoned_stationary_s = args.abandoned_s,
        abandoned_owner_radius_px = args.abandoned_px,
    )
    ev_engine = EventEngine(dedup_window_s=10.0, snapshot_dir="data/snapshots/behaviour")
    ev_logger  = EventLogger(log_path="data/logs/behaviour_events.jsonl")
    renderer   = Renderer(fps_window=30)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.target_fps, (width, height))
        logger.info(f"Writing to: {args.output}")

    WIN = "IBVAP P5 -- behaviour analytics  (q=quit, s=screenshot, p=pause)"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(width, 1280), min(height, 720))

    raw_count    = 0
    proc_count   = 0
    total_events = 0
    paused       = False
    t_start      = time.monotonic()
    logger.info("Running P5 pipeline -- press q / ESC to quit")

    while True:
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("p"), ord(" ")): paused = False
            elif key in (ord("q"), 27):     break
            continue

        ok, frame = cap.read()
        if not ok:
            logger.info("Stream ended."); break

        raw_count += 1
        if raw_count % skip_n != 0:
            continue
        proc_count += 1

        # -- Detect + track ----------------------------------------------------
        tracks = tracker.update(
            frame=frame, frame_id=proc_count,
            camera_id="demo", timestamp=time.monotonic(),
        )

        # -- Behaviour analytics -----------------------------------------------
        bev_events = behav.update(tracks, proc_count, "demo")
        fired      = ev_engine.process(bev_events, frame)
        ev_logger.log_many(fired)
        total_events += len(fired)

        # -- Render ------------------------------------------------------------
        annotated = renderer.draw(
            frame, tracks, frame_id=proc_count,
            extra_info=f"events={total_events}",
        )

        # Speed overlay per track
        if behav._speed:
            run_thr = behav._speed._run_thr
            for track in tracks:
                spd    = behav._speed.get_speed(track.track_id)
                is_run = spd >= run_thr and track.class_name == "person"
                _draw_behaviour_overlay(annotated, track, spd, is_run)

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break
            elif key == ord("s"):
                ts   = int(time.time())
                path = f"screenshot_p5_{ts}.jpg"
                cv2.imwrite(path, annotated)
                logger.info(f"Screenshot: {path}")
            elif key in (ord("p"), ord(" ")):
                paused = True

        if proc_count % 30 == 0:
            elapsed = time.monotonic() - t_start
            logger.info(
                f"frame={proc_count:5d} | fps={proc_count/elapsed:5.1f} | "
                f"tracks={len(tracks):3d} | events_total={total_events}"
            )

        if args.max_frames and proc_count >= args.max_frames:
            break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    ev_logger.print_summary()
    elapsed = time.monotonic() - t_start
    logger.info(f"Done -- {proc_count} frames in {elapsed:.1f}s ({proc_count/elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
