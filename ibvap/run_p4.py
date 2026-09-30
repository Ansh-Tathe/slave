#!/usr/bin/env python3
"""
run_p4.py
=========
IBVAP P4 -- End-to-end demo:
  ingest -> night-enhance -> detect/track -> event engine -> alerts

The pipeline runs NightEnhancer on every frame (gated by darkness detection).
A split-view window shows the ORIGINAL frame on the left and the ENHANCED
frame on the right, so the improvement is immediately visible.

Usage examples
--------------
# Run on a local video file (auto-detect dark frames):
python run_p4.py --source data/samples/general/test_clip_720p.mp4

# Force-enhance every frame (great for demo on daytime video):
python run_p4.py --source video.mp4 --always-enhance

# Use only CLAHE (faster, no gamma pass):
python run_p4.py --source video.mp4 --enhance-mode clahe

# Disable time-of-day gate (emit NIGHT_MOVEMENT at any hour):
python run_p4.py --source video.mp4 --always-enhance --no-hour-gate

# Save annotated output:
python run_p4.py --source video.mp4 --output output_p4.mp4 --no-display

Controls (when window is open):
  q / ESC  -- quit
  s        -- save screenshot
  p        -- pause / resume
  e        -- toggle enhancement on/off (live comparison)
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

from services.analytics.night.night_engine  import NightEngine
from services.detect_track.renderer         import Renderer
from services.detect_track.tracker          import ByteTracker
from services.event_engine.engine           import EventEngine
from services.event_engine.logger           import EventLogger
from services.event_engine.models           import Severity

# -- Logging -------------------------------------------------------------------
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


# =============================================================================
# Helpers
# =============================================================================

def _open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if source.startswith("rtsp"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)
    return cap


def _draw_luminance_bar(img: np.ndarray, lum: float, is_dark: bool) -> None:
    """Draw a coloured luminance indicator bar at the bottom of the image."""
    h, w = img.shape[:2]
    bar_h = 8
    # Background
    cv2.rectangle(img, (0, h - bar_h), (w, h), (30, 30, 30), cv2.FILLED)
    # Fill proportional to luminance (0-255 -> 0-w)
    fill_w = int(w * min(lum / 255.0, 1.0))
    color  = (0, 80, 255) if is_dark else (0, 200, 80)   # red-ish if dark, green if bright
    if fill_w > 0:
        cv2.rectangle(img, (0, h - bar_h), (fill_w, h), color, cv2.FILLED)


def _make_splitview(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """Stack original (left) and enhanced (right) side by side with divider."""
    h, w = original.shape[:2]
    out  = np.zeros((h, w * 2 + 2, 3), dtype=np.uint8)
    out[:, :w]         = original
    out[:, w:w+2]      = 128          # grey divider
    out[:, w+2:]       = enhanced

    # Labels
    for x, label in [(10, "ORIGINAL"), (w + 12, "ENHANCED (P4)")]:
        cv2.putText(out, label, (x, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IBVAP P4 -- night enhancement demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source",        required=True,
                        help="Video file, RTSP URL, or device index")
    parser.add_argument("--model",         default="yolov8n.pt")
    parser.add_argument("--device",        default="cuda:0")
    parser.add_argument("--no-half",       action="store_true")
    parser.add_argument("--conf",          type=float, default=0.35)
    parser.add_argument("--target-fps",    type=float, default=10.0)
    parser.add_argument("--enhance-mode",  default="combined",
                        choices=["clahe", "gamma", "combined", "dnn"],
                        help="Enhancement algorithm (default: combined)")
    parser.add_argument("--dark-threshold",type=float, default=80.0,
                        help="LAB-L mean below this -> dark frame (0-255, default 80)")
    parser.add_argument("--dnn-model",     default=None,
                        help="Path to Zero-DCE ONNX model (--enhance-mode dnn only)")
    parser.add_argument("--always-enhance",action="store_true",
                        help="Enhance every frame, skip darkness check")
    parser.add_argument("--no-hour-gate",  action="store_true",
                        help="Emit NIGHT_MOVEMENT events at any hour (not just night)")
    parser.add_argument("--dedup",         type=float, default=15.0,
                        help="Event dedup window in seconds (default 15)")
    parser.add_argument("--split-view",    action="store_true",
                        help="Show original + enhanced side by side")
    parser.add_argument("--output",        default=None)
    parser.add_argument("--no-display",    action="store_true")
    parser.add_argument("--max-frames",    type=int, default=0)
    args = parser.parse_args()

    # -- Open source -----------------------------------------------------------
    cap     = _open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip_n  = max(1, round(src_fps / args.target_fps))
    logger.info(f"Source: {width}x{height} @ {src_fps:.1f} FPS | skip={skip_n}")

    # -- Build pipeline --------------------------------------------------------
    tracker = ByteTracker(
        model_path=args.model,
        device=args.device,
        half=not args.no_half,
        conf=args.conf,
    )

    from services.analytics.night.night_engine import _DEFAULT_NIGHT_HOURS
    night_engine = NightEngine(
        enhancer_mode  = args.enhance_mode,
        dark_threshold = args.dark_threshold,
        dnn_model_path = args.dnn_model,
        night_hours    = None if args.no_hour_gate else _DEFAULT_NIGHT_HOURS,
        dedup_window_s = args.dedup,
        event_severity = Severity.HIGH,
        camera_id      = "demo",
        always_enhance = args.always_enhance,
    )

    ev_engine = EventEngine(dedup_window_s=args.dedup, snapshot_dir="data/snapshots/night")
    ev_logger  = EventLogger(log_path="data/logs/night_events.jsonl")
    renderer   = Renderer(fps_window=30)

    # -- Optional writer -------------------------------------------------------
    writer = None
    out_w  = width * 2 + 2 if args.split_view else width
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.target_fps, (out_w, height))
        logger.info(f"Writing to: {args.output}")

    # -- Window ----------------------------------------------------------------
    WIN = "IBVAP P4 — night enhancement  (q=quit, s=screenshot, p=pause, e=toggle)"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(out_w, 1600), min(height, 720))

    # -- Main loop -------------------------------------------------------------
    raw_count    = 0
    proc_count   = 0
    total_events = 0
    enhanced_on  = True      # toggle with 'e'
    paused       = False
    t_start      = time.monotonic()

    logger.info("Running P4 pipeline — press q / ESC to quit")

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

        # -- Night enhance + events --------------------------------------------
        lum     = night_engine.enhancer.mean_luminance(frame)
        is_dark = night_engine.enhancer.is_dark(frame)

        if enhanced_on:
            bright, night_events = night_engine.update(
                frame, [], frame_id=proc_count, camera_id="demo"
            )
        else:
            bright, night_events = frame, []

        # -- Detect + track (on the enhanced frame for better accuracy) --------
        tracks = tracker.update(
            frame=bright, frame_id=proc_count,
            camera_id="demo", timestamp=time.monotonic(),
        )

        # If engine was called with empty tracks above, re-emit with real tracks
        if enhanced_on and tracks:
            _, night_events = night_engine.update(
                frame, tracks, frame_id=proc_count, camera_id="demo"
            )

        # -- Event engine (dedup + snapshots) ----------------------------------
        fired = ev_engine.process(night_events, frame)
        ev_logger.log_many(fired)
        total_events += len(fired)

        # -- Render ------------------------------------------------------------
        t_inf   = time.monotonic()
        inf_ms  = (t_inf - t_start) * 0  # placeholder

        annotated = renderer.draw(
            bright if enhanced_on else frame,
            tracks,
            frame_id=proc_count,
            extra_info=f"lum={lum:.0f}{'*DARK*' if is_dark else ''}  events={total_events}",
        )

        _draw_luminance_bar(annotated, lum, is_dark)

        if args.split_view:
            display_frame = _make_splitview(frame, annotated)
        else:
            display_frame = annotated

        if writer:
            writer.write(display_frame)

        if not args.no_display:
            cv2.imshow(WIN, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                ts   = int(time.time())
                path = f"screenshot_p4_{ts}.jpg"
                cv2.imwrite(path, display_frame)
                logger.info(f"Screenshot: {path}")
            elif key in (ord("p"), ord(" ")):
                paused = True
            elif key == ord("e"):
                enhanced_on = not enhanced_on
                logger.info(f"Enhancement {'ON' if enhanced_on else 'OFF'}")

        if proc_count % 30 == 0:
            elapsed = time.monotonic() - t_start
            logger.info(
                f"frame={proc_count:5d} | "
                f"fps={proc_count/elapsed:5.1f} | "
                f"tracks={len(tracks):3d} | "
                f"lum={lum:5.1f} | "
                f"dark={'Y' if is_dark else 'N'} | "
                f"events={total_events}"
            )

        if args.max_frames and proc_count >= args.max_frames:
            break

    # -- Cleanup ---------------------------------------------------------------
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    ev_logger.print_summary()

    elapsed = time.monotonic() - t_start
    stats   = night_engine.stats
    logger.info(
        f"Done — {proc_count} frames in {elapsed:.1f}s "
        f"({proc_count/elapsed:.1f} FPS avg)"
    )
    logger.info(
        f"Enhanced: {stats['enhanced_frames']} frames | "
        f"Skipped: {stats['skipped_frames']} frames | "
        f"Events fired: {total_events}"
    )


if __name__ == "__main__":
    main()
