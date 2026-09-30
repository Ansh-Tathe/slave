#!/usr/bin/env python3
"""
run_p1.py
=========
IBVAP P1 — End-to-end demo: ingest → detect → track → annotated output.

Usage examples
--------------
# Run on a local video file, show window:
python run_p1.py --source data/samples/general/BigBuckBunny_360p.mp4

# Run on a local file, write to output.mp4 (no window):
python run_p1.py --source path/to/video.mp4 --output output_p1.mp4 --no-display

# Run on RTSP stream:
python run_p1.py --source rtsp://admin:pass@192.168.1.100:554/stream1

# Use a smaller/faster model:
python run_p1.py --source video.mp4 --model yolov8n.pt

# Use CPU (no GPU):
python run_p1.py --source video.mp4 --device cpu --no-half

Key controls (when display window is open):
  q / ESC  — quit
  s        — save current annotated frame as screenshot
  p        — pause / resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# ── Make repo root importable regardless of CWD ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.detect_track.models   import DETECT_CLASS_IDS
from services.detect_track.renderer import Renderer
from services.detect_track.tracker  import ByteTracker


# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


# =============================================================================
# Helpers
# =============================================================================

def _open_source(source: str) -> cv2.VideoCapture:
    """Open a file path, RTSP URL, or integer device index."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
        if source.startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)
    return cap


def _make_writer(
    path:  str,
    fps:   float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.error(f"Cannot create output file: {path}")
        sys.exit(1)
    return writer


def _print_stats(
    n_frames: int,
    n_dropped: int,
    elapsed:   float,
    tracker:   ByteTracker,
) -> None:
    fps = n_frames / elapsed if elapsed > 0 else 0
    logger.info("─" * 55)
    logger.info(f"  Frames processed : {n_frames}")
    logger.info(f"  Frames dropped   : {n_dropped}")
    logger.info(f"  Elapsed          : {elapsed:.1f}s")
    logger.info(f"  Avg FPS          : {fps:.1f}")
    try:
        import torch
        free_b, total_b = torch.cuda.mem_get_info(0)
        used = (total_b - free_b) / 1024**3
        logger.info(f"  GPU VRAM used    : {used:.2f} GB")
    except Exception:
        pass
    logger.info("─" * 55)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IBVAP P1 — detection + tracking demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source",      required=True,
                        help="Video file, RTSP URL, or device index (0)")
    parser.add_argument("--model",       default="yolov8n.pt",
                        help="YOLO weights file (default: yolov8n.pt)")
    parser.add_argument("--tracker",     default="bytetrack.yaml",
                        choices=["bytetrack.yaml", "botsort.yaml"],
                        help="Tracker config (default: bytetrack.yaml)")
    parser.add_argument("--device",      default="cuda:0",
                        help="Inference device (default: cuda:0)")
    parser.add_argument("--no-half",     action="store_true",
                        help="Disable FP16 (use FP32)")
    parser.add_argument("--conf",        type=float, default=0.40,
                        help="Detection confidence threshold (default: 0.40)")
    parser.add_argument("--iou",         type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz",       type=int,   default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--target-fps",  type=float, default=10.0,
                        help="Target processing FPS — skip frames to match (default: 10)")
    parser.add_argument("--output",      default=None,
                        help="Save annotated video to this path (optional)")
    parser.add_argument("--no-display",  action="store_true",
                        help="Don't show live window (useful for headless servers)")
    parser.add_argument("--max-frames",  type=int, default=0,
                        help="Stop after N frames (0 = unlimited)")
    args = parser.parse_args()

    half = not args.no_half

    # ── Open source ──────────────────────────────────────────────────────────
    logger.info(f"Opening source: {args.source}")
    cap = _open_source(args.source)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Source: {width}x{height} @ {src_fps:.1f} FPS"
        + (f", {total} frames" if total > 0 else "")
    )

    # Frame-skip ratio to hit target FPS
    skip_n = max(1, round(src_fps / args.target_fps))
    logger.info(
        f"Processing every {skip_n} frame(s) "
        f"→ ~{src_fps/skip_n:.1f} processing FPS"
    )

    # ── Load model + tracker ─────────────────────────────────────────────────
    tracker  = ByteTracker(
        model_path  = args.model,
        tracker_cfg = args.tracker,
        device      = args.device,
        half        = half,
        conf        = args.conf,
        iou         = args.iou,
        imgsz       = args.imgsz,
    )
    renderer = Renderer(fps_window=30)

    # ── Optional video writer ────────────────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    if args.output:
        writer = _make_writer(
            args.output, args.target_fps, width, height
        )
        logger.info(f"Writing annotated output to: {args.output}")

    # ── Display window ───────────────────────────────────────────────────────
    WIN = "IBVAP P1 — detection + tracking  (q=quit, s=screenshot, p=pause)"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(width, 1280), min(height, 720))

    # ── Main loop ─────────────────────────────────────────────────────────────
    raw_count  = 0
    proc_count = 0
    dropped    = 0
    paused     = False
    t_start    = time.monotonic()

    logger.info("Starting — press 'q' or ESC in window to quit")

    while True:
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("p"), ord(" ")):
                paused = False
            elif key in (ord("q"), 27):
                break
            continue

        ok, frame = cap.read()
        if not ok:
            logger.info("End of stream / file.")
            break

        raw_count += 1

        # Frame-skip
        if raw_count % skip_n != 0:
            continue

        proc_count += 1

        # ── Inference ────────────────────────────────────────────────────────
        t_inf = time.monotonic()
        tracks = tracker.update(
            frame     = frame,
            frame_id  = proc_count,
            camera_id = "demo",
            timestamp = t_inf,
        )
        inf_ms = (time.monotonic() - t_inf) * 1000

        # ── Render ───────────────────────────────────────────────────────────
        annotated = renderer.draw(
            frame,
            tracks,
            frame_id  = proc_count,
            extra_info= f"inf {inf_ms:.0f}ms",
        )

        # ── Output ───────────────────────────────────────────────────────────
        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # q / ESC
                break
            elif key == ord("s"):              # screenshot
                ts   = int(time.time())
                path = f"screenshot_{ts}.jpg"
                cv2.imwrite(path, annotated)
                logger.info(f"Screenshot saved: {path}")
            elif key in (ord("p"), ord(" ")): # pause
                paused = True
                logger.info("Paused. Press p or space to resume.")

        # ── Per-frame log (every 30 processed frames) ─────────────────────
        if proc_count % 30 == 0:
            elapsed = time.monotonic() - t_start
            fps     = proc_count / elapsed
            logger.info(
                f"frame={proc_count:5d} | "
                f"fps={fps:5.1f} | "
                f"tracks={len(tracks):3d} | "
                f"inf={inf_ms:5.1f}ms"
            )

        # ── Max-frames limit ─────────────────────────────────────────────
        if args.max_frames and proc_count >= args.max_frames:
            logger.info(f"Reached max-frames limit ({args.max_frames}). Stopping.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    _print_stats(proc_count, dropped, elapsed, tracker)


if __name__ == "__main__":
    main()
