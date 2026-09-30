#!/usr/bin/env python3
"""
run_p6.py
=========
IBVAP P6 -- Face detection + watchlist demo.

Pipeline:
  ingest -> detect/track -> face_engine (detect+embed+match) -> events -> display

Watchlist management
--------------------
  Add persons at startup via --watchlist-add:
    python run_p6.py --source video.mp4 --watchlist-add "John Doe:photo.jpg"

  Load a saved watchlist .npz:
    python run_p6.py --source video.mp4 --watchlist-file data/watchlist/wl.npz

  Save updated watchlist:
    python run_p6.py --source video.mp4 --save-watchlist data/watchlist/wl.npz

Display overlay
---------------
  - Green box + "UNKNOWN" for detected faces not on watchlist
  - RED box + name + similarity for watchlist hits

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

from services.analytics.face.face_engine    import FaceEngine
from services.analytics.face.face_detector  import FaceDetection
from services.detect_track.renderer         import Renderer
from services.detect_track.tracker          import ByteTracker
from services.event_engine.engine           import EventEngine
from services.event_engine.logger           import EventLogger
from services.event_engine.models           import Severity

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


# =============================================================================
# Face overlay renderer
# =============================================================================

def _draw_faces(
    img:            np.ndarray,
    faces:          list,
    match_results:  dict,           # center_bucket -> (name, sim) or None
) -> None:
    """Draw face boxes with match info overlay."""
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        cx, cy = int(face.center[0]), int(face.center[1])

        match = match_results.get((cx // 60, cy // 60))
        if match:
            name, sim = match
            color = (0, 0, 220)     # red -- watchlist hit
            label = f"{name}  {sim:.2f}"
        else:
            color = (0, 200, 60)    # green -- unknown face
            label = "UNKNOWN"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x2, y1), color, cv2.FILLED)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Landmark dots
        if face.landmarks is not None:
            for lx, ly in face.landmarks.astype(int):
                cv2.circle(img, (lx, ly), 2, (255, 200, 0), -1)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="IBVAP P6 -- face detection + watchlist demo")
    p.add_argument("--source",          required=True)
    p.add_argument("--model",           default="yolov8n.pt")
    p.add_argument("--device",          default="cuda:0")
    p.add_argument("--no-half",         action="store_true")
    p.add_argument("--conf",            type=float, default=0.35)
    p.add_argument("--target-fps",      type=float, default=8.0)
    # Face
    p.add_argument("--face-backend",    default="insightface",
                   choices=["insightface","yolo"])
    p.add_argument("--face-model",      default=None,
                   help="YOLO face model path (--face-backend yolo)")
    p.add_argument("--face-conf",       type=float, default=0.60)
    p.add_argument("--face-min-area",   type=int,   default=900)
    p.add_argument("--face-sharpness",  type=float, default=20.0)
    # Watchlist
    p.add_argument("--watchlist-file",  default=None,
                   help="Load watchlist .npz")
    p.add_argument("--watchlist-add",   nargs="*", default=[],
                   help="Add persons: 'Name:image.jpg' ...")
    p.add_argument("--match-threshold", type=float, default=0.45)
    p.add_argument("--save-watchlist",  default=None)
    # Output
    p.add_argument("--output",          default=None)
    p.add_argument("--no-display",      action="store_true")
    p.add_argument("--max-frames",      type=int, default=0)
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
    tracker = ByteTracker(
        model_path=args.model, device=args.device,
        half=not args.no_half, conf=args.conf,
    )
    face_engine = FaceEngine(
        watchlist_path       = args.watchlist_file,
        face_backend         = args.face_backend,
        yolo_face_model_path = args.face_model,
        device               = args.device,
        min_det_conf         = args.face_conf,
        min_area_px2         = args.face_min_area,
        min_sharpness        = args.face_sharpness,
        match_threshold      = args.match_threshold,
        watchlist_severity   = Severity.CRITICAL,
        camera_id            = "demo",
    )

    # Add persons from CLI
    for entry in args.watchlist_add:
        parts = entry.split(":", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid --watchlist-add entry: '{entry}' (use 'Name:image.jpg')")
            continue
        name, img_path = parts
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue
        ok = face_engine.add_to_watchlist(name, img)
        logger.info(f"Watchlist: {'added' if ok else 'FAILED'} '{name}' from {img_path}")

    ev_engine = EventEngine(dedup_window_s=30.0, snapshot_dir="data/snapshots/face")
    ev_logger  = EventLogger(log_path="data/logs/face_events.jsonl")
    renderer   = Renderer(fps_window=30)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.target_fps, (width, height))

    WIN = "IBVAP P6 -- face detection + watchlist  (q=quit, s=screenshot)"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(width, 1280), min(height, 720))

    raw_count    = 0
    proc_count   = 0
    total_events = 0
    paused       = False
    t_start      = time.monotonic()
    logger.info("Running P6 pipeline -- press q / ESC to quit")

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

        # -- Detect + track (persons) ------------------------------------------
        tracks = tracker.update(
            frame=frame, frame_id=proc_count,
            camera_id="demo", timestamp=time.monotonic(),
        )

        # -- Face detection + watchlist matching --------------------------------
        face_events, face_dets = face_engine.update(
            frame, tracks=tracks, frame_id=proc_count, camera_id="demo"
        )
        fired = ev_engine.process(face_events, frame)
        ev_logger.log_many(fired)
        total_events += len(fired)

        # -- Render ------------------------------------------------------------
        annotated = renderer.draw(
            frame, tracks, frame_id=proc_count,
            extra_info=f"faces={len(face_dets)}  events={total_events}",
        )

        # Build match_results for face overlay
        match_results = {}
        for evt in face_events:
            if evt.metadata.get("name"):
                cx = int(evt.track.center[0])
                cy = int(evt.track.center[1])
                match_results[(cx // 60, cy // 60)] = (
                    evt.metadata["name"], evt.metadata["similarity"]
                )

        _draw_faces(annotated, face_dets, match_results)

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break
            elif key == ord("s"):
                ts   = int(time.time())
                cv2.imwrite(f"screenshot_p6_{ts}.jpg", annotated)
            elif key in (ord("p"), ord(" ")):
                paused = True

        if proc_count % 30 == 0:
            st = face_engine.stats
            elapsed = time.monotonic() - t_start
            logger.info(
                f"frame={proc_count:5d} | fps={proc_count/elapsed:5.1f} | "
                f"faces={st['total_detected']}  matched={st['total_matched']}  "
                f"events={total_events}"
            )

        if args.max_frames and proc_count >= args.max_frames:
            break

    # -- Cleanup ---------------------------------------------------------------
    if args.save_watchlist:
        face_engine.save_watchlist(args.save_watchlist)
        logger.info(f"Watchlist saved: {args.save_watchlist}")

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    ev_logger.print_summary()
    elapsed = time.monotonic() - t_start
    st = face_engine.stats
    logger.info(
        f"Done -- {proc_count} frames in {elapsed:.1f}s | "
        f"faces={st['total_detected']} matched={st['total_matched']}"
    )


if __name__ == "__main__":
    main()
