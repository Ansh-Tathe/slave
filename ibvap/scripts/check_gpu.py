#!/usr/bin/env python3
"""
scripts/check_gpu.py
====================
IBVAP P0 — GPU & TensorRT readiness check.

Validates:
  1. Python version >= 3.11
  2. CUDA available via PyTorch
  3. GPU name & VRAM (warns if < 5 GB free)
  4. cuDNN enabled
  5. FP16 (half-precision) tensor operations
  6. TensorRT Python bindings (optional, warns if absent)
  7. OpenCV build info (CUDA flag)
  8. Ultralytics YOLO warm inference pass (YOLOv8n on a blank frame)
  9. Runtime FPS estimate for a tiny batch

Usage:
    python scripts/check_gpu.py
    python scripts/check_gpu.py --strict   # exit 1 on any warning

Exit codes:
    0  — all checks passed (warnings are acceptable)
    1  -- a required check failed OR --strict mode with any warning
"""

from __future__ import annotations

import sys
import time
import platform
import argparse
from typing import List, Tuple

# ── pretty terminal output ────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore

# ── result helpers ────────────────────────────────────────────────────────────
PASS  = "✅ PASS"
WARN  = "⚠️  WARN"
FAIL  = "❌ FAIL"

CheckResult = Tuple[str, str, str]  # (check_name, status, detail)
results: List[CheckResult] = []


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    sym = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[status]
    print(f"  {sym}  {name:<35} {detail}")


# =============================================================================
# Check 1 — Python version
# =============================================================================
def check_python() -> None:
    ver = sys.version_info
    detail = f"Python {ver.major}.{ver.minor}.{ver.micro} on {platform.system()}"
    if ver >= (3, 11):
        record("Python version", "PASS", detail)
    else:
        record("Python version", "FAIL", f"{detail} — need >= 3.11")


# =============================================================================
# Check 2 — PyTorch + CUDA
# =============================================================================
def check_pytorch() -> bool:
    """Returns True if CUDA is available."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        version  = torch.__version__
        if cuda_ok:
            record("PyTorch CUDA", "PASS",
                   f"torch {version}, CUDA {torch.version.cuda}")
            return True
        else:
            record("PyTorch CUDA", "FAIL",
                   f"torch {version} — CUDA NOT available. "
                   "Install: uv pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False
    except ImportError:
        record("PyTorch CUDA", "FAIL", "PyTorch not installed")
        return False


# =============================================================================
# Check 3 — GPU name & VRAM
# =============================================================================
def check_vram() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        props     = torch.cuda.get_device_properties(0)
        total_gb  = props.total_memory / 1024**3
        free_b, _ = torch.cuda.mem_get_info(0)
        free_gb   = free_b / 1024**3
        detail = (
            f"{props.name} | total={total_gb:.1f} GB, "
            f"free={free_gb:.1f} GB, "
            f"compute={props.major}.{props.minor}"
        )
        if free_gb >= 5.0:
            record("GPU VRAM (>= 5 GB free)", "PASS", detail)
        elif free_gb >= 3.0:
            record("GPU VRAM (>= 5 GB free)", "WARN",
                   f"{detail} — may be tight; close other GPU apps")
        else:
            record("GPU VRAM (>= 5 GB free)", "FAIL",
                   f"{detail} — too little free VRAM for real-time FP16")
    except Exception as exc:
        record("GPU VRAM", "FAIL", str(exc))


# =============================================================================
# Check 4 — cuDNN
# =============================================================================
def check_cudnn() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        enabled = torch.backends.cudnn.is_available()
        version  = torch.backends.cudnn.version()
        detail   = f"cuDNN {version}"
        record("cuDNN", "PASS" if enabled else "FAIL", detail)
    except Exception as exc:
        record("cuDNN", "FAIL", str(exc))


# =============================================================================
# Check 5 — FP16 computation
# =============================================================================
def check_fp16() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        a = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
        b = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
        t0 = time.perf_counter()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record("FP16 matmul (2048x2048)", "PASS",
               f"{elapsed_ms:.1f} ms — FP16 ops work")
    except Exception as exc:
        record("FP16 matmul", "FAIL", str(exc))


# =============================================================================
# Check 6 — TensorRT (optional)
# =============================================================================
def check_tensorrt() -> None:
    try:
        import tensorrt as trt  # type: ignore
        record("TensorRT", "PASS", f"TensorRT {trt.__version__}")
    except ImportError:
        record("TensorRT", "WARN",
               "tensorrt not installed — ONNX/PyTorch inference will be used. "
               "Install for max perf: pip install nvidia-tensorrt==10.x.x")


# =============================================================================
# Check 7 — OpenCV CUDA
# =============================================================================
def check_opencv() -> None:
    try:
        import cv2
        build_info = cv2.getBuildInformation()
        cuda_flag  = "Use CUDA" in build_info and "YES" in build_info.split("Use CUDA")[-1][:20]
        detail = f"OpenCV {cv2.__version__}, CUDA={'yes' if cuda_flag else 'no (CPU-only build)'}"
        # CPU build is fine; we do GPU work in PyTorch/ONNX
        record("OpenCV", "PASS", detail)
    except ImportError:
        record("OpenCV", "FAIL", "opencv-python not installed")


# =============================================================================
# Check 8 — Ultralytics YOLO warm inference
# =============================================================================
def check_yolo_inference() -> None:
    try:
        import numpy as np
        from ultralytics import YOLO

        print("\n  [YOLO warm-up] Downloading YOLOv8n (~6 MB) if not cached …")
        model = YOLO("yolov8n.pt")  # auto-downloads to ~/.config/ultralytics/

        # Create a synthetic 720p blank frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # --- timing over 30 frames ---
        N      = 30
        t0     = time.perf_counter()
        for _ in range(N):
            model.predict(frame, verbose=False, device=0, half=True, imgsz=640)
        elapsed = time.perf_counter() - t0

        fps    = N / elapsed
        ms_per = elapsed / N * 1000
        detail = f"{fps:.1f} FPS  ({ms_per:.1f} ms/frame)  [30 blank 720p frames, FP16]"

        if fps >= 10:
            record("YOLO FP16 inference speed", "PASS", detail)
        elif fps >= 5:
            record("YOLO FP16 inference speed", "WARN",
                   f"{detail} — marginal; try TensorRT export")
        else:
            record("YOLO FP16 inference speed", "FAIL",
                   f"{detail} — too slow; check CUDA install")
    except Exception as exc:
        record("YOLO FP16 inference speed", "FAIL", str(exc))


# =============================================================================
# Check 9 — Redis connectivity (optional)
# =============================================================================
def check_redis() -> None:
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        r.ping()
        record("Redis connection", "PASS", "localhost:6379 reachable")
    except Exception:
        record("Redis connection", "WARN",
               "Redis not running — needed for event bus in P7. "
               "Start with: docker run -p 6379:6379 redis:alpine")


# =============================================================================
# Summary
# =============================================================================
def print_summary(strict: bool) -> int:
    fails  = [r for r in results if r[1] == "FAIL"]
    warns  = [r for r in results if r[1] == "WARN"]
    passed = [r for r in results if r[1] == "PASS"]

    print("\n" + "=" * 65)
    print(f"  IBVAP GPU Check Summary")
    print("=" * 65)
    print(f"  {PASS} : {len(passed)}")
    print(f"  {WARN} : {len(warns)}")
    print(f"  FAIL : {len(fails)}")
    print("=" * 65)

    if fails:
        print("\n  ❌ FAILED checks (must fix before proceeding):")
        for name, _, detail in fails:
            print(f"      • {name}: {detail}")

    if warns:
        print("\n  ⚠️  WARNINGS (system will work but may be slower):")
        for name, _, detail in warns:
            print(f"      • {name}: {detail}")

    if not fails and not warns:
        print("\n  🎉 All checks passed! Your environment is ready for P1.\n")

    if fails:
        return 1
    if strict and warns:
        return 1
    return 0


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="IBVAP GPU readiness checker")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 on warnings as well as failures")
    parser.add_argument("--skip-yolo", action="store_true",
                        help="Skip the YOLO warm-inference check (faster)")
    args = parser.parse_args()

    print("=" * 65)
    print("  IBVAP — GPU & Environment Check")
    print("=" * 65 + "\n")

    check_python()
    cuda_ok = check_pytorch()
    if cuda_ok:
        check_vram()
        check_cudnn()
        check_fp16()
    check_tensorrt()
    check_opencv()
    if not args.skip_yolo:
        check_yolo_inference()
    check_redis()

    sys.exit(print_summary(args.strict))


if __name__ == "__main__":
    main()
