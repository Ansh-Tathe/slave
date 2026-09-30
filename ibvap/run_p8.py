#!/usr/bin/env python3
"""
run_p8.py
=========
IBVAP P8 — Hardening, Pre-flight Verification, and Capacity Benchmarks.

Runs:
  1. Configuration Schema & Coordinate Integrity Checks (ConfigValidator)
  2. Ingest Resilience & Watchdog Auto-Recovery Verification
  3. End-to-end Latency, Throughput, and Multi-stream Scaling Benchmarks

Usage:
  python run_p8.py
  python run_p8.py --validate-only
  python run_p8.py --benchmark-only
  python run_p8.py --output data/benchmarks/custom_report.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from services.benchmarks.profiler import PerformanceProfiler
from services.hardening.buffer import BoundedFrameBuffer, DropPolicy
from services.hardening.validator import ConfigValidator
from services.hardening.watchdog import StreamState, StreamWatchdog

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)


def run_configuration_audit() -> bool:
    print("\n" + "=" * 72)
    print(" 1. PRE-FLIGHT CONFIGURATION INTEGRITY AUDIT")
    print("=" * 72)

    validator = ConfigValidator(config_dir="configs")
    report = validator.validate_all()

    if report.warnings:
        print("\n [!] Warnings:")
        for w in report.warnings:
            print(f"     * {w}")

    if not report.is_valid:
        print("\n [X] Validation Failed with Errors:")
        for e in report.errors:
            print(f"     * {e}")
        print("\n Configuration integrity check: FAILED")
        return False

    print("\n [*] cameras.yaml : Valid schema, verified capture fps, no duplicate IDs")
    print(" [*] zones.yaml   : Valid polygons, coordinate bounds, and tripwire segments")
    print(" [*] rules.yaml   : Valid event types, severity levels, and webhook destinations")
    print("\n Configuration integrity check: PASSED (All configs verified)")
    return True


def run_watchdog_resilience_test():
    print("\n" + "=" * 72)
    print(" 2. INGEST RESILIENCE & STREAM WATCHDOG TEST")
    print("=" * 72)

    reconnect_attempts = 0

    def mock_reconnect():
        nonlocal reconnect_attempts
        reconnect_attempts += 1
        logger.info(f"[Mock Stream] Reconnection handler invoked (Attempt #{reconnect_attempts})")
        return True

    # 1. Test Buffer overflow backpressure
    buf: BoundedFrameBuffer[int] = BoundedFrameBuffer(maxsize=5, policy=DropPolicy.DROP_OLDEST)
    for i in range(12):
        buf.put(i)
    stats = buf.get_stats()
    print(f" [*] BoundedFrameBuffer: Pushed 12 frames into maxsize=5 buffer.")
    print(f"     -> Current size: {stats['current_size']}, Total dropped: {stats['total_dropped']} frames (DROP_OLDEST)")

    # 2. Test StreamWatchdog timeout & self-healing
    dog = StreamWatchdog(
        camera_id="cam_sim_01",
        timeout_s=0.3,
        reconnect_fn=mock_reconnect,
        base_delay_s=0.1,
    )
    dog.heartbeat(frame_id=1)
    time.sleep(0.4)  # Force timeout
    dog.check_health()
    assert dog.state == StreamState.STALLED, "Watchdog should detect stalled stream"

    print(" [*] Watchdog detected stalled stream after 0.3s timeout.")
    dog.attempt_reconnect()
    assert dog.state == StreamState.ONLINE, "Watchdog should restore stream to ONLINE"
    print(" [*] Watchdog executed auto-recovery callback and restored stream to ONLINE.")
    print("\n Ingest resilience test: PASSED")


def run_capacity_benchmarks(output_file: str):
    print("\n" + "=" * 72)
    print(" 3. ANALYTICS LATENCY & MULTI-STREAM CAPACITY BENCHMARKS")
    print("=" * 72)

    profiler = PerformanceProfiler()
    results = profiler.run_full_suite()
    report_path = profiler.generate_markdown_report(results, filename=output_file)

    print("\n --- Subsystem Latencies ---")
    for name, stats in results.benchmarks.items():
        print(f"  * {name:24s}: {stats.fps:6.1f} FPS | p50: {stats.p50_ms:5.2f} ms | p95: {stats.p95_ms:5.2f} ms")

    print("\n --- Multi-Stream Scaling Margin ---")
    for streams, sdata in sorted(results.stream_scaling.items()):
        status = "VIABLE (>= 10 FPS)" if sdata["realtime_viable_10fps"] else "BOTTLENECK"
        print(f"  * {streams} Streams: {sdata['aggregate_fps']:6.1f} FPS total ({sdata['per_stream_fps']:5.1f} FPS/cam) [{status}]")

    print(f"\n Capacity recommendation: Up to {results.max_recommended_streams_10fps} simultaneous 10 FPS camera streams.")
    print(f" Executive report written to: {report_path}")
    print("\n Benchmarks complete: PASSED")


def main():
    parser = argparse.ArgumentParser(description="IBVAP P8 — Hardening, Verification & Capacity Benchmarks")
    parser.add_argument("--validate-only", action="store_true", help="Only run config audit")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmarks")
    parser.add_argument("--output", default="benchmark_report.md", help="Output report filename")
    args = parser.parse_args()

    print("=" * 72)
    print(" IBVAP P8 — Hardening, Pre-Flight Verification & Benchmark Suite")
    print("=" * 72)

    if args.validate_only:
        run_configuration_audit()
        return

    if args.benchmark_only:
        run_capacity_benchmarks(args.output)
        return

    # Full run
    valid = run_configuration_audit()
    if not valid:
        sys.exit(1)

    run_watchdog_resilience_test()
    run_capacity_benchmarks(args.output)

    print("\n" + "=" * 72)
    print(" ALL P8 HARDENING & CAPACITY BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
