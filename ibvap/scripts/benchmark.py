#!/usr/bin/env python3
"""
scripts/benchmark.py
====================
IBVAP P8 — Command-line Performance Benchmarking Tool.
Runs the complete profiler suite and produces an executive Markdown report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from services.benchmarks.profiler import PerformanceProfiler

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)


def main():
    parser = argparse.ArgumentParser(description="IBVAP System Performance & Capacity Profiler")
    parser.add_argument("--output", default="benchmark_report.md", help="Filename of generated markdown report")
    parser.add_argument("--json", action="store_true", help="Also save results as raw JSON")
    args = parser.parse_args()

    profiler = PerformanceProfiler()
    results = profiler.run_full_suite()
    report_file = profiler.generate_markdown_report(results, filename=args.output)

    print("\n" + "=" * 70)
    print(f" BENCHMARK COMPLETE: Report saved to {report_file}")
    print("=" * 70)
    print(f" * Maximum recommended real-time streams (10 FPS): {results.max_recommended_streams_10fps}")
    for name, stats in results.benchmarks.items():
        print(f" * {name:25s}: {stats.fps:6.1f} FPS  (mean: {stats.mean_ms:5.2f} ms | p95: {stats.p95_ms:5.2f} ms)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
