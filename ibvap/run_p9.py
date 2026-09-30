#!/usr/bin/env python3
"""
run_p9.py
=========
IBVAP P9 — Natural Language Surveillance Search & Incident Report Generator.

Features:
  1. Plain-English Semantic Querying across historical surveillance logs.
  2. Automatic Intent Extraction (event type, severity, camera, time window, vehicle plate).
  3. Formal Security Incident Dossier Generation (HTML, Markdown, JSON).

Usage:
  python run_p9.py
  python run_p9.py --search "critical loitering near main gate yesterday"
  python run_p9.py --search "wanted vehicle plate KA01AB1234"
  python run_p9.py --report-event <event_id>
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from services.api.models import EventResponse
from services.api.state import state
from services.reports.generator import IncidentReportGenerator
from services.search.engine import NLSearchEngine
from services.search.query_parser import NLQueryParser

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)


def seed_demo_events_if_empty() -> None:
    """Populate sample events if event store is empty."""
    if state.event_store.count() > 0:
        return

    now = time.time()
    sample_events = [
        EventResponse(
            event_id="e1010101-0000-0000-0000-000000000001",
            event_type="LOITERING",
            severity="HIGH",
            camera_id="cam_01",
            zone_id="cam_01_zones",
            track_id=104,
            class_name="person",
            class_id=0,
            confidence=0.94,
            bbox=[120.0, 140.0, 200.0, 360.0],
            center=[160.0, 250.0],
            frame_id=450,
            timestamp=now - 1200,
            confirmed=True,
            metadata={"dwell_s": 45.2, "operator_notes": "Suspect stood near security checkpoint"},
        ),
        EventResponse(
            event_id="e1010101-0000-0000-0000-000000000002",
            event_type="TRIPWIRE_CROSS",
            severity="CRITICAL",
            camera_id="cam_01",
            zone_id="wire_border_line",
            track_id=104,
            class_name="person",
            class_id=0,
            confidence=0.96,
            bbox=[140.0, 160.0, 210.0, 370.0],
            center=[175.0, 265.0],
            frame_id=490,
            timestamp=now - 1160,
            confirmed=True,
            metadata={"direction": "inbound", "operator_notes": "Confirmed unauthorized border crossing"},
        ),
        EventResponse(
            event_id="e1010101-0000-0000-0000-000000000003",
            event_type="ANPR_WATCHLIST",
            severity="CRITICAL",
            camera_id="cam_01",
            zone_id="cam_01_zones",
            track_id=208,
            class_name="car",
            class_id=2,
            confidence=0.91,
            bbox=[300.0, 400.0, 600.0, 650.0],
            center=[450.0, 525.0],
            frame_id=820,
            timestamp=now - 3600,
            confirmed=True,
            metadata={"plate": "KA01AB1234", "watchlist_reason": "Stolen vehicle alert"},
        ),
        EventResponse(
            event_id="e1010101-0000-0000-0000-000000000004",
            event_type="GATHERING",
            severity="HIGH",
            camera_id="cam_01",
            zone_id="cam_01_zones",
            track_id=312,
            class_name="person",
            class_id=0,
            confidence=0.89,
            bbox=[250.0, 300.0, 450.0, 500.0],
            center=[350.0, 400.0],
            frame_id=1240,
            timestamp=now - 7200,
            confirmed=None,
            metadata={"count": 6, "duration_s": 15.0},
        ),
        EventResponse(
            event_id="e1010101-0000-0000-0000-000000000005",
            event_type="NIGHT_MOVEMENT",
            severity="HIGH",
            camera_id="cam_test_file",
            track_id=405,
            class_name="person",
            class_id=0,
            confidence=0.86,
            bbox=[50.0, 50.0, 120.0, 200.0],
            center=[85.0, 125.0],
            frame_id=1800,
            timestamp=now - 28000,
            confirmed=False,
            metadata={"lux_estimate": 4.5, "operator_notes": "Authorized perimeter maintenance crew"},
        ),
    ]

    for ev in sample_events:
        state.event_store.add(ev)


def run_nl_search(query: str) -> None:
    print("\n" + "=" * 72)
    print(f" NATURAL LANGUAGE SURVEILLANCE SEARCH: '{query}'")
    print("=" * 72)

    engine = NLSearchEngine()
    events = list(state.event_store._events_list)
    parsed, results, total = engine.search(query, events=events)

    print("\n [1] Parsed Intent & Extracted Filters:")
    if parsed.event_types:
        print(f"     * Target Event Types: {parsed.event_types}")
    if parsed.severities:
        print(f"     * Target Severities : {parsed.severities}")
    if parsed.camera_ids:
        print(f"     * Target Cameras    : {parsed.camera_ids}")
    if parsed.classes:
        print(f"     * Target Classes    : {parsed.classes}")
    if parsed.plate:
        print(f"     * License Plate     : {parsed.plate}")
    if parsed.start_time:
        print(f"     * Time Range (Start): {time.ctime(parsed.start_time)}")
    if parsed.keywords:
        print(f"     * Semantic Keywords : {parsed.keywords}")

    print(f"\n [2] Ranked Search Results ({len(results)} of {total} matches):")
    if not results:
        print("     No matching surveillance events found.")
        return

    for idx, scored in enumerate(results, 1):
        ev = scored.event
        t_str = time.strftime("%H:%M:%S UTC", time.gmtime(ev.timestamp))
        print(f"\n  #{idx} [Score: {scored.score:4.1f}] Event ID: {ev.event_id[:8]}... | {ev.event_type} ({ev.severity})")
        print(f"      Time: {t_str} | Camera: {ev.camera_id} | Class: {ev.class_name.upper()} (Track #{ev.track_id})")
        print(f"      Reasons: {', '.join(scored.match_reasons)}")
        if ev.metadata:
            print(f"      Metadata: {ev.metadata}")


def run_generate_report(event_id: str, officer: str = "Duty Operations Officer") -> None:
    print("\n" + "=" * 72)
    print(f" GENERATING FORMAL INCIDENT REPORT FOR EVENT: {event_id}")
    print("=" * 72)

    ev = state.event_store.get(event_id)
    if not ev:
        print(f" [X] Event {event_id} not found in event store.")
        return

    # Find correlated tracks
    all_events = list(state.event_store._events_list)
    correlated = [
        e for e in all_events
        if e.event_id != ev.event_id and (
            e.track_id == ev.track_id or (e.camera_id == ev.camera_id and abs(e.timestamp - ev.timestamp) <= 300)
        )
    ]

    generator = IncidentReportGenerator()
    report = generator.generate(
        primary_event=ev,
        correlated_events=correlated,
        investigating_officer=officer,
        facility="Sector 4 Border Surveillance Post",
    )

    paths = generator.export_all(report)
    print(f"\n [*] Incident Report Generated: {report.report_id}")
    print(f"     Title: {report.title}")
    print(f"     Status: {report.status} | Severity: {report.severity}")
    print(f"     Summary: {report.summary[:140]}...")
    print(f"\n [*] Exported Files:")
    print(f"     * HTML Report : {paths['html']}")
    print(f"     * Markdown    : {paths['md']}")
    print(f"     * JSON Schema : {paths['json']}")


def main():
    parser = argparse.ArgumentParser(description="IBVAP P9 — NL Search & Incident Report Generator")
    parser.add_argument("--search", help="Execute natural language search query")
    parser.add_argument("--report-event", help="Generate incident report for a specific event ID")
    parser.add_argument("--officer", default="Duty Officer", help="Officer name for incident report")
    args = parser.parse_args()

    # Load / seed events
    state.init()
    seed_demo_events_if_empty()

    if args.search:
        run_nl_search(args.search)
        return

    if args.report_event:
        run_generate_report(args.report_event, officer=args.officer)
        return

    # Default interactive demonstration:
    print("=" * 72)
    print(" IBVAP P9 — Natural Language Search & Incident Reporting Demo")
    print("=" * 72)

    # Demo 1: Multi-criteria NL Search
    run_nl_search("find critical border crossings near main gate")

    # Demo 2: Vehicle Plate NL Search
    run_nl_search("show blacklisted car KA01AB1234")

    # Demo 3: Incident Report generation on the critical intrusion event
    intrusion_ev_id = "e1010101-0000-0000-0000-000000000002"
    run_generate_report(intrusion_ev_id, officer="Inspector R. Sharma (Border Guard SOC)")

    print("\n" + "=" * 72)
    print(" ALL P9 DEMONSTRATION WORKFLOWS COMPLETED SUCCESSFULLY")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
