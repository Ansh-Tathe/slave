"""
tests/test_p9_search_reports.py
===============================
IBVAP P9 — Test Suite for Natural Language Search and Incident Reports.
"""

from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient

from services.api.auth import create_access_token
from services.api.main import app
from services.api.models import EventResponse
from services.api.state import state
from services.reports.generator import IncidentReportGenerator
from services.search.engine import NLSearchEngine
from services.search.query_parser import NLQueryParser

client = TestClient(app)


# ── Fixtures & Mock Events ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def populate_test_events():
    """Seed test events into the global state for search and report tests."""
    now = time.time()
    events = [
        EventResponse(
            event_id="test-ev-001",
            event_type="LOITERING",
            severity="HIGH",
            camera_id="cam_01",
            zone_id="zone_restricted",
            track_id=10,
            class_name="person",
            class_id=0,
            confidence=0.92,
            bbox=[100.0, 100.0, 150.0, 250.0],
            center=[125.0, 175.0],
            frame_id=100,
            timestamp=now - 600,
            confirmed=True,
            metadata={"dwell_s": 35.0, "operator_notes": "Suspicious person pacing near fence"},
        ),
        EventResponse(
            event_id="test-ev-002",
            event_type="TRIPWIRE_CROSS",
            severity="CRITICAL",
            camera_id="cam_01",
            zone_id="wire_border_line",
            track_id=10,
            class_name="person",
            class_id=0,
            confidence=0.95,
            bbox=[110.0, 110.0, 160.0, 260.0],
            center=[135.0, 185.0],
            frame_id=130,
            timestamp=now - 550,
            confirmed=True,
            metadata={"direction": "inbound"},
        ),
        EventResponse(
            event_id="test-ev-003",
            event_type="ANPR_WATCHLIST",
            severity="CRITICAL",
            camera_id="cam_01",
            track_id=25,
            class_name="car",
            class_id=2,
            confidence=0.88,
            bbox=[200.0, 300.0, 400.0, 500.0],
            center=[300.0, 400.0],
            frame_id=300,
            timestamp=now - 3000,
            confirmed=False,
            metadata={"plate": "KA01AB1234", "operator_notes": "Dismissed as false OCR read"},
        ),
        EventResponse(
            event_id="test-ev-004",
            event_type="RUNNING",
            severity="MEDIUM",
            camera_id="cam_test_file",
            track_id=40,
            class_name="person",
            class_id=0,
            confidence=0.84,
            bbox=[50.0, 50.0, 100.0, 180.0],
            center=[75.0, 115.0],
            frame_id=500,
            timestamp=now - 12000,
            confirmed=None,
            metadata={"speed_kmh": 22.5},
        ),
    ]

    for ev in events:
        state.event_store.add(ev)


# ── 1. Query Parser Unit Tests ────────────────────────────────────────────────

def test_query_parser_severity():
    parser = NLQueryParser()
    p1 = parser.parse("show critical emergency events")
    assert "CRITICAL" in p1.severities

    p2 = parser.parse("minor low priority warnings")
    assert "LOW" in p2.severities


def test_query_parser_event_types():
    parser = NLQueryParser()
    p1 = parser.parse("find all loitering and dwelling persons")
    assert "LOITERING" in p1.event_types

    p2 = parser.parse("boundary intrusion crossed tripwire line")
    assert "TRIPWIRE_CROSS" in p2.event_types

    p3 = parser.parse("wanted car plate read")
    assert "ANPR_WATCHLIST" in p3.event_types or "ANPR_READ" in p3.event_types


def test_query_parser_cameras_and_classes():
    parser = NLQueryParser()
    p = parser.parse("car detected near main gate on camera 1")
    assert "cam_01" in p.camera_ids
    assert "car" in p.classes


def test_query_parser_license_plate():
    parser = NLQueryParser()
    p = parser.parse("find vehicle KA-01-AB-1234 speeding")
    assert p.plate == "KA01AB1234"


def test_query_parser_temporal():
    parser = NLQueryParser()
    now = time.time()
    p1 = parser.parse("events in the last 15 minutes", ref_time=now)
    assert p1.start_time is not None
    assert int(now - p1.start_time) == 15 * 60

    p2 = parser.parse("events from the last 2 hours", ref_time=now)
    assert p2.start_time is not None
    assert int(now - p2.start_time) == 2 * 3600


def test_query_parser_confirmation():
    parser = NLQueryParser()
    p1 = parser.parse("confirmed intrusions")
    assert p1.confirmed is True

    p2 = parser.parse("rejected false alarms")
    assert p2.confirmed is False


# ── 2. Search Engine Tests ────────────────────────────────────────────────────

def test_search_engine_ranking():
    engine = NLSearchEngine()
    events = list(state.event_store._events_list)

    # Query targeting test-ev-002 specifically
    parsed, results, total = engine.search("critical intrusion tripwire crossed", events=events)
    assert total >= 1
    top_match = results[0]
    assert top_match.event.event_type == "TRIPWIRE_CROSS"
    assert top_match.event.severity == "CRITICAL"
    assert top_match.score > 20.0


def test_search_engine_plate_query():
    engine = NLSearchEngine()
    events = list(state.event_store._events_list)

    parsed, results, total = engine.search("car plate KA01AB1234", events=events)
    assert total >= 1
    assert results[0].event.event_id == "test-ev-003"
    assert any("License plate" in r for r in results[0].match_reasons)


def test_search_engine_empty_match():
    engine = NLSearchEngine()
    events = list(state.event_store._events_list)
    parsed, results, total = engine.search("nonexistent spaceship galaxy alert", events=events)
    assert total == 0
    assert len(results) == 0


# ── 3. Incident Report Generator Tests ────────────────────────────────────────

def test_incident_report_generation(tmp_path):
    gen = IncidentReportGenerator(output_dir=str(tmp_path))
    ev = state.event_store.get("test-ev-002")
    correlated = [state.event_store.get("test-ev-001")]

    report = gen.generate(
        primary_event=ev,
        correlated_events=correlated,
        investigating_officer="Inspector A. Singh",
        facility="North Gate Post",
    )

    assert report.report_id.startswith("INC-")
    assert report.severity == "CRITICAL"
    assert report.status == "CONFIRMED BREACH"
    assert len(report.timeline) == 2
    assert "Inspector A. Singh" in report.investigating_officer

    # Export
    paths = gen.export_all(report)
    assert paths["json"].exists()
    assert paths["md"].exists()
    assert paths["html"].exists()

    html_content = paths["html"].read_text(encoding="utf-8")
    assert report.report_id in html_content
    assert "TRIPWIRE_CROSS" in html_content
    assert "North Gate Post" in html_content


# ── 4. API Router Tests (Search & Reports) ────────────────────────────────────

def test_api_nl_search_post():
    res = client.post(
        "/api/v1/search/nl",
        json={"query": "critical border intrusion on cam 01", "limit": 10},
    )
    assert res.status_code == 200
    data = res.json()
    assert "parsed" in data
    assert "results" in data
    assert data["total"] >= 1
    assert data["results"][0]["event"]["camera_id"] == "cam_01"


def test_api_nl_search_get():
    res = client.get("/api/v1/search?q=loitering+person")
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "loitering person"
    assert data["total"] >= 1


def test_api_generate_and_view_report():
    token = create_access_token({"sub": "operator", "role": "operator"})
    headers = {"Authorization": f"Bearer {token}"}

    # Generate Report
    res = client.post(
        "/api/v1/reports/incident",
        json={
            "event_id": "test-ev-002",
            "investigating_officer": "Capt. V. Rao",
            "facility": "Sector 4 Headquarters",
        },
        headers=headers,
    )
    assert res.status_code == 201
    data = res.json()
    rep_id = data["report_id"]
    assert rep_id.startswith("INC-")
    assert "html_url" in data

    # Retrieve Report JSON
    get_json = client.get(f"/api/v1/reports/{rep_id}")
    assert get_json.status_code == 200
    assert get_json.json()["report_id"] == rep_id

    # Retrieve Report HTML
    get_html = client.get(f"/api/v1/reports/{rep_id}/html")
    assert get_html.status_code == 200
    assert "text/html" in get_html.headers["content-type"]
    assert rep_id in get_html.text
    assert "Sector 4 Headquarters" in get_html.text
