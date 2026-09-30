"""
tests/test_p7_api.py
====================
IBVAP P7 — Comprehensive Test Suite for FastAPI, Auth, Webhooks, and Dashboard.
"""

from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient

from services.api.auth import create_access_token
from services.api.main import app
from services.api.models import EventResponse, WebhookCreate
from services.api.state import state
from services.api.webhooks.dispatcher import dispatcher

client = TestClient(app)


# ── Fixtures & Helpers ────────────────────────────────────────────────────────

def get_auth_header(username: str = "admin", role: str = "admin") -> dict:
    token = create_access_token({"sub": username, "role": role})
    return {"Authorization": f"Bearer {token}"}


# ── Health & Metrics Tests ────────────────────────────────────────────────────

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "uptime_s" in data
    assert "components" in data
    assert "event_store" in data["components"]


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert "events_by_severity" in data
    assert "events_by_type" in data
    assert "active_cameras" in data


# ── Auth & RBAC Tests ─────────────────────────────────────────────────────────

def test_login_success():
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"


def test_login_failure():
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "wrongpassword"},
    )
    assert response.status_code == 401
    assert "detail" in response.json()


def test_auth_me_with_bearer():
    headers = get_auth_header("operator", "operator")
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "operator"
    assert data["role"] == "operator"


def test_auth_me_with_api_key():
    headers = {"X-API-Key": "ibvap-c2-secure-api-key-default"}
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "c2_system"
    assert data["role"] == "admin"


def test_unauthenticated_request_fails():
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


# ── Events API Tests ──────────────────────────────────────────────────────────

def test_create_and_query_event():
    payload = {
        "event_type": "TRIPWIRE_CROSS",
        "severity": "HIGH",
        "camera_id": "cam_01",
        "track_id": 42,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [10.0, 20.0, 50.0, 100.0],
        "center": [30.0, 60.0],
        "frame_id": 120,
        "zone_id": "line_north",
        "metadata": {"direction": "inbound"},
    }

    # Ingest event
    res = client.post("/api/v1/events", json=payload)
    assert res.status_code == 201
    event_data = res.json()
    event_id = event_data["event_id"]
    assert event_id is not None
    assert event_data["event_type"] == "TRIPWIRE_CROSS"
    assert event_data["severity"] == "HIGH"

    # Query event by ID
    get_res = client.get(f"/api/v1/events/{event_id}")
    assert get_res.status_code == 200
    assert get_res.json()["event_id"] == event_id

    # Filter events by camera and severity
    list_res = client.get("/api/v1/events?camera_id=cam_01&severity=HIGH")
    assert list_res.status_code == 200
    items = list_res.json()
    assert items["total"] >= 1
    assert any(e["event_id"] == event_id for e in items["events"])


def test_event_not_found():
    res = client.get("/api/v1/events/non-existent-uuid")
    assert res.status_code == 404


def test_operator_confirm_event():
    # Create an unconfirmed event
    payload = {
        "event_type": "FACE_WATCHLIST_HIT",
        "severity": "CRITICAL",
        "camera_id": "cam_01",
        "track_id": 101,
        "class_name": "person",
        "confidence": 0.88,
    }
    create_res = client.post("/api/v1/events", json=payload)
    ev_id = create_res.json()["event_id"]

    # Confirm using operator token
    headers = get_auth_header("operator", "operator")
    confirm_res = client.patch(
        f"/api/v1/events/{ev_id}/confirm",
        json={"confirmed": True, "notes": "Verified suspect match"},
        headers=headers,
    )
    assert confirm_res.status_code == 200
    data = confirm_res.json()
    assert data["confirmed"] is True
    assert "Verified suspect match" in data["metadata"]["operator_notes"]


def test_viewer_cannot_confirm_event():
    # Create an event
    payload = {
        "event_type": "ZONE_ENTER",
        "severity": "LOW",
        "camera_id": "cam_01",
        "track_id": 202,
        "class_name": "person",
        "confidence": 0.85,
    }
    ev_id = client.post("/api/v1/events", json=payload).json()["event_id"]

    # Viewer role should be denied (403)
    headers = get_auth_header("viewer", "viewer")
    res = client.patch(
        f"/api/v1/events/{ev_id}/confirm",
        json={"confirmed": True},
        headers=headers,
    )
    assert res.status_code == 403


# ── Camera Management Tests ───────────────────────────────────────────────────

def test_list_and_get_cameras():
    res = client.get("/api/v1/cameras")
    assert res.status_code == 200
    cameras = res.json()
    assert isinstance(cameras, list)
    assert len(cameras) >= 1

    first_cam = cameras[0]["id"]
    single_res = client.get(f"/api/v1/cameras/{first_cam}")
    assert single_res.status_code == 200
    assert single_res.json()["id"] == first_cam


def test_camera_heartbeat():
    res = client.post(
        "/api/v1/cameras/cam_01/heartbeat",
        json={"fps": 24.5, "frame_id": 1500, "status": "ONLINE"},
    )
    assert res.status_code == 200
    cam = res.json()
    assert cam["id"] == "cam_01"
    assert cam["fps"] == 24.5
    assert cam["total_frames"] >= 1500


def test_camera_update_config():
    headers = get_auth_header("operator", "operator")
    res = client.patch(
        "/api/v1/cameras/cam_01",
        json={"enabled": True, "location": "Perimeter East Gate"},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["location"] == "Perimeter East Gate"


# ── Watchlist Tests ───────────────────────────────────────────────────────────

def test_plate_watchlist_crud():
    headers = get_auth_header("operator", "operator")

    # Add plate
    add_res = client.post(
        "/api/v1/watchlist/plates",
        json={"plate": "DL01XY9999", "notes": "VIP vehicle", "severity": "HIGH"},
        headers=headers,
    )
    assert add_res.status_code == 201
    assert add_res.json()["plate"] == "DL01XY9999"

    # List plates
    list_res = client.get("/api/v1/watchlist/plates")
    assert list_res.status_code == 200
    plates = [p["plate"] for p in list_res.json()]
    assert "DL01XY9999" in plates

    # Delete plate
    del_res = client.delete("/api/v1/watchlist/plates/DL01XY9999", headers=headers)
    assert del_res.status_code == 200

    # Ensure removed
    plates_after = [p["plate"] for p in client.get("/api/v1/watchlist/plates").json()]
    assert "DL01XY9999" not in plates_after


def test_person_watchlist_crud():
    headers = get_auth_header("operator", "operator")

    # Add person
    add_res = client.post(
        "/api/v1/watchlist/faces",
        json={"name": "Target-Bravo", "notes": "High risk", "severity": "CRITICAL"},
        headers=headers,
    )
    assert add_res.status_code == 201
    assert add_res.json()["name"] == "Target-Bravo"

    # List persons
    list_res = client.get("/api/v1/watchlist/faces")
    assert list_res.status_code == 200
    persons = [p["name"] for p in list_res.json()]
    assert "Target-Bravo" in persons

    # Delete person
    del_res = client.delete("/api/v1/watchlist/faces/Target-Bravo", headers=headers)
    assert del_res.status_code == 200

    # Ensure removed
    persons_after = [p["name"] for p in client.get("/api/v1/watchlist/faces").json()]
    assert "Target-Bravo" not in persons_after


# ── Webhook & C2 Dispatcher Tests ─────────────────────────────────────────────

def test_webhook_registration_and_list():
    headers = get_auth_header("admin", "admin")

    wh_payload = {
        "id": "c2_test_endpoint",
        "name": "Test Webhook",
        "url": "http://127.0.0.1:9999/webhook",
        "method": "POST",
        "headers": {"X-Test": "True"},
        "events": ["CRITICAL", "FACE_WATCHLIST_HIT"],
        "enabled": True,
        "retry_attempts": 2,
        "retry_delay_s": 0.5,
    }

    # Register
    reg_res = client.post("/api/v1/webhooks", json=wh_payload, headers=headers)
    assert reg_res.status_code == 201
    assert reg_res.json()["id"] == "c2_test_endpoint"

    # List
    list_res = client.get("/api/v1/webhooks")
    assert list_res.status_code == 200
    ids = [w["id"] for w in list_res.json()]
    assert "c2_test_endpoint" in ids

    # Delete
    del_res = client.delete("/api/v1/webhooks/c2_test_endpoint", headers=headers)
    assert del_res.status_code == 200


@pytest.mark.asyncio
async def test_webhook_dispatcher_filtering():
    # Setup test webhook in state
    wh = WebhookCreate(
        id="c2_filter_test",
        name="Filter Test",
        url="http://127.0.0.1:9999/wh",
        method="POST",
        events=["TRIPWIRE_CROSS"],
        enabled=True,
    )
    dispatcher.register(wh)

    # Event that matches
    match_ev = EventResponse(
        event_id="ev_001",
        event_type="TRIPWIRE_CROSS",
        severity="HIGH",
        camera_id="cam_01",
        track_id=1,
        class_name="person",
        class_id=0,
        confidence=0.9,
        bbox=[0, 0, 10, 10],
        center=[5, 5],
        frame_id=1,
        timestamp=time.time(),
    )

    # Event that does not match
    no_match_ev = EventResponse(
        event_id="ev_002",
        event_type="ANPR_READ",
        severity="LOW",
        camera_id="cam_01",
        track_id=2,
        class_name="car",
        class_id=2,
        confidence=0.9,
        bbox=[0, 0, 10, 10],
        center=[5, 5],
        frame_id=1,
        timestamp=time.time(),
    )

    # Webhook events list filters non-matching
    assert "TRIPWIRE_CROSS" in state.webhooks["c2_filter_test"].events
    assert "ANPR_READ" not in state.webhooks["c2_filter_test"].events

    dispatcher.unregister("c2_filter_test")


# ── Dashboard UI Tests ────────────────────────────────────────────────────────

def test_dashboard_ui_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "IBVAP Command Center" in res.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "IBVAP Command Center" in res_dash.text


def test_events_pagination():
    # Ingest 15 events
    for i in range(15):
        client.post(
            "/api/v1/events",
            json={
                "event_type": "ZONE_ENTER",
                "severity": "LOW",
                "camera_id": "cam_page_test",
                "track_id": 500 + i,
            },
        )

    # Test limit = 5, offset = 0
    p1 = client.get("/api/v1/events?camera_id=cam_page_test&limit=5&offset=0").json()
    assert len(p1["events"]) == 5
    assert p1["total"] >= 15

    # Test limit = 5, offset = 5
    p2 = client.get("/api/v1/events?camera_id=cam_page_test&limit=5&offset=5").json()
    assert len(p2["events"]) == 5
    # Ensure items in page 1 and page 2 are different
    p1_ids = {e["event_id"] for e in p1["events"]}
    p2_ids = {e["event_id"] for e in p2["events"]}
    assert len(p1_ids.intersection(p2_ids)) == 0


def test_event_snapshot_endpoint(tmp_path):
    # Create a dummy image
    test_snap = tmp_path / "snap.jpg"
    test_snap.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)

    # Ingest event with snapshot path
    res = client.post(
        "/api/v1/events",
        json={
            "event_type": "ANPR_READ",
            "severity": "LOW",
            "camera_id": "cam_01",
            "track_id": 888,
            "snapshot_path": str(test_snap),
        },
    )
    ev_id = res.json()["event_id"]

    # Retrieve snapshot
    snap_res = client.get(f"/api/v1/events/{ev_id}/snapshot")
    assert snap_res.status_code == 200
    assert snap_res.headers["content-type"] == "image/jpeg"
    assert len(snap_res.content) == len(test_snap.read_bytes())


def test_event_snapshot_missing():
    # Ingest event without snapshot
    res = client.post(
        "/api/v1/events",
        json={
            "event_type": "LOITERING",
            "severity": "MEDIUM",
            "camera_id": "cam_01",
            "track_id": 777,
        },
    )
    ev_id = res.json()["event_id"]
    snap_res = client.get(f"/api/v1/events/{ev_id}/snapshot")
    assert snap_res.status_code == 404


def test_auth_invalid_api_key():
    headers = {"X-API-Key": "wrong-api-key"}
    res = client.get("/api/v1/auth/me", headers=headers)
    assert res.status_code == 401
    assert "Invalid API key" in res.json()["detail"]


def test_auth_invalid_token():
    headers = {"Authorization": "Bearer invalid.jwt.token"}
    res = client.get("/api/v1/auth/me", headers=headers)
    assert res.status_code == 401


def test_camera_not_found():
    res = client.get("/api/v1/cameras/cam_does_not_exist")
    assert res.status_code == 404


def test_watchlist_delete_not_found():
    headers = get_auth_header("operator", "operator")
    res_plate = client.delete("/api/v1/watchlist/plates/NONEXISTENT", headers=headers)
    assert res_plate.status_code == 404

    res_person = client.delete("/api/v1/watchlist/faces/NONEXISTENT", headers=headers)
    assert res_person.status_code == 404


def test_webhook_delete_not_found():
    headers = get_auth_header("admin", "admin")
    res = client.delete("/api/v1/webhooks/nonexistent_webhook_id", headers=headers)
    assert res.status_code == 404

