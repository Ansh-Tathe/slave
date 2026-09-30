"""
tests/test_vlm.py
=================
Unit tests for IBVAP Vision-Language Model (VLM) inspector and REST endpoints.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.analytics.vlm.inspector import VLMInspector
from services.api.main import app
from services.api.models import EventCreate, EventResponse
from services.api.state import state


@pytest.fixture
def client():
    return TestClient(app)


def test_vlm_cv_heuristic_color_detection(tmp_path: Path):
    """Verify synthetic image clothing color detection (blue shirt, dark pants)."""
    inspector = VLMInspector()
    # Create 200x100 synthetic person crop (H=200, W=100)
    # Upper half blue (BGR: 220, 50, 20), lower half dark/black (BGR: 20, 20, 20)
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    img[:100, :] = [220, 50, 20]   # blue
    img[100:, :] = [20, 20, 20]    # black/dark

    res = inspector.inspect_image(img)
    assert res.provider == "cv_heuristic"
    assert "blue" in res.upper_color.lower()
    assert "black" in res.lower_color.lower() or "dark" in res.lower_color.lower()
    assert len(res.description) > 0


def test_vlm_missing_image():
    """Verify handling of missing image file path."""
    inspector = VLMInspector()
    res = inspector.inspect_image("non_existent_snapshot_path_12345.jpg")
    assert res.provider == "error"
    assert "not found" in res.description.lower()


def test_vlm_small_image():
    """Verify handling of very small target crops."""
    inspector = VLMInspector()
    tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
    res = inspector.inspect_image(tiny_img)
    assert "too small" in res.description.lower()


def test_vlm_api_endpoint_with_image(client: TestClient, tmp_path: Path):
    """Test POST /api/v1/vlm/inspect with direct image path."""
    test_img_path = tmp_path / "target.jpg"
    img = np.full((120, 60, 3), (40, 180, 50), dtype=np.uint8)  # green
    cv2.imwrite(str(test_img_path), img)

    resp = client.post("/api/v1/vlm/inspect", json={"image_path": str(test_img_path)})
    assert resp.status_code == 200
    data = resp.json()
    assert "description" in data
    assert data["provider"] in ("cv_heuristic", "ollama")
    assert data["latency_ms"] >= 0.0


def test_vlm_api_endpoint_with_event(client: TestClient, tmp_path: Path):
    """Test POST /api/v1/vlm/inspect linked to an existing event."""
    test_img_path = tmp_path / "person_alert.jpg"
    img = np.full((150, 75, 3), (30, 30, 200), dtype=np.uint8)  # red
    cv2.imwrite(str(test_img_path), img)

    # Inject event into state
    ev_in = EventResponse(
        event_id="ev_test_vlm_001",
        event_type="ZONE_INTRUSION",
        severity="HIGH",
        camera_id="cam_usb_0",
        track_id=101,
        class_name="person",
        class_id=0,
        confidence=0.92,
        bbox=[50, 50, 200, 200],
        center=[125, 125],
        frame_id=1,
        timestamp=1700000000.0,
        snapshot_path=str(test_img_path),
        metadata={},
    )
    state.event_store.add(ev_in)

    resp = client.post("/api/v1/vlm/inspect", json={"event_id": ev_in.event_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["event_id"] == ev_in.event_id
    assert "description" in data

    # Verify event store was enriched
    refreshed_ev = state.event_store.get(ev_in.event_id)
    assert "vlm_description" in refreshed_ev.metadata
    assert refreshed_ev.metadata["vlm_provider"] in ("cv_heuristic", "ollama")


def test_vlm_api_missing_params(client: TestClient):
    """Test validation when neither event_id nor image_path is provided."""
    resp = client.post("/api/v1/vlm/inspect", json={})
    assert resp.status_code == 400
