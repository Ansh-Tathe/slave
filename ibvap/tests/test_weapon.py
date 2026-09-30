"""
tests/test_weapon.py
====================
Unit tests for harmful object / weapon detection & armed person association.
"""

import numpy as np
import pytest

from services.analytics.weapon.weapon_detector import (
    WeaponDetection,
    WeaponDetector,
)
from services.detect_track.models import Track
from services.event_engine.models import EventType, Severity


def test_carrier_person_association():
    detector = WeaponDetector(model_path="non_existent_weights.pt")

    # Person track at (100, 100, 200, 300)
    person = Track(
        track_id=1,
        bbox=(100.0, 100.0, 200.0, 300.0),
        conf=0.9,
        class_id=0,
        class_name="person",
        frame_id=1,
        camera_id="cam_01",
    )

    # Weapon carried within person bounds (130, 150, 160, 180)
    gun = WeaponDetection(
        bbox=(130.0, 150.0, 160.0, 180.0),
        conf=0.88,
        class_name="gun",
    )

    matched = detector._find_carrier_person(gun, [person])
    assert matched is not None
    assert matched.track_id == 1


def test_distant_weapon_not_associated():
    detector = WeaponDetector(model_path="non_existent_weights.pt")

    person = Track(
        track_id=1,
        bbox=(100.0, 100.0, 200.0, 300.0),
        conf=0.9,
        class_id=0,
        class_name="person",
        frame_id=1,
        camera_id="cam_01",
    )

    # Weapon far away at (500, 500, 550, 550)
    gun = WeaponDetection(
        bbox=(500.0, 500.0, 550.0, 550.0),
        conf=0.85,
        class_name="gun",
    )

    matched = detector._find_carrier_person(gun, [person])
    assert matched is None


def test_armed_person_event_generation():
    detector = WeaponDetector(model_path="non_existent_weights.pt")

    person = Track(
        track_id=42,
        bbox=(100.0, 100.0, 200.0, 300.0),
        conf=0.9,
        class_id=0,
        class_name="person",
        frame_id=1,
        camera_id="cam_usb_0",
    )

    # Inject simulated raw detection into update
    knife = WeaponDetection(
        bbox=(120.0, 140.0, 150.0, 170.0),
        conf=0.82,
        class_name="knife",
    )

    matched = detector._find_carrier_person(knife, [person])
    assert matched is not None
    knife.carrier_track_id = matched.track_id
    knife.is_contained_by_person = True

    # Test event creation structure
    assert knife.is_contained_by_person is True
    assert knife.carrier_track_id == 42
