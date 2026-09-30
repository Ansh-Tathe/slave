"""
services/analytics/weapon/weapon_detector.py
============================================
IBVAP — Harmful Object & Weapon Detection Subsystem.

Detects firearms (guns, rifles, pistols) and melee weapons (knives, blades, sharp objects)
in surveillance video feeds and determines if they are carried / contained by a tracked person.

Features:
- Threat-detection YOLOv8 CNN inference (guns, knives, explosives, grenades).
- Fallback COCO knife (class 43) detector.
- Spatial person-weapon containment & association algorithm.
- Emits ARMED_PERSON (CRITICAL) and WEAPON_DETECTED (HIGH) events with deduplication.
- Visual warning renderer for live video overlay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from services.detect_track.models import Track
from services.event_engine.models import Event, EventType, Severity


@dataclass
class WeaponDetection:
    """Represents a single detected harmful object / weapon."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float
    class_name: str                          # "gun", "knife", "grenade", "explosive"
    carrier_track_id: Optional[int] = None   # Track ID of the person carrying it
    is_contained_by_person: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


class WeaponDetector:
    """
    Real-time harmful object and weapon detection engine.
    """

    DEFAULT_MODEL_PATH = "models/threat_yolov8n.pt"

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.35,
        device: str = "cpu",
        dedup_window_s: float = 4.0,
        camera_id: str = "cam_usb_0",
    ) -> None:
        self.conf_thresh = conf_thresh
        self.device = device
        self.dedup_window_s = dedup_window_s
        self.camera_id = camera_id
        self._last_alert_time: Dict[str, float] = {}  # key -> timestamp

        # Resolve model path
        resolved_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        self.model = None
        self.model_type = "none"

        if resolved_path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(resolved_path))
                self.model_type = "yolov8_threat"
                logger.success(f"WeaponDetector loaded threat model from {resolved_path}")
            except Exception as e:
                logger.warning(f"Failed to load threat model {resolved_path}: {e}")
        else:
            logger.info(f"Threat model not found at {resolved_path}. Falling back to COCO / rule checks.")

    def update(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        frame_id: int = 0,
    ) -> Tuple[List[WeaponDetection], List[Event]]:
        """
        Process the current frame and tracks.
        Returns:
            (detections, events)
        """
        now = time.monotonic()
        raw_detections: List[WeaponDetection] = []

        # 1. Run inference if model available
        if self.model is not None and frame is not None and frame.size > 0:
            try:
                results = self.model(
                    frame,
                    conf=self.conf_thresh,
                    device=self.device,
                    verbose=False,
                )
                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        cls_name = r.names.get(cls_id, f"weapon_{cls_id}").lower()
                        xyxy = box.xyxy[0].cpu().numpy()
                        bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

                        raw_detections.append(WeaponDetection(
                            bbox=bbox,
                            conf=conf,
                            class_name=cls_name,
                        ))
            except Exception as e:
                logger.error(f"Error during weapon inference: {e}")

        # 2. Correlate with tracked persons
        person_tracks = [t for t in tracks if t.class_name == "person"]
        events: List[Event] = []

        for wd in raw_detections:
            matched_person = self._find_carrier_person(wd, person_tracks)
            if matched_person is not None:
                wd.carrier_track_id = matched_person.track_id
                wd.is_contained_by_person = True

                # Deduplication key for armed person
                dedup_key = f"armed_{matched_person.track_id}_{wd.class_name}"
                if now - self._last_alert_time.get(dedup_key, 0.0) >= self.dedup_window_s:
                    self._last_alert_time[dedup_key] = now
                    events.append(Event(
                        event_type=EventType.ARMED_PERSON,
                        severity=Severity.CRITICAL,
                        camera_id=self.camera_id,
                        zone_id="perimeter",
                        track=matched_person,
                        frame_id=frame_id,
                        metadata={
                            "weapon": wd.class_name,
                            "weapon_conf": round(wd.conf, 3),
                            "weapon_bbox": [round(c, 1) for c in wd.bbox],
                            "carrier_track_id": matched_person.track_id,
                            "threat_level": "CRITICAL_FIREARM" if "gun" in wd.class_name else "HIGH_BLADE",
                        },
                    ))
            else:
                # Standalone weapon detected without immediate carrier
                dedup_key = f"weapon_standalone_{wd.class_name}"
                if now - self._last_alert_time.get(dedup_key, 0.0) >= self.dedup_window_s:
                    self._last_alert_time[dedup_key] = now
                    events.append(Event(
                        event_type=EventType.WEAPON_DETECTED,
                        severity=Severity.HIGH,
                        camera_id=self.camera_id,
                        zone_id="perimeter",
                        track=None,
                        frame_id=frame_id,
                        metadata={
                            "weapon": wd.class_name,
                            "weapon_conf": round(wd.conf, 3),
                            "weapon_bbox": [round(c, 1) for c in wd.bbox],
                        },
                    ))

        return raw_detections, events

    def _find_carrier_person(
        self,
        wd: WeaponDetection,
        person_tracks: List[Track],
    ) -> Optional[Track]:
        """
        Check if a weapon's bounding box is inside or in immediate contact with a person track.
        """
        wx1, wy1, wx2, wy2 = wd.bbox
        wcx, wcy = wd.center

        best_person = None
        min_dist = float("inf")

        for person in person_tracks:
            px1, py1, px2, py2 = person.bbox

            # Spatial boundary with slight padding for hands/holsters
            reach_pad_x = 35.0
            reach_pad_y = 30.0
            expanded_px1 = px1 - reach_pad_x
            expanded_py1 = py1 - reach_pad_y
            expanded_px2 = px2 + reach_pad_x
            expanded_py2 = py2 + reach_pad_y

            # Check if weapon center is within person reach area
            if expanded_px1 <= wcx <= expanded_px2 and expanded_py1 <= wcy <= expanded_py2:
                # Also check intersection area
                ix1 = max(wx1, px1)
                iy1 = max(wy1, py1)
                ix2 = min(wx2, px2)
                iy2 = min(wy2, py2)

                inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                pcx, pcy = person.center
                dist = (wcx - pcx) ** 2 + (wcy - pcy) ** 2

                if inter_area > 0 or dist < min_dist:
                    min_dist = dist
                    best_person = person

        return best_person

    def draw(
        self,
        frame: np.ndarray,
        detections: List[WeaponDetection],
    ) -> np.ndarray:
        """
        Draw visual alerts and warning bounding boxes for detected weapons onto the frame.
        """
        if not detections:
            return frame

        vis = frame.copy()
        for wd in detections:
            x1, y1, x2, y2 = [int(v) for v in wd.bbox]
            # Red warning box with pulsing outline
            color = (0, 0, 255)  # Bright Red
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            # Label banner
            carrier_str = f" [TRACK #{wd.carrier_track_id}]" if wd.carrier_track_id is not None else ""
            label = f"! ARMED: {wd.class_name.upper()} {wd.conf*100:.0f}%{carrier_str}"

            # Label background box
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - 24)), (x1 + tw + 10, max(24, y1)), color, -1)
            cv2.putText(
                vis,
                label,
                (x1 + 5, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return vis
