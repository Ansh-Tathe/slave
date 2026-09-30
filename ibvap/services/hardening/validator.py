"""
services/hardening/validator.py
===============================
IBVAP P8 — System Configuration Validator.
Performs pre-flight integrity, schema, and logic checks on cameras.yaml,
zones.yaml, and rules.yaml before starting video pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml
from loguru import logger

from services.event_engine.models import EventType, Severity

VALID_EVENT_TYPES = {e.value for e in EventType} | {
    "ABANDONED_OBJECT",
    "ANPR_WATCHLIST_HIT",
    "FACE_WATCHLIST_HIT",
}
VALID_SEVERITIES = {s.value for s in Severity}

VALID_TRIPWIRE_DIRECTIONS = {
    "both",
    "north_to_south",
    "south_to_north",
    "east_to_west",
    "west_to_east",
    "in",
    "out",
    "inbound",
    "outbound",
}


@dataclass
class ValidationReport:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class ConfigValidator:
    """Validates IBVAP YAML configuration files."""

    def __init__(self, config_dir: str = "configs") -> None:
        self.config_dir = Path(config_dir)

    def validate_all(self) -> ValidationReport:
        """Validate cameras.yaml, zones.yaml, and rules.yaml."""
        report = ValidationReport()

        cam_file = self.config_dir / "cameras.yaml"
        zone_file = self.config_dir / "zones.yaml"
        rule_file = self.config_dir / "rules.yaml"

        cams_data = self._load_yaml(cam_file, report)
        zones_data = self._load_yaml(zone_file, report)
        rules_data = self._load_yaml(rule_file, report)

        if cams_data is not None:
            self._validate_cameras(cams_data, report)

        if zones_data is not None:
            self._validate_zones(zones_data, report)

        if rules_data is not None:
            self._validate_rules(rules_data, report)

        # Cross-file referential integrity check:
        if cams_data and zones_data:
            self._validate_camera_zone_refs(cams_data, zones_data, report)

        return report

    def _load_yaml(self, path: Path, report: ValidationReport) -> Optional[dict]:
        if not path.exists():
            report.add_error(f"Missing required configuration file: {path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    report.add_error(f"{path.name} root must be a YAML mapping (dictionary)")
                    return None
                return data
        except yaml.YAMLError as exc:
            report.add_error(f"Syntax error parsing {path.name}: {exc}")
            return None
        except Exception as exc:
            report.add_error(f"Failed to read {path.name}: {exc}")
            return None

    def _validate_cameras(self, data: dict, report: ValidationReport) -> None:
        cameras = data.get("cameras")
        if not isinstance(cameras, list):
            report.add_error("cameras.yaml must contain a 'cameras' list")
            return

        seen_ids = set()
        for idx, cam in enumerate(cameras):
            prefix = f"cameras.yaml (camera #{idx + 1})"
            if not isinstance(cam, dict):
                report.add_error(f"{prefix}: entry must be a dictionary")
                continue

            cam_id = cam.get("id")
            if not cam_id or not isinstance(cam_id, str):
                report.add_error(f"{prefix}: missing or invalid 'id'")
            elif cam_id in seen_ids:
                report.add_error(f"{prefix}: duplicate camera id '{cam_id}'")
            else:
                seen_ids.add(cam_id)

            source = cam.get("source")
            if not isinstance(source, dict):
                report.add_error(f"{prefix} '{cam_id}': missing 'source' section")
            else:
                stype = source.get("type")
                if stype not in ["rtsp", "file", "usb"]:
                    report.add_error(f"{prefix} '{cam_id}': source type must be rtsp, file, or usb")

                if stype == "rtsp":
                    url = source.get("url", "")
                    if not url.startswith("rtsp://"):
                        report.add_warning(f"{prefix} '{cam_id}': RTSP url '{url}' does not start with rtsp://")
                elif stype == "file":
                    fpath = source.get("path")
                    if not fpath:
                        report.add_error(f"{prefix} '{cam_id}': source file missing 'path'")

            capture = cam.get("capture")
            if isinstance(capture, dict):
                fps = capture.get("target_fps", 10)
                if not isinstance(fps, (int, float)) or fps <= 0:
                    report.add_error(f"{prefix} '{cam_id}': capture target_fps must be positive number")

    def _validate_zones(self, data: dict, report: ValidationReport) -> None:
        cameras_map = data.get("zones") or data.get("cameras")
        if not isinstance(cameras_map, dict):
            report.add_error("zones.yaml must contain a top-level 'zones' or 'cameras' mapping")
            return

        for cam_key, cam_zones in cameras_map.items():
            prefix = f"zones.yaml (key '{cam_key}')"
            if not isinstance(cam_zones, dict):
                continue

            # Validate polygons
            polys = cam_zones.get("polygons", [])
            for p in polys:
                poly_id = p.get("id", "unnamed")
                pts = p.get("points", [])
                if not isinstance(pts, list) or len(pts) < 3:
                    report.add_error(f"{prefix} polygon '{poly_id}': requires at least 3 points")
                else:
                    for pt_idx, pt in enumerate(pts):
                        if not isinstance(pt, list) or len(pt) != 2:
                            report.add_error(f"{prefix} polygon '{poly_id}' point #{pt_idx}: must be [x, y]")

            # Validate tripwires
            tripwires = cam_zones.get("tripwires", [])
            for tw in tripwires:
                tw_id = tw.get("id", "unnamed")
                has_start_end = "start" in tw and "end" in tw
                has_points = "points" in tw

                if has_start_end:
                    start_pt = tw.get("start")
                    end_pt = tw.get("end")
                    if not isinstance(start_pt, list) or len(start_pt) != 2 or not isinstance(end_pt, list) or len(end_pt) != 2:
                        report.add_error(f"{prefix} tripwire '{tw_id}': start and end must each be [x, y]")
                elif has_points:
                    pts = tw.get("points")
                    if not isinstance(pts, list) or len(pts) != 2:
                        report.add_error(f"{prefix} tripwire '{tw_id}': must have exactly 2 points [[x1, y1], [x2, y2]]")
                else:
                    report.add_error(f"{prefix} tripwire '{tw_id}': must specify 'start'/'end' or 'points'")

                direction = tw.get("direction", "both")
                if direction not in VALID_TRIPWIRE_DIRECTIONS:
                    report.add_warning(f"{prefix} tripwire '{tw_id}': unrecognized direction '{direction}'")

    def _validate_rules(self, data: dict, report: ValidationReport) -> None:
        rules = data.get("rules")
        if not isinstance(rules, list):
            report.add_error("rules.yaml must contain a 'rules' list")
        else:
            seen_rule_ids = set()
            for idx, rule in enumerate(rules):
                prefix = f"rules.yaml (rule #{idx + 1})"
                rid = rule.get("id")
                if not rid or not isinstance(rid, str):
                    report.add_error(f"{prefix}: missing 'id'")
                elif rid in seen_rule_ids:
                    report.add_error(f"{prefix}: duplicate rule id '{rid}'")
                else:
                    seen_rule_ids.add(rid)

                ev_type = rule.get("event_type")
                if not ev_type or ev_type not in VALID_EVENT_TYPES:
                    report.add_error(f"{prefix} '{rid}': unknown event_type '{ev_type}'")

                sev = rule.get("severity")
                if not sev or sev not in VALID_SEVERITIES:
                    report.add_error(f"{prefix} '{rid}': unknown severity '{sev}'")

        # Validate webhooks
        webhooks = data.get("webhooks", [])
        for wh in webhooks:
            wh_id = wh.get("id", "unnamed")
            url = wh.get("url", "")
            parsed = urlparse(url)
            if not parsed.scheme or parsed.scheme not in ["http", "https"]:
                report.add_warning(f"rules.yaml webhook '{wh_id}': url '{url}' missing valid http/https scheme")

    def _validate_camera_zone_refs(self, cams_data: dict, zones_data: dict, report: ValidationReport) -> None:
        zone_keys = set((zones_data.get("zones") or zones_data.get("cameras") or {}).keys())
        for cam in cams_data.get("cameras", []):
            zone_ref = cam.get("zones_ref")
            if zone_ref and zone_ref not in zone_keys:
                report.add_warning(
                    f"Camera '{cam.get('id')}' references zones_ref '{zone_ref}', "
                    f"which is not found in zones.yaml"
                )
