"""
services/reports/generator.py
=============================
IBVAP P9 — Automated Security Incident Report Generator.
Synthesizes verified surveillance events into formal, multi-format
incident reports (HTML, Markdown, JSON) for command personnel.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from services.api.models import EventResponse


@dataclass
class IncidentReport:
    report_id: str
    title: str
    generated_at: str
    investigating_officer: str
    facility: str
    severity: str
    status: str  # CONFIRMED, UNDER_REVIEW, REJECTED
    summary: str
    primary_event: Dict[str, Any]
    correlated_events: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IncidentReportGenerator:
    """Compiles multi-sensor surveillance events into formal security dossiers."""

    def __init__(self, output_dir: str = "data/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        primary_event: EventResponse,
        correlated_events: Optional[List[EventResponse]] = None,
        investigating_officer: str = "Surveillance Operator (SOC-1)",
        facility: str = "Sector 4 Command & Surveillance Station",
    ) -> IncidentReport:
        correlated = correlated_events or []
        all_events = [primary_event] + correlated
        all_events.sort(key=lambda x: x.timestamp)

        ev_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(primary_event.timestamp))
        gen_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        rep_id = f"INC-{time.strftime('%Y%m%d')}-{primary_event.event_id[:8].upper()}"

        # Status determination
        if primary_event.confirmed is True:
            status = "CONFIRMED BREACH"
        elif primary_event.confirmed is False:
            status = "REJECTED (FALSE POSITIVE)"
        else:
            status = "PENDING REVIEW"

        # Automated narrative summary
        summary = (
            f"On {ev_time}, automated surveillance system IBVAP flagged a {primary_event.severity} severity "
            f"alert ({primary_event.event_type}) on camera '{primary_event.camera_id}'. "
            f"Target identity Track #{primary_event.track_id} (classified as {primary_event.class_name.upper()}, "
            f"confidence {int(primary_event.confidence * 100)}%) triggered spatial boundary "
            f"'{primary_event.zone_id or 'General Sector'}'. "
        )
        if primary_event.metadata.get("operator_notes"):
            summary += f"Operator Review: \"{primary_event.metadata['operator_notes']}\". "
        if correlated:
            summary += f"A total of {len(correlated)} correlated telemetry events were registered along this track."

        # Chronological timeline
        timeline = []
        for ev in all_events:
            t_str = time.strftime("%H:%M:%S", time.gmtime(ev.timestamp))
            timeline.append({
                "time": t_str,
                "timestamp": ev.timestamp,
                "camera_id": ev.camera_id,
                "event_type": ev.event_type,
                "severity": ev.severity,
                "details": f"{ev.class_name.upper()} (Track #{ev.track_id}) at zone '{ev.zone_id or 'N/A'}'",
            })

        # Evidence dossier
        evidence = {
            "primary_snapshot": primary_event.snapshot_path,
            "target_bbox": primary_event.bbox,
            "target_confidence": primary_event.confidence,
            "metadata_attributes": primary_event.metadata,
        }

        # Operational recommendations
        recs = []
        if primary_event.severity in ["CRITICAL", "HIGH"]:
            recs.append("Dispatch Quick Reaction Team (QRT) to inspect boundary.")
            recs.append("Preserve 60-second video evidence buffer in secure MinIO vault.")
        if "ANPR" in primary_event.event_type:
            plate = primary_event.metadata.get("plate", "UNKNOWN")
            recs.append(f"Flag vehicle registration '{plate}' across inter-agency check posts.")
        if "FACE" in primary_event.event_type:
            recs.append("Forward facial biometric crop to Central Command Database for identity verification.")
        recs.append("Log operator audit verification into immutable security register.")

        report = IncidentReport(
            report_id=rep_id,
            title=f"SURVEILLANCE INCIDENT: {primary_event.event_type} on {primary_event.camera_id}",
            generated_at=gen_time,
            investigating_officer=investigating_officer,
            facility=facility,
            severity=primary_event.severity,
            status=status,
            summary=summary,
            primary_event=primary_event.model_dump(),
            correlated_events=[e.model_dump() for e in correlated],
            timeline=timeline,
            evidence=evidence,
            recommendations=recs,
        )

        return report

    def export_all(self, report: IncidentReport) -> Dict[str, Path]:
        """Save report in JSON, Markdown, and print-ready HTML formats."""
        base_name = f"incident_{report.report_id.lower().replace('-', '_')}"
        json_path = self.output_dir / f"{base_name}.json"
        md_path = self.output_dir / f"{base_name}.md"
        html_path = self.output_dir / f"{base_name}.html"

        # 1. JSON
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        # 2. Markdown
        md_content = self._render_markdown(report)
        md_path.write_text(md_content, encoding="utf-8")

        # 3. HTML
        html_content = self._render_html(report)
        html_path.write_text(html_content, encoding="utf-8")

        logger.info(f"Incident report exported: {json_path.name}, {md_path.name}, {html_path.name}")
        return {"json": json_path, "md": md_path, "html": html_path}

    def _render_markdown(self, r: IncidentReport) -> str:
        md = f"""# {r.title}
**Incident ID:** `{r.report_id}` | **Status:** `{r.status}` | **Severity:** `{r.severity}`  
**Facility:** {r.facility}  
**Investigating Officer:** {r.investigating_officer}  
**Generated Date:** {r.generated_at}  

---

## 1. Executive Summary
{r.summary}

---

## 2. Event Timeline
| Time (UTC) | Camera | Event Type | Severity | Description |
|---|---|---|---|---|
"""
        for item in r.timeline:
            md += f"| {item['time']} | `{item['camera_id']}` | `{item['event_type']}` | {item['severity']} | {item['details']} |\n"

        md += f"""
---

## 3. Evidence Dossier
- **Primary Event UUID:** `{r.primary_event.get('event_id')}`
- **Track Class & ID:** `{r.primary_event.get('class_name')}` (Track #{r.primary_event.get('track_id')})
- **Detection Confidence:** {int(r.primary_event.get('confidence', 0) * 100)}%
- **Snapshot File:** `{r.evidence.get('primary_snapshot') or 'Stored in object vault'}`
- **Metadata Attributes:**
```json
{json.dumps(r.evidence.get('metadata_attributes', {}), indent=2)}
```

---

## 4. Operational Recommendations
"""
        for rec in r.recommendations:
            md += f"- [ ] {rec}\n"

        return md

    def _render_html(self, r: IncidentReport) -> str:
        timeline_rows = "".join(
            f"<tr><td>{t['time']}</td><td><code>{t['camera_id']}</code></td><td>{t['event_type']}</td><td><span class='badge badge-{t['severity'].lower()}'>{t['severity']}</span></td><td>{t['details']}</td></tr>"
            for t in r.timeline
        )
        rec_items = "".join(f"<li>{rec}</li>" for rec in r.recommendations)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{r.report_id} — {r.title}</title>
  <style>
    body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; margin: 40px; color: #1e293b; background: #fff; line-height: 1.6; }}
    .header {{ border-bottom: 2px solid #0284c7; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-start; }}
    .title {{ font-size: 24px; font-weight: 700; color: #0f172a; margin-bottom: 4px; }}
    .subtitle {{ font-size: 13px; color: #64748b; font-family: monospace; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; text-transform: uppercase; font-family: monospace; }}
    .badge-critical {{ background: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }}
    .badge-high {{ background: #ffedd5; color: #c2410c; border: 1px solid #fb923c; }}
    .badge-medium {{ background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }}
    .badge-low {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 30px; }}
    .meta-item label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; display: block; }}
    .meta-item span {{ font-size: 14px; font-weight: 600; color: #0f172a; }}
    h2 {{ font-size: 18px; color: #0f172a; border-left: 4px solid #0284c7; padding-left: 10px; margin-top: 30px; margin-bottom: 15px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 10px 12px; text-align: left; }}
    th {{ background: #f1f5f9; font-weight: 600; }}
    pre {{ background: #0f172a; color: #f8fafc; padding: 14px; border-radius: 6px; font-size: 12px; overflow-x: auto; }}
    ul {{ margin-left: 20px; }}
    .footer {{ margin-top: 50px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">{r.title}</div>
      <div class="subtitle">INTELLIGENT BORDER VIDEO ANALYTICS PLATFORM • OFFICIAL INCIDENT DOSSIER</div>
    </div>
    <div>
      <span class="badge badge-{r.severity.lower()}">{r.severity}</span>
    </div>
  </div>

  <div class="meta-grid">
    <div class="meta-item"><label>Incident ID</label><span>{r.report_id}</span></div>
    <div class="meta-item"><label>Status</label><span>{r.status}</span></div>
    <div class="meta-item"><label>Facility</label><span>{r.facility}</span></div>
    <div class="meta-item"><label>Officer</label><span>{r.investigating_officer}</span></div>
  </div>

  <h2>1. Executive Summary</h2>
  <p>{r.summary}</p>

  <h2>2. Chronological Timeline</h2>
  <table>
    <thead>
      <tr><th>Time (UTC)</th><th>Camera</th><th>Event Type</th><th>Severity</th><th>Details</th></tr>
    </thead>
    <tbody>
      {timeline_rows}
    </tbody>
  </table>

  <h2>3. Evidence & Metadata</h2>
  <pre>{json.dumps(r.evidence, indent=2)}</pre>

  <h2>4. Operational Directives</h2>
  <ul>{rec_items}</ul>

  <div class="footer">
    IBVAP Surveillance Dossier • Generated {r.generated_at} • Classification: RESTRICTED SECURITY EVIDENCE
  </div>
</body>
</html>
"""
