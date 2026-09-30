"""
services/api/routers/reports.py
===============================
IBVAP P9 — Security Incident Report Generation REST Endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from services.api.auth import require_role
from services.api.models import EventResponse
from services.api.state import state
from services.reports.generator import IncidentReport, IncidentReportGenerator

router = APIRouter(prefix="/reports", tags=["Incident Reports"])
report_generator = IncidentReportGenerator()

# Cache generated reports in memory
REPORTS_CACHE: Dict[str, IncidentReport] = {}


class GenerateReportRequest(BaseModel):
    event_id: str
    investigating_officer: Optional[str] = "Surveillance Operator"
    facility: Optional[str] = "Sector 4 Command Station"


class ReportSummaryResponse(BaseModel):
    report_id: str
    title: str
    generated_at: str
    severity: str
    status: str
    html_url: str
    summary: str


@router.post("/incident", response_model=ReportSummaryResponse, status_code=status.HTTP_201_CREATED)
async def generate_incident_report(
    body: GenerateReportRequest,
    current_user: dict = Depends(require_role("operator")),
) -> ReportSummaryResponse:
    """Generate a formal security incident dossier for a specific surveillance alert."""
    ev = state.event_store.get(body.event_id)
    if not ev:
        raise HTTPException(status_code=404, detail=f"Event {body.event_id} not found")

    # Find correlated events (same camera within 5 minutes or same track_id)
    all_events = list(state.event_store._events_list)
    correlated = [
        e for e in all_events
        if e.event_id != ev.event_id and (
            e.track_id == ev.track_id or (e.camera_id == ev.camera_id and abs(e.timestamp - ev.timestamp) <= 300)
        )
    ]

    officer = body.investigating_officer or f"Officer {current_user['username'].capitalize()}"
    report = report_generator.generate(
        primary_event=ev,
        correlated_events=correlated,
        investigating_officer=officer,
        facility=body.facility or "Command & Surveillance Sector 4",
    )

    report_generator.export_all(report)
    REPORTS_CACHE[report.report_id] = report

    return ReportSummaryResponse(
        report_id=report.report_id,
        title=report.title,
        generated_at=report.generated_at,
        severity=report.severity,
        status=report.status,
        html_url=f"/api/v1/reports/{report.report_id}/html",
        summary=report.summary,
    )


@router.get("/{report_id}")
async def get_report_json(report_id: str) -> Dict[str, Any]:
    """Retrieve full incident dossier in JSON format."""
    rep = REPORTS_CACHE.get(report_id)
    if not rep:
        # Check disk
        path = report_generator.output_dir / f"incident_{report_id.lower().replace('-', '_')}.json"
        if path.exists():
            import json
            return json.loads(path.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return rep.to_dict()


@router.get("/{report_id}/html", response_class=HTMLResponse)
async def view_report_html(report_id: str) -> HTMLResponse:
    """Render print-ready incident dossier in HTML."""
    rep = REPORTS_CACHE.get(report_id)
    if rep:
        return HTMLResponse(report_generator._render_html(rep))

    path = report_generator.output_dir / f"incident_{report_id.lower().replace('-', '_')}.html"
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8"))

    raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
