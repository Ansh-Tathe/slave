"""
services/api/routers/watchlist.py
=================================
IBVAP P7 — Watchlist management for license plates (ANPR) and persons (Face).
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from services.api.auth import require_role
from services.api.models import (
    WatchlistPersonCreate,
    WatchlistPersonResponse,
    WatchlistPlateCreate,
    WatchlistPlateResponse,
)
from services.api.state import state

router = APIRouter(prefix="/watchlist", tags=["Watchlist"])


# ── Vehicle Plates ────────────────────────────────────────────────────────────

@router.get("/plates", response_model=List[WatchlistPlateResponse])
async def list_plates() -> List[WatchlistPlateResponse]:
    """List all license plates currently on the ANPR watchlist."""
    return list(state.watchlist_store.plates.values())


@router.post("/plates", response_model=WatchlistPlateResponse, status_code=status.HTTP_201_CREATED)
async def add_plate(
    body: WatchlistPlateCreate,
    _user: dict = Depends(require_role("operator")),
) -> WatchlistPlateResponse:
    """Add a license plate to the watchlist. Requires operator role."""
    return state.watchlist_store.add_plate(
        plate=body.plate,
        notes=body.notes,
        severity=body.severity,
    )


@router.delete("/plates/{plate}", status_code=status.HTTP_200_OK)
async def remove_plate(
    plate: str,
    _user: dict = Depends(require_role("operator")),
):
    """Remove a license plate from the watchlist."""
    removed = state.watchlist_store.remove_plate(plate)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Plate '{plate}' not found in watchlist")
    return {"message": f"Plate '{plate}' removed successfully"}


# ── Persons of Interest ───────────────────────────────────────────────────────

@router.get("/faces", response_model=List[WatchlistPersonResponse])
async def list_persons() -> List[WatchlistPersonResponse]:
    """List all enrolled persons on the face recognition watchlist."""
    return list(state.watchlist_store.persons.values())


@router.post("/faces", response_model=WatchlistPersonResponse, status_code=status.HTTP_201_CREATED)
async def add_person(
    body: WatchlistPersonCreate,
    _user: dict = Depends(require_role("operator")),
) -> WatchlistPersonResponse:
    """Enroll a person into the watchlist. Requires operator role."""
    return state.watchlist_store.add_person(
        name=body.name,
        notes=body.notes,
        severity=body.severity,
    )


@router.delete("/faces/{name}", status_code=status.HTTP_200_OK)
async def remove_person(
    name: str,
    _user: dict = Depends(require_role("operator")),
):
    """Remove an enrolled person from the watchlist."""
    removed = state.watchlist_store.remove_person(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Person '{name}' not found in watchlist")
    return {"message": f"Person '{name}' removed successfully"}
