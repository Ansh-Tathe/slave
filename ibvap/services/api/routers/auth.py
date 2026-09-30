"""
services/api/routers/auth.py
============================
IBVAP P7 — Authentication endpoints (JWT login, current user info).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from services.api.auth import authenticate_user, create_access_token, get_current_user
from services.api.models import TokenResponse, UserLogin, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin) -> TokenResponse:
    """Authenticate with username and password to receive a JWT Bearer token."""
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = {"sub": user["username"], "role": user["role"]}
    token = create_access_token(token_data)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        username=user["username"],
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)) -> UserResponse:
    """Get profile information for the authenticated user."""
    return UserResponse(
        username=current_user["username"],
        role=current_user["role"],
        is_active=current_user.get("is_active", True),
    )
