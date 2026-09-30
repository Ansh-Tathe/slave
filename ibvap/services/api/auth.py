"""
services/api/auth.py
====================
IBVAP P7 — JWT authentication, password hashing, API key verification, and RBAC.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from loguru import logger

from services.api.models import TokenResponse, UserResponse

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY: str = os.getenv("IBVAP_SECRET_KEY", "ibvap-super-secret-jwt-key-for-surveillance-2026")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS: int = 86400  # 24 hours
DEFAULT_API_KEY: str = os.getenv("IBVAP_API_KEY", "ibvap-c2-secure-api-key-default")

# Role hierarchy
ROLE_RANKS = {
    "viewer": 1,
    "operator": 2,
    "admin": 3,
}

# In-memory user database (initial seeds)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

USERS_DB: Dict[str, dict] = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "is_active": True,
    },
    "operator": {
        "username": "operator",
        "hashed_password": pwd_context.hash("operator123"),
        "role": "operator",
        "is_active": True,
    },
    "viewer": {
        "username": "viewer",
        "hashed_password": pwd_context.hash("viewer123"),
        "role": "viewer",
        "is_active": True,
    },
}

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Password & Token Helpers ──────────────────────────────────────────────────

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta_s: Optional[int] = None) -> str:
    to_encode = data.copy()
    expire = time.time() + (expires_delta_s if expires_delta_s is not None else ACCESS_TOKEN_EXPIRE_SECONDS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


# ── FastAPI Dependencies ──────────────────────────────────────────────────────

async def get_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> dict:
    """
    Validates either a Bearer JWT token or an X-API-Key header.
    """
    # 1. Check API Key
    if api_key:
        if api_key == DEFAULT_API_KEY or api_key == os.getenv("C2_API_KEY", DEFAULT_API_KEY):
            return {
                "username": "c2_system",
                "role": "admin",
                "is_active": True,
            }
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # 2. Check Bearer Token
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token or X-API-Key header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        role: Optional[str] = payload.get("role")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload missing subject",
            )
        user = USERS_DB.get(username)
        if user:
            return user
        return {
            "username": username,
            "role": role or "operator",
            "is_active": True,
        }
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(exc)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(min_role: str):
    """
    Enforces minimum role required to access an endpoint.
    """
    min_rank = ROLE_RANKS.get(min_role, 1)

    async def _role_checker(user: dict = Depends(get_current_user)) -> dict:
        user_role = user.get("role", "viewer")
        user_rank = ROLE_RANKS.get(user_role, 0)
        if user_rank < min_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: requires minimum role '{min_role}' (you have '{user_role}')",
            )
        return user

    return _role_checker
