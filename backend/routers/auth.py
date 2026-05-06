import uuid

import httpx

try:
    from ml.inference.inference_api import initialize_new_user as _init_user
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.config import settings
from backend.dependencies import get_db, get_current_user
from backend.models import db as models
from backend.models.schemas import AuthResponse, LoginRequest, ProfileResponse, SignupRequest
from backend.routers.profile import build_profile_response

router = APIRouter(prefix="/auth", tags=["auth"])

_ANON_HEADERS = {
    "Content-Type": "application/json",
}


def _anon_headers() -> dict:
    return {"apikey": settings.supabase_anon_key, "Content-Type": "application/json"}


@router.post("/signup", response_model=AuthResponse)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    resp = httpx.post(
        f"{settings.supabase_url}/auth/v1/signup",
        headers=_anon_headers(),
        json={"email": body.email, "password": body.password},
    )
    data = resp.json()
    if resp.status_code != 200 or not data.get("id"):
        raise HTTPException(status_code=400, detail=data.get("msg", "Signup failed"))

    user_id = uuid.UUID(data["id"])
    existing = db.query(models.User).filter(models.User.id == user_id).first()
    if not existing:
        embedding_id = _init_user() if _ML_AVAILABLE else None
        db.add(models.User(id=user_id, email=body.email, embedding_id=embedding_id))
        db.commit()

    # session is None until email is confirmed — return empty token in that case
    session = data.get("session") or {}
    return AuthResponse(access_token=session.get("access_token", ""))


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    resp = httpx.post(
        f"{settings.supabase_url}/auth/v1/token?grant_type=password",
        headers=_anon_headers(),
        json={"email": body.email, "password": body.password},
    )
    data = resp.json()
    if resp.status_code != 200 or not data.get("access_token"):
        raise HTTPException(status_code=401, detail=data.get("error_description", "Invalid credentials"))

    return AuthResponse(access_token=data["access_token"])


@router.get("/me", response_model=ProfileResponse)
def me(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return build_profile_response(current_user, db)
