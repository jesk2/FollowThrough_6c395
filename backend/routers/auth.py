import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from supabase import create_client

from backend.config import settings
from backend.dependencies import get_db, get_current_user
from backend.models import db as models
from backend.models.schemas import AuthResponse, LoginRequest, ProfileResponse, SignupRequest
from backend.routers.profile import build_profile_response

router = APIRouter(prefix="/auth", tags=["auth"])

# Lazy: Supabase rejects empty/placeholder keys at construction time, which
# would break module import (and test collection) in environments without
# real credentials. Build the client on first use instead.
_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(settings.supabase_url, settings.supabase_service_key)
    return _supabase

DEVICE_LABELS = ["Salience Nudge", "Implementation Intention", "Planning Correction", "Virtual Stakes", "Precommitment Lock"]


@router.post("/signup", response_model=AuthResponse)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    try:
        resp = _get_supabase().auth.sign_up({"email": body.email, "password": body.password})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if resp.user is None:
        raise HTTPException(status_code=400, detail="Signup failed")

    # create User record with population prior
    user_id = uuid.UUID(resp.user.id)
    existing = db.query(models.User).filter(models.User.id == user_id).first()
    if not existing:
        user = models.User(id=user_id, email=body.email)
        db.add(user)
        db.commit()

    return AuthResponse(access_token=resp.session.access_token)


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    try:
        resp = _get_supabase().auth.sign_in_with_password({"email": body.email, "password": body.password})
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    if resp.session is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return AuthResponse(access_token=resp.session.access_token)


@router.get("/me", response_model=ProfileResponse)
def me(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return build_profile_response(current_user, db)
