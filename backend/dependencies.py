from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from supabase import Client, create_client
import uuid

from backend.config import settings
from backend.database import SessionLocal
from backend.models import db as models

security = HTTPBearer()

_supabase: Client | None = None


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(settings.supabase_url, settings.supabase_service_key)
    return _supabase


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> models.User:
    token = credentials.credentials
    try:
        response = _get_supabase().auth.get_user(token)
        user_data = response.user
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = str(user_data.id)
        email = user_data.email or ""
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(models.User).filter(models.User.id == uuid.UUID(user_id)).first()
    if user is None:
        user = models.User(id=uuid.UUID(user_id), email=email)
        db.add(user)
        db.commit()
        db.refresh(user)

    return user
