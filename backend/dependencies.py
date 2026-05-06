import uuid

import httpx

try:
    from ml.inference.inference_api import initialize_new_user as _init_user
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import SessionLocal
from backend.models import db as models

security = HTTPBearer()


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
    resp = httpx.get(
        f"{settings.supabase_url}/auth/v1/user",
        headers={
            "apikey": settings.supabase_service_key,
            "Authorization": f"Bearer {token}",
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid token")

    data = resp.json()
    user_id = data.get("id")
    email = data.get("email", "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(models.User).filter(models.User.id == uuid.UUID(user_id)).first()
    if user is None:
        embedding_id = _init_user() if _ML_AVAILABLE else None
        user = models.User(id=uuid.UUID(user_id), email=email, embedding_id=embedding_id)
        db.add(user)
        db.commit()
        db.refresh(user)

    return user
