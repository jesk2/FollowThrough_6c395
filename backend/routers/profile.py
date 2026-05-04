from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.dependencies import get_current_user, get_db
from backend.models import db as models
from backend.models.schemas import EmbeddingResponse, EmbeddingPoint, ProfileResponse

router = APIRouter(prefix="/profile", tags=["profile"])

DEVICE_LABELS = [
    "Salience Nudge",
    "Implementation Intention",
    "Planning Correction",
    "Virtual Stakes",
    "Precommitment Lock",
]


def _compute_14d_completion_rate(user: models.User, db: Session) -> float:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=14)
    checkins = (
        db.query(models.Checkin)
        .filter(models.Checkin.user_id == user.id, models.Checkin.checked_in_at >= cutoff)
        .all()
    )
    if not checkins:
        return 0.0
    return round(sum(c.completed for c in checkins) / len(checkins), 3)


def _get_drift_status(user: models.User, db: Session) -> str:
    recent_drift = (
        db.query(models.DriftEvent)
        .filter(models.DriftEvent.user_id == user.id)
        .order_by(models.DriftEvent.detected_at.desc())
        .first()
    )
    if recent_drift is None:
        return "stable"
    if recent_drift.direction == "improvement":
        return "improving"
    return "shifting"


def build_profile_response(user: models.User, db: Session) -> ProfileResponse:
    return ProfileResponse(
        id=user.id,
        email=user.email,
        beta_proxy=round(user.beta_proxy, 4),
        proj_bias_score=round(user.proj_bias_score, 4),
        current_device=user.current_device,
        device_label=DEVICE_LABELS[user.current_device],
        streak=user.streak,
        completion_rate_14d=_compute_14d_completion_rate(user, db),
        drift_status=_get_drift_status(user, db),
        created_at=user.created_at,
    )


@router.get("", response_model=ProfileResponse)
def get_profile(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return build_profile_response(current_user, db)


@router.get("/embedding", response_model=EmbeddingResponse)
def get_embedding(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """2D PCA projection of the user's embedding vs. anonymized population sample."""
    from backend.ml.cf_model import get_user_embedding, get_population_embeddings_2d
    import numpy as np

    user_emb = get_user_embedding(str(current_user.id))
    if user_emb is None:
        # return origin before the user has enough data
        return EmbeddingResponse(
            user=EmbeddingPoint(x=0.0, y=0.0),
            population=[],
        )

    user_2d, pop_2d = get_population_embeddings_2d(str(current_user.id), user_emb)

    return EmbeddingResponse(
        user=EmbeddingPoint(x=float(user_2d[0]), y=float(user_2d[1])),
        population=[EmbeddingPoint(x=float(p[0]), y=float(p[1])) for p in pop_2d],
    )
