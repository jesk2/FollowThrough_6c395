import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import notifications as notif_service
from backend.dependencies import get_current_user, get_db
from backend.models import db as models
from backend.models.schemas import CheckinCreate, CheckinHistoryResponse, CheckinResponse
from backend.ml import process_checkin, run_reevaluation, on_device_change

try:
    from ml.inference.inference_api import (
        TaskFeatures,
        get_user_state,
        incremental_update_api,
    )
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False

_CATEGORY_INDEX = {"academic": 0, "exercise": 1, "work": 2, "personal": 3}
_DEADLINE_INDEX = {"today": 0, "this_week": 1, "none": 2}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/checkins", tags=["checkins"])


def _apply_checkin_actions(user: models.User, actions, db: Session) -> None:
    """Mutate ``user`` based on the descriptors returned by process_checkin."""
    if actions.proj_bias_update is not None:
        user.proj_bias_score = actions.proj_bias_update

    if actions.streak_set_to is not None:
        user.streak = actions.streak_set_to
    elif actions.streak_delta:
        user.streak = max(0, user.streak + actions.streak_delta)

    for note in actions.notifications:
        notif_service.send_notification(user.email, note)


def _refresh_cf_and_beta(user: models.User, task: models.Task, completed: float) -> None:
    """Best-effort CF embedding + beta probe refresh; failures are logged only."""
    if not _ML_AVAILABLE or user.embedding_id is None:
        return
    try:
        days_until = max(
            0,
            (task.planned_start - datetime.now(timezone.utc).replace(tzinfo=None)).days,
        )
        features = TaskFeatures(
            difficulty=float(task.difficulty - 1) / 4.0,
            category_index=_CATEGORY_INDEX.get(task.category, 0),
            planned_duration_minutes=float(task.planned_duration),
            days_until_planned_start=days_until,
            deadline_pressure_index=_DEADLINE_INDEX.get(task.deadline_pressure, 2),
        )
        incremental_update_api(user.embedding_id, features, float(completed))
        user.beta_proxy = get_user_state(user.embedding_id)
    except Exception as exc:
        logger.warning("CF/beta update failed for user %s: %s", user.id, exc)


@router.post("", response_model=CheckinResponse, status_code=201)
def submit_checkin(
    body: CheckinCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = (
        db.query(models.Task)
        .filter(models.Task.id == body.task_id, models.Task.user_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if db.query(models.Checkin).filter(models.Checkin.task_id == task.id).first():
        raise HTTPException(status_code=409, detail="Check-in already submitted for this task")

    checkin = models.Checkin(
        task_id=task.id,
        user_id=current_user.id,
        completed=body.completed,
        actual_duration=body.actual_duration,
        failure_reason=body.failure_reason,
    )
    db.add(checkin)
    db.flush()  # so the new check-in is visible to today's-rate query

    actions = process_checkin(current_user, task, body, db)
    _apply_checkin_actions(current_user, actions, db)

    _refresh_cf_and_beta(current_user, task, body.completed)

    # Always re-evaluate: the recommender's failure-streak, beta-baseline, and
    # projection-bias paths must fire on every check-in, not only on confirmed
    # drift. ``actions.trigger_reevaluation`` still indicates *why* the call
    # happened (drift event vs. routine), used to pick the notification tone.
    old_device = current_user.current_device
    result = run_reevaluation(current_user, db)
    if result.changed:
        note = on_device_change(current_user, old_device, result.recommended_device)
        notif_service.send_notification(current_user.email, note)

    db.commit()
    db.refresh(checkin)
    return checkin


@router.get("/history", response_model=CheckinHistoryResponse)
def checkin_history(
    page: int = 1,
    page_size: int = 20,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    offset = (page - 1) * page_size
    total = db.query(models.Checkin).filter(models.Checkin.user_id == current_user.id).count()
    checkins = (
        db.query(models.Checkin)
        .filter(models.Checkin.user_id == current_user.id)
        .order_by(models.Checkin.checked_in_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return CheckinHistoryResponse(
        checkins=checkins,
        total=total,
        page=page,
        page_size=page_size,
    )
