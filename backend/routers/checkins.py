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
from backend.ml.cf_model import get_user_embedding
from backend.ml.features import encode_task
from backend.ml.probe import get_beta_proxy
from backend.ml.train import incremental_update

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
    try:
        days_until = max(
            0,
            (task.planned_start - datetime.now(timezone.utc).replace(tzinfo=None)).days,
        )
        task_features = encode_task(
            category=task.category,
            deadline_pressure=task.deadline_pressure,
            difficulty=task.difficulty,
            planned_duration=task.planned_duration,
            days_until=days_until,
        )
        incremental_update(
            user_id=str(user.id),
            task_features=task_features,
            completed=completed,
        )
        embedding = get_user_embedding(str(user.id))
        if embedding is not None:
            user.beta_proxy = float(get_beta_proxy(embedding))
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
