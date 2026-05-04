from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.dependencies import get_current_user, get_db
from backend.models import db as models
from backend.models.schemas import CheckinCreate, CheckinHistoryResponse, CheckinResponse
from backend.ml.train import incremental_update
from backend.ml.bocd import get_or_create_detector
from backend.ml.recommender import evaluate_device_level
from backend.ml.probe import get_beta_proxy
from backend.ml.cf_model import get_user_embedding

router = APIRouter(prefix="/checkins", tags=["checkins"])

DEVICE_LABELS = ["Salience Nudge", "Implementation Intention", "Planning Correction", "Virtual Stakes", "Precommitment Lock"]


def _compute_14d_completion_rate(user_id: UUID, db: Session) -> float:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=14)
    checkins = (
        db.query(models.Checkin)
        .filter(models.Checkin.user_id == user_id, models.Checkin.checked_in_at >= cutoff)
        .all()
    )
    if not checkins:
        return 0.0
    return sum(c.completed for c in checkins) / len(checkins)


def _count_consecutive_failures(user_id: UUID, db: Session) -> int:
    recent = (
        db.query(models.Checkin)
        .filter(models.Checkin.user_id == user_id)
        .order_by(models.Checkin.checked_in_at.desc())
        .limit(10)
        .all()
    )
    streak = 0
    for c in recent:
        if c.completed == 0.0:
            streak += 1
        else:
            break
    return streak


def _update_proj_bias(user: models.User, task: models.Task, actual_duration: int, db: Session):
    if task.planned_duration and task.planned_duration > 0:
        overrun = (actual_duration - task.planned_duration) / task.planned_duration
        # exponential moving average with alpha=0.2
        user.proj_bias_score = 0.8 * user.proj_bias_score + 0.2 * overrun


def _assign_device(user: models.User, new_device: int, db: Session):
    if new_device == user.current_device:
        return
    rate_14d = _compute_14d_completion_rate(user.id, db)
    assignment = models.DeviceAssignment(
        user_id=user.id,
        device_type=new_device,
        beta_at_assignment=user.beta_proxy,
        pre_completion_rate=rate_14d,
    )
    db.add(assignment)
    user.current_device = new_device


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

    existing = db.query(models.Checkin).filter(models.Checkin.task_id == task.id).first()
    if existing:
        raise HTTPException(status_code=409, detail="Check-in already submitted for this task")

    checkin = models.Checkin(
        task_id=task.id,
        user_id=current_user.id,
        completed=body.completed,
        actual_duration=body.actual_duration,
        failure_reason=body.failure_reason,
    )
    db.add(checkin)

    # update proj_bias_score if task was completed and we have duration data
    if body.completed > 0 and body.actual_duration:
        _update_proj_bias(current_user, task, body.actual_duration, db)

    # streak update
    if body.completed == 1.0:
        current_user.streak += 1
    elif body.completed == 0.0:
        current_user.streak = 0

    db.commit()
    db.refresh(checkin)

    # --- ML updates (non-blocking; log errors rather than failing the request) ---
    try:
        from backend.ml.features import encode_task
        from datetime import timezone as tz

        days_until = max(0, (task.planned_start - datetime.now(timezone.utc).replace(tzinfo=None)).days)
        task_features = encode_task(
            category=task.category,
            deadline_pressure=task.deadline_pressure,
            difficulty=task.difficulty,
            planned_duration=task.planned_duration,
            days_until=days_until,
        )

        incremental_update(
            user_id=str(current_user.id),
            task_features=task_features,
            completed=body.completed,
        )

        embedding = get_user_embedding(str(current_user.id))
        if embedding is not None:
            new_beta = get_beta_proxy(embedding)
            current_user.beta_proxy = float(new_beta)

        # BOCD update using today's daily completion rate
        today = datetime.now(timezone.utc).date()
        today_start = datetime.combine(today, datetime.min.time())
        today_checkins = (
            db.query(models.Checkin)
            .filter(
                models.Checkin.user_id == current_user.id,
                models.Checkin.checked_in_at >= today_start,
            )
            .all()
        )
        daily_rate = sum(c.completed for c in today_checkins) / max(len(today_checkins), 1)

        detector = get_or_create_detector(str(current_user.id))
        cp_prob = detector.update(daily_rate)

        if cp_prob > 0.5:
            failure_streak = _count_consecutive_failures(current_user.id, db)
            new_device = evaluate_device_level(
                beta_proxy=current_user.beta_proxy,
                proj_bias_score=current_user.proj_bias_score,
                drift_flag=True,
                failure_streak=failure_streak,
            )
            _assign_device(current_user, new_device, db)
        else:
            failure_streak = _count_consecutive_failures(current_user.id, db)
            new_device = evaluate_device_level(
                beta_proxy=current_user.beta_proxy,
                proj_bias_score=current_user.proj_bias_score,
                drift_flag=False,
                failure_streak=failure_streak,
            )
            _assign_device(current_user, new_device, db)

        db.commit()
    except Exception as exc:
        # ML errors must not break check-in submission
        import logging
        logging.getLogger(__name__).warning("ML update failed: %s", exc)

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
