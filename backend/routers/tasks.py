from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.dependencies import get_current_user, get_db
from backend.models import db as models
from backend.models.schemas import (
    RescheduleRequest,
    RescheduleResponseSchema,
    TaskCreate,
    TaskListResponse,
    TaskResponse,
)
from backend.ml import process_reschedule_attempt, process_task_creation

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _task_to_response(task: models.Task, db: Session) -> TaskResponse:
    has_checkin = db.query(models.Checkin).filter(models.Checkin.task_id == task.id).first() is not None
    return TaskResponse(
        **{c.name: getattr(task, c.name) for c in task.__table__.columns},
        has_checkin=has_checkin,
    )


@router.post("", response_model=TaskResponse, status_code=201)
def create_task(
    body: TaskCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Level 1: implementation-intention fields are required
    if current_user.current_device >= 1:
        if not body.impl_where or not body.impl_what_first:
            raise HTTPException(
                status_code=422,
                detail="impl_where and impl_what_first are required at your current commitment level",
            )

    mods = process_task_creation(current_user, body)

    task = models.Task(
        user_id=current_user.id,
        name=body.name,
        category=body.category,
        difficulty=body.difficulty,
        deadline_pressure=body.deadline_pressure,
        planned_start=body.planned_start,
        planned_duration=body.planned_duration,
        corrected_duration=mods.corrected_duration,
        locked=mods.locked,
        impl_where=body.impl_where,
        impl_what_first=body.impl_what_first,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return _task_to_response(task, db)


@router.get("", response_model=TaskListResponse)
def list_tasks(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    tasks = (
        db.query(models.Task)
        .filter(models.Task.user_id == current_user.id, models.Task.is_active.is_(True))
        .order_by(models.Task.planned_start)
        .all()
    )
    return TaskListResponse(
        tasks=[_task_to_response(t, db) for t in tasks],
        total=len(tasks),
    )


@router.get("/pending", response_model=TaskListResponse)
def pending_tasks(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Tasks past their planned start time with no check-in yet."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    checked_in_task_ids = (
        db.query(models.Checkin.task_id)
        .filter(models.Checkin.user_id == current_user.id)
        .subquery()
    )
    tasks = (
        db.query(models.Task)
        .filter(
            models.Task.user_id == current_user.id,
            models.Task.is_active.is_(True),
            models.Task.planned_start <= now,
            models.Task.id.not_in(checked_in_task_ids),
        )
        .order_by(models.Task.planned_start)
        .all()
    )
    return TaskListResponse(
        tasks=[_task_to_response(t, db) for t in tasks],
        total=len(tasks),
    )


@router.put("/{task_id}/reschedule", response_model=RescheduleResponseSchema)
def reschedule_task(
    task_id: UUID,
    body: RescheduleRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Reschedule a task. Locked tasks within 24h require a confirmation token."""
    task = (
        db.query(models.Task)
        .filter(models.Task.id == task_id, models.Task.user_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    decision = process_reschedule_attempt(
        current_user, task, body.new_start, db, confirmation_token=body.confirmation_token
    )

    if not decision.allowed and decision.requires_confirmation:
        # 409 carries the confirmation token; client retries with it
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "requires_confirmation": True,
                "confirmation_token": decision.confirmation_token,
                "reason": decision.reason,
            },
        )
    if not decision.allowed:
        raise HTTPException(status_code=400, detail=decision.reason or "Reschedule denied")

    task.planned_start = body.new_start
    db.commit()
    db.refresh(task)
    return RescheduleResponseSchema(
        allowed=True,
        requires_confirmation=False,
        task=_task_to_response(task, db),
    )


@router.delete("/{task_id}", status_code=204)
def delete_task(
    task_id: UUID,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = (
        db.query(models.Task)
        .filter(models.Task.id == task_id, models.Task.user_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.is_active = False
    db.commit()
