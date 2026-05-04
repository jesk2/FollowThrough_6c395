from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.dependencies import get_current_user, get_db
from backend.models import db as models
from backend.models.schemas import TaskCreate, TaskListResponse, TaskResponse
from backend.ml.recommender import apply_projection_correction

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
    # Level 1 — implementation intention fields are required
    if current_user.current_device >= 1:
        if not body.impl_where or not body.impl_what_first:
            raise HTTPException(
                status_code=422,
                detail="impl_where and impl_what_first are required at your current commitment level",
            )

    # Level 2 — apply projection bias correction to planned duration
    corrected_duration = None
    if current_user.current_device >= 2 and current_user.proj_bias_score > 0.3:
        corrected_duration = apply_projection_correction(
            body.planned_duration, current_user.proj_bias_score
        )

    task = models.Task(
        user_id=current_user.id,
        name=body.name,
        category=body.category,
        difficulty=body.difficulty,
        deadline_pressure=body.deadline_pressure,
        planned_start=body.planned_start,
        planned_duration=body.planned_duration,
        corrected_duration=corrected_duration,
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
