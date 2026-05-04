from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.models import db as models
from backend.models.schemas import ReminderRequest
from backend import notifications as notif_service

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.post("/checkin-reminder", status_code=204)
def send_checkin_reminder(
    body: ReminderRequest,
    db: Session = Depends(get_db),
):
    """Internal endpoint called by the APScheduler job. Not authenticated via JWT."""
    user = db.query(models.User).filter(models.User.id == body.user_id).first()
    task = db.query(models.Task).filter(models.Task.id == body.task_id).first()
    if not user or not task:
        raise HTTPException(status_code=404, detail="User or task not found")

    notif_service.send_reminder(user.email, task.name, task.planned_start)
