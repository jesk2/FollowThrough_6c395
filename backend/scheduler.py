"""
APScheduler job definitions.

Two jobs:
  1. checkin_reminder_job — runs every hour.
     Finds tasks where planned_start is within the next 2 hours with no check-in.
     Sends a reminder notification for each.

  2. device_reeval_job — runs every Monday at midnight.
     For each user: compute 14-day completion rate, run de-escalation check,
     run BOCD weekly summary, log device changes.
"""
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.database import SessionLocal
from backend.models import db as models
from backend import notifications

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


@scheduler.scheduled_job(IntervalTrigger(hours=1), id="checkin_reminder")
def checkin_reminder_job():
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        window_end = now + timedelta(hours=2)

        # tasks starting within the next 2 hours with no check-in
        checked_in_ids = db.query(models.Checkin.task_id).subquery()
        due_tasks = (
            db.query(models.Task)
            .filter(
                models.Task.is_active.is_(True),
                models.Task.planned_start >= now,
                models.Task.planned_start <= window_end,
                models.Task.id.not_in(checked_in_ids),
            )
            .all()
        )

        for task in due_tasks:
            user = db.query(models.User).filter(models.User.id == task.user_id).first()
            if user:
                notifications.send_reminder(user.email, task.name, task.planned_start)
                logger.info("Sent reminder to %s for task '%s'", user.email, task.name)
    except Exception:
        logger.exception("checkin_reminder_job failed")
    finally:
        db.close()


@scheduler.scheduled_job(CronTrigger(day_of_week="mon", hour=0, minute=0), id="device_reeval")
def device_reeval_job():
    from backend.ml.recommender import evaluate_device_level
    from backend.ml.bocd import get_or_create_detector

    db = SessionLocal()
    try:
        users = db.query(models.User).all()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff_14d = now - timedelta(days=14)

        for user in users:
            checkins_14d = (
                db.query(models.Checkin)
                .filter(
                    models.Checkin.user_id == user.id,
                    models.Checkin.checked_in_at >= cutoff_14d,
                )
                .all()
            )
            if not checkins_14d:
                continue

            completion_rate = sum(c.completed for c in checkins_14d) / len(checkins_14d)

            # weekly BOCD update
            detector = get_or_create_detector(str(user.id))
            cp_prob = detector.update(completion_rate)

            # count consecutive failures
            recent = (
                db.query(models.Checkin)
                .filter(models.Checkin.user_id == user.id)
                .order_by(models.Checkin.checked_in_at.desc())
                .limit(10)
                .all()
            )
            failure_streak = 0
            for c in recent:
                if c.completed == 0.0:
                    failure_streak += 1
                else:
                    break

            new_device = evaluate_device_level(
                beta_proxy=user.beta_proxy,
                proj_bias_score=user.proj_bias_score,
                drift_flag=cp_prob > 0.5,
                failure_streak=failure_streak,
            )

            # de-escalation check: 90%+ completion rate for 2 weeks → drop one level
            if completion_rate >= 0.90 and new_device > 0:
                new_device = max(0, user.current_device - 1)

            if new_device != user.current_device:
                assignment = models.DeviceAssignment(
                    user_id=user.id,
                    device_type=new_device,
                    beta_at_assignment=user.beta_proxy,
                    pre_completion_rate=completion_rate,
                )
                db.add(assignment)
                user.current_device = new_device
                logger.info(
                    "Device re-evaluated for %s: %d → %d (rate=%.2f)",
                    user.email, user.current_device, new_device, completion_rate,
                )

        db.commit()
    except Exception:
        logger.exception("device_reeval_job failed")
    finally:
        db.close()


def start_scheduler():
    scheduler.start()
    logger.info("Scheduler started")


def stop_scheduler():
    scheduler.shutdown()
