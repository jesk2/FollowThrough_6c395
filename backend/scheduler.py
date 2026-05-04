"""
APScheduler job definitions.

Three jobs:
  1. checkin_reminder_job — every hour. Walks pending reminders for tasks
     starting in the next 2 hours with no check-in.
  2. bocd_daily_job — every day at 00:05. Ticks each user's BOCD detector
     with yesterday's completion rate; runs re-evaluation on confirmed drift.
  3. device_reeval_job — every Monday at 00:00. Runs re-evaluation for every
     user; logs device changes.
"""
import logging
from datetime import date, datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend import notifications
from backend.database import SessionLocal
from backend.ml import get_pending_reminders, on_device_change, run_reevaluation
from backend.ml.bocd import (
    load_detector,
    mark_ticked,
    save_detector,
    should_tick_today,
)
from backend.models import db as models
from backend.response_logic import _completion_rate_for_day  # internal helper reuse

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


@scheduler.scheduled_job(IntervalTrigger(hours=1), id="checkin_reminder")
def checkin_reminder_job():
    db = SessionLocal()
    try:
        for user, task, payload in get_pending_reminders(db):
            notifications.send_reminder(user.email, payload["task_name"], payload["planned_start"])
            logger.info("Sent reminder to %s for task '%s'", user.email, task.name)
    except Exception:
        logger.exception("checkin_reminder_job failed")
    finally:
        db.close()


@scheduler.scheduled_job(CronTrigger(hour=0, minute=5), id="bocd_daily")
def bocd_daily_job():
    """Tick each user's detector once a day with yesterday's completion rate.

    This catches users who haven't checked in today (so the on-checkin tick
    didn't fire). On confirmed drift, runs re-evaluation immediately.
    """
    db = SessionLocal()
    try:
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        users = db.query(models.User).all()

        for user in users:
            detector = load_detector(user)
            if not should_tick_today(detector, today):
                continue

            yest_rate = _completion_rate_for_day(user.id, yesterday, db)
            if yest_rate is None:
                continue

            prior_status = detector.drift_status
            detector.update(yest_rate)
            mark_ticked(detector, today)
            save_detector(user, detector)

            if (
                detector.drift_status != prior_status
                and detector.drift_status in ("confirmed_decline", "confirmed_improvement")
            ):
                direction = "decline" if detector.drift_status == "confirmed_decline" else "improvement"
                db.add(models.DriftEvent(
                    user_id=user.id,
                    drift_type="level_shift",
                    direction=direction,
                    beta_before=user.beta_proxy,
                ))
                old_device = user.current_device
                result = run_reevaluation(user, db)
                if result.changed:
                    note = on_device_change(user, old_device, result.recommended_device)
                    notifications.send_reminder(
                        user.email,
                        note.payload.get("message", "Your commitment level changed"),
                        datetime.utcnow(),
                    )

        db.commit()
    except Exception:
        logger.exception("bocd_daily_job failed")
    finally:
        db.close()


@scheduler.scheduled_job(CronTrigger(day_of_week="mon", hour=0, minute=0), id="device_reeval")
def device_reeval_job():
    db = SessionLocal()
    try:
        for user in db.query(models.User).all():
            old_device = user.current_device
            result = run_reevaluation(user, db)
            if result.changed:
                note = on_device_change(user, old_device, result.recommended_device)
                notifications.send_reminder(
                    user.email,
                    note.payload.get("message", "Your commitment level changed"),
                    datetime.utcnow(),
                )
                logger.info(
                    "Weekly re-eval for %s: %d → %d (%s)",
                    user.email, old_device, result.recommended_device, result.reason,
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
