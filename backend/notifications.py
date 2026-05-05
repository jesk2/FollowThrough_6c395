"""Email notification sending. Falls back to console logging when SMTP is not configured."""
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

from backend.config import settings

if TYPE_CHECKING:
    from backend.response_logic import NotificationDescriptor

logger = logging.getLogger(__name__)


def _send_email(email: str, subject: str, body: str) -> None:
    if not settings.smtp_user:
        logger.info("[NOTIFICATION] To %s — %s", email, subject)
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = settings.smtp_user
    msg["To"] = email

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)
    except Exception:
        logger.exception("Failed to send email to %s", email)


def send_notification(email: str, descriptor: "NotificationDescriptor") -> None:
    """Dispatch an email for any NotificationDescriptor kind."""
    kind = descriptor.kind
    p = descriptor.payload

    if kind == "reminder":
        task_name = p.get("task_name", "your task")
        planned_start = p.get("planned_start", datetime.utcnow())
        subject = f"FollowThrough: time to start '{task_name}'"
        body = (
            f"Hi,\n\n"
            f"This is a reminder that you planned to start '{task_name}' "
            f"at {planned_start.strftime('%H:%M')} today.\n"
        )
        if p.get("impl_where") and p.get("impl_what_first"):
            body += (
                f"\nYou said you'd do this {p['impl_where']}, "
                f"starting with {p['impl_what_first']}.\n"
            )
        body += "\nOpen FollowThrough to check in when you're done.\n"

    elif kind == "streak_milestone":
        milestone = p.get("milestone", "")
        subject = f"FollowThrough: {milestone}-day streak!"
        body = (
            f"Hi,\n\n"
            f"You hit a {milestone}-day streak — amazing consistency!\n\n"
            f"Keep it up.\n"
        )

    elif kind == "level_3_failure":
        task_name = p.get("task_name", "your task")
        streak_lost = p.get("streak_lost", 0)
        new_streak = p.get("new_streak", 0)
        subject = f"FollowThrough: missed '{task_name}'"
        body = (
            f"Hi,\n\n"
            f"You missed '{task_name}'. You lost {streak_lost} streak "
            f"point{'s' if streak_lost != 1 else ''}; you're now at {new_streak}.\n\n"
            f"One miss doesn't define you — get back on track today.\n"
        )

    elif kind == "device_change":
        from_label = p.get("from_label", "")
        to_label = p.get("to_label", "")
        tone = p.get("tone", "neutral")
        message = p.get("message", "Your commitment level changed.")
        if tone == "positive":
            subject = f"FollowThrough: commitment level updated to {to_label}"
            body = (
                f"Hi,\n\n"
                f"Great news — your consistency has earned you a lighter commitment level.\n\n"
                f"{message}\n\n"
                f"You moved from {from_label} to {to_label}.\n"
            )
        else:
            subject = f"FollowThrough: commitment level updated to {to_label}"
            body = (
                f"Hi,\n\n"
                f"{message}\n\n"
                f"You moved from {from_label} to {to_label}.\n"
            )

    elif kind == "drift_nudge":
        message = p.get("message", "We noticed a brief dip but you're back on track — keep going.")
        subject = "FollowThrough: you're back on track"
        body = f"Hi,\n\n{message}\n"

    else:
        logger.warning("Unknown notification kind %r — skipping", kind)
        return

    _send_email(email, subject, body)


def send_reminder(email: str, task_name: str, planned_start: datetime) -> None:
    """Legacy helper; prefer send_notification for new call sites."""
    from backend.response_logic import NotificationDescriptor
    send_notification(
        email,
        NotificationDescriptor(
            kind="reminder",
            payload={"task_name": task_name, "planned_start": planned_start},
        ),
    )
