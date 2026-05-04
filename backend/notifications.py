"""Email notification sending. Falls back to console logging when SMTP is not configured."""
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText

from backend.config import settings

logger = logging.getLogger(__name__)


def send_reminder(email: str, task_name: str, planned_start: datetime) -> None:
    subject = f"FollowThrough: time to start '{task_name}'"
    body = (
        f"Hi,\n\n"
        f"This is a reminder that you planned to start '{task_name}' "
        f"at {planned_start.strftime('%H:%M')} today.\n\n"
        f"Open FollowThrough to check in when you're done.\n"
    )

    if not settings.smtp_user:
        logger.info("[REMINDER] To %s — %s", email, subject)
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
        logger.exception("Failed to send reminder email to %s", email)
