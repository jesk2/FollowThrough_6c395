"""
Device-driven response logic.

The router and scheduler call into this module to translate user/task/check-in
state into *action descriptors* — pure data describing what should happen next.
The router applies side effects (DB writes, email sends) based on the
descriptors. Keeping side effects out of this layer makes the decisions easy to
unit-test and easy to dry-run from a notebook.

Public surface (re-exported by ``backend/ml/__init__.py``):

    process_task_creation(user, task_input)         -> TaskModification
    process_checkin(user, task, checkin, db)        -> CheckinActions
    process_reschedule_attempt(user, task, new_start) -> RescheduleResponse
    on_device_change(user, old_device, new_device)  -> NotificationDescriptor
    run_reevaluation(user, db)                      -> RecommendationResult
    get_pending_reminders(db)                       -> list[(User, Task)]
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from backend.models import db as models
from backend.ml.bocd import (
    BOCDDetector,
    load_detector,
    mark_ticked,
    save_detector,
    should_tick_today,
)
from backend.ml.recommender import (
    RecommendationResult,
    UserProfile,
    apply_projection_correction,
    evaluate,
)


logger = logging.getLogger(__name__)

REMINDER_WINDOW_HOURS = 2
RESCHEDULE_LOCK_WINDOW_HOURS = 24
PROJ_BIAS_EMA_ALPHA = 0.2
LEVEL_3_FAILURE_PENALTY = 5
STREAK_MILESTONES = (7, 14, 30)
DEVICE_LABELS = (
    "Salience Nudge",
    "Implementation Intention",
    "Planning Correction",
    "Virtual Stakes",
    "Precommitment Lock",
)


# ----------------------------------------------------------------------
# Action descriptors
# ----------------------------------------------------------------------


@dataclass
class NotificationDescriptor:
    kind: str  # reminder | streak_milestone | level_3_failure | device_change | drift_nudge
    payload: dict


@dataclass
class TaskModification:
    corrected_duration: Optional[int]
    locked: bool


@dataclass
class CheckinActions:
    streak_delta: int
    streak_set_to: Optional[int]  # None = additive; int = override (e.g. reset to 0)
    proj_bias_update: Optional[float]
    notifications: list[NotificationDescriptor] = field(default_factory=list)
    trigger_reevaluation: bool = False
    daily_rate: Optional[float] = None
    bocd_ticked: bool = False
    drift_status: str = "stable"


@dataclass
class RescheduleResponse:
    allowed: bool
    requires_confirmation: bool
    confirmation_token: Optional[str] = None
    reason: Optional[str] = None


# In-memory store of pending reschedule confirmation tokens.
# Keyed by token -> (user_id, task_id, requested_new_start, expires_at).
# Lost on server restart; acceptable for short-lived UI confirmations.
_pending_reschedule_tokens: dict[str, dict[str, Any]] = {}
_TOKEN_TTL = timedelta(minutes=10)


# ----------------------------------------------------------------------
# Task creation
# ----------------------------------------------------------------------


def process_task_creation(user: models.User, task_input) -> TaskModification:
    """Compute device-driven modifications for a new task.

    ``task_input`` is duck-typed: needs ``planned_duration``, ``difficulty``,
    ``deadline_pressure`` attributes (works for the Pydantic ``TaskCreate`` body
    or an ORM ``Task``).
    """
    corrected: Optional[int] = None
    if user.current_device >= 2 and user.proj_bias_score > 0.0:
        candidate = apply_projection_correction(task_input.planned_duration, user.proj_bias_score)
        if candidate > task_input.planned_duration:
            corrected = candidate

    locked = False
    if user.current_device >= 4:
        if task_input.difficulty >= 4 or task_input.deadline_pressure == "today":
            locked = True

    return TaskModification(corrected_duration=corrected, locked=locked)


# ----------------------------------------------------------------------
# Check-in
# ----------------------------------------------------------------------


def process_checkin(
    user: models.User,
    task: models.Task,
    checkin_input,
    db: Session,
) -> CheckinActions:
    """Compute side-effect descriptors for a fresh check-in.

    Persists detector state (and DriftEvent rows when drift is confirmed) but does
    not commit — the caller owns the session.
    """
    notifications: list[NotificationDescriptor] = []

    # --- projection bias: rolling EMA on overrun ---
    proj_bias_update: Optional[float] = None
    if (
        checkin_input.completed > 0
        and checkin_input.actual_duration is not None
        and task.planned_duration
        and task.planned_duration > 0
    ):
        overrun = (checkin_input.actual_duration - task.planned_duration) / task.planned_duration
        proj_bias_update = (
            (1 - PROJ_BIAS_EMA_ALPHA) * user.proj_bias_score + PROJ_BIAS_EMA_ALPHA * overrun
        )

    # --- streak: depends on level + outcome ---
    streak_delta = 0
    streak_set_to: Optional[int] = None
    new_streak = user.streak

    if user.current_device == 3:
        if checkin_input.completed == 1.0:
            streak_delta = 1
            new_streak = user.streak + 1
        elif checkin_input.completed == 0.0:
            new_streak = max(0, user.streak - LEVEL_3_FAILURE_PENALTY)
            streak_delta = new_streak - user.streak
            streak_set_to = new_streak
            notifications.append(NotificationDescriptor(
                kind="level_3_failure",
                payload={
                    "user_id": str(user.id),
                    "task_name": task.name,
                    "streak_lost": LEVEL_3_FAILURE_PENALTY,
                    "new_streak": new_streak,
                },
            ))
    else:
        if checkin_input.completed == 1.0:
            streak_delta = 1
            new_streak = user.streak + 1
        elif checkin_input.completed == 0.0:
            streak_set_to = 0
            streak_delta = -user.streak
            new_streak = 0

    if user.current_device == 3 and new_streak in STREAK_MILESTONES and streak_delta > 0:
        notifications.append(NotificationDescriptor(
            kind="streak_milestone",
            payload={
                "user_id": str(user.id),
                "milestone": new_streak,
            },
        ))

    # --- BOCD daily tick ---
    today = datetime.now(timezone.utc).date()
    daily_rate, bocd_ticked, drift_status, drift_changed = _maybe_tick_bocd(user, db, today)
    trigger_reevaluation = drift_changed and drift_status in (
        "confirmed_decline",
        "confirmed_improvement",
    )

    return CheckinActions(
        streak_delta=streak_delta,
        streak_set_to=streak_set_to,
        proj_bias_update=proj_bias_update,
        notifications=notifications,
        trigger_reevaluation=trigger_reevaluation,
        daily_rate=daily_rate,
        bocd_ticked=bocd_ticked,
        drift_status=drift_status,
    )


def _maybe_tick_bocd(
    user: models.User, db: Session, today: date
) -> tuple[Optional[float], bool, str, bool]:
    """Advance the user's BOCD detector at most once per calendar day.

    Uses *yesterday's* full-day completion rate; today is still in progress.
    Returns (yesterday_rate, ticked, drift_status, drift_changed).
    """
    detector = load_detector(user)
    prior_status = detector.drift_status

    if not should_tick_today(detector, today):
        return None, False, detector.drift_status, False

    yesterday = today - timedelta(days=1)
    yesterday_rate = _completion_rate_for_day(user.id, yesterday, db)
    if yesterday_rate is None:
        # no tasks yesterday — skip the tick (per spec)
        return None, False, detector.drift_status, False

    detector.update(yesterday_rate)
    mark_ticked(detector, today)
    save_detector(user, detector)

    drift_changed = detector.drift_status != prior_status
    if drift_changed and detector.drift_status in ("confirmed_decline", "confirmed_improvement"):
        direction = "decline" if detector.drift_status == "confirmed_decline" else "improvement"
        db.add(models.DriftEvent(
            user_id=user.id,
            drift_type="level_shift",
            direction=direction,
            beta_before=user.beta_proxy,
        ))
    elif drift_changed and prior_status == "potential" and detector.drift_status == "stable":
        # transient classification — log so the dashboard can show it
        db.add(models.DriftEvent(
            user_id=user.id,
            drift_type="transient",
            direction="decline",  # transients are typically dips; placeholder
            beta_before=user.beta_proxy,
        ))

    return yesterday_rate, True, detector.drift_status, drift_changed


def _completion_rate_for_day(user_id: UUID, day: date, db: Session) -> Optional[float]:
    """Mean completion across all check-ins for ``user_id`` on ``day`` (UTC)."""
    start = datetime.combine(day, datetime.min.time())
    end = start + timedelta(days=1)
    checkins = (
        db.query(models.Checkin)
        .filter(
            models.Checkin.user_id == user_id,
            models.Checkin.checked_in_at >= start,
            models.Checkin.checked_in_at < end,
        )
        .all()
    )
    if not checkins:
        return None
    return sum(c.completed for c in checkins) / len(checkins)


# ----------------------------------------------------------------------
# Reschedule
# ----------------------------------------------------------------------


def process_reschedule_attempt(
    user: models.User,
    task: models.Task,
    new_start: datetime,
    db: Session,
    confirmation_token: Optional[str] = None,
) -> RescheduleResponse:
    """Decide whether to allow a reschedule on a (possibly locked) task.

    If the task is locked AND its planned_start is within the next 24h, the
    first call returns ``requires_confirmation=True`` with a short-lived token.
    A subsequent call with a valid token logs a ``RescheduleEvent`` and allows
    the edit.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    within_lock_window = (task.planned_start - now) < timedelta(hours=RESCHEDULE_LOCK_WINDOW_HOURS)

    if not task.locked or not within_lock_window:
        return RescheduleResponse(allowed=True, requires_confirmation=False)

    if confirmation_token is None:
        token = _issue_reschedule_token(user.id, task.id, new_start)
        return RescheduleResponse(
            allowed=False,
            requires_confirmation=True,
            confirmation_token=token,
            reason="locked task within 24h — please confirm",
        )

    if not _consume_reschedule_token(confirmation_token, user.id, task.id, new_start):
        return RescheduleResponse(
            allowed=False,
            requires_confirmation=True,
            reason="invalid or expired confirmation token",
        )

    db.add(models.RescheduleEvent(
        task_id=task.id,
        user_id=user.id,
        original_start=task.planned_start,
        rescheduled_to=new_start,
    ))
    return RescheduleResponse(allowed=True, requires_confirmation=False)


def _issue_reschedule_token(user_id: UUID, task_id: UUID, new_start: datetime) -> str:
    _purge_expired_tokens()
    token = secrets.token_urlsafe(24)
    _pending_reschedule_tokens[token] = {
        "user_id": str(user_id),
        "task_id": str(task_id),
        "new_start": new_start,
        "expires_at": datetime.now(timezone.utc).replace(tzinfo=None) + _TOKEN_TTL,
    }
    return token


def _consume_reschedule_token(
    token: str, user_id: UUID, task_id: UUID, new_start: datetime
) -> bool:
    record = _pending_reschedule_tokens.pop(token, None)
    if not record:
        return False
    if record["expires_at"] < datetime.now(timezone.utc).replace(tzinfo=None):
        return False
    if record["user_id"] != str(user_id) or record["task_id"] != str(task_id):
        return False
    if record["new_start"] != new_start:
        return False
    return True


def _purge_expired_tokens() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    expired = [t for t, r in _pending_reschedule_tokens.items() if r["expires_at"] < now]
    for t in expired:
        _pending_reschedule_tokens.pop(t, None)


# ----------------------------------------------------------------------
# Device-change notification
# ----------------------------------------------------------------------


def on_device_change(user: models.User, old_device: int, new_device: int) -> NotificationDescriptor:
    """Build the right-toned device-change notification descriptor."""
    if new_device == old_device:
        return NotificationDescriptor(
            kind="drift_nudge",
            payload={
                "user_id": str(user.id),
                "tone": "supportive",
                "message": (
                    "We noticed a brief dip but you're back on track — keep going."
                ),
            },
        )
    if new_device > old_device:
        return NotificationDescriptor(
            kind="device_change",
            payload={
                "user_id": str(user.id),
                "tone": "neutral",
                "direction": "escalation",
                "from_level": old_device,
                "to_level": new_device,
                "from_label": DEVICE_LABELS[old_device],
                "to_label": DEVICE_LABELS[new_device],
                "message": (
                    f"Your commitment level moved from {DEVICE_LABELS[old_device]} to "
                    f"{DEVICE_LABELS[new_device]}. Here's what changes for you."
                ),
            },
        )
    return NotificationDescriptor(
        kind="device_change",
        payload={
            "user_id": str(user.id),
            "tone": "positive",
            "direction": "de_escalation",
            "from_level": old_device,
            "to_level": new_device,
            "from_label": DEVICE_LABELS[old_device],
            "to_label": DEVICE_LABELS[new_device],
            "message": (
                f"Sustained progress earned you a step down to "
                f"{DEVICE_LABELS[new_device]}. Nice work."
            ),
        },
    )


# ----------------------------------------------------------------------
# Re-evaluation (called by scheduler and on confirmed drift)
# ----------------------------------------------------------------------


def run_reevaluation(user: models.User, db: Session) -> RecommendationResult:
    """Build the user's profile from the DB, evaluate, and persist any change.

    If the profile carries a ``confirmed_*`` drift flag, the detector is reset
    to ``stable`` afterwards: the drift signal is one-shot (consumed by the
    recommender on this call) so subsequent reevaluations don't repeatedly
    escalate/de-escalate the device on the same drift event.
    """
    profile = _build_user_profile(user, db)
    result = evaluate(profile)

    if result.changed:
        rate_14d = _completion_rate_window_days(user.id, 14, db)
        db.add(models.DeviceAssignment(
            user_id=user.id,
            device_type=result.recommended_device,
            beta_at_assignment=user.beta_proxy,
            pre_completion_rate=rate_14d,
        ))
        user.current_device = result.recommended_device

    if profile.drift_flag in ("confirmed_decline", "confirmed_improvement"):
        detector = load_detector(user)
        detector.drift_status = "stable"
        save_detector(user, detector)

    return result


def _build_user_profile(user: models.User, db: Session) -> UserProfile:
    detector = load_detector(user)
    drift = detector.get_drift_status()

    rate_14d = _completion_rate_window_days(user.id, 14, db) or 0.0
    failure_streak = _consecutive_failure_streak(user.id, db)
    total_checkins = (
        db.query(models.Checkin).filter(models.Checkin.user_id == user.id).count()
    )
    weeks = _weeks_at_current_level(user, db)

    return UserProfile(
        beta_proxy=user.beta_proxy,
        proj_bias_score=user.proj_bias_score,
        drift_flag=drift,
        recent_failure_streak=failure_streak,
        current_device=user.current_device,
        weeks_at_current_level=weeks,
        recent_completion_rate=rate_14d,
        total_checkins=total_checkins,
    )


def _completion_rate_window_days(user_id: UUID, days: int, db: Session) -> Optional[float]:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    checkins = (
        db.query(models.Checkin)
        .filter(models.Checkin.user_id == user_id, models.Checkin.checked_in_at >= cutoff)
        .all()
    )
    if not checkins:
        return None
    return sum(c.completed for c in checkins) / len(checkins)


def _consecutive_failure_streak(user_id: UUID, db: Session, lookback: int = 10) -> int:
    recent = (
        db.query(models.Checkin)
        .filter(models.Checkin.user_id == user_id)
        .order_by(models.Checkin.checked_in_at.desc())
        .limit(lookback)
        .all()
    )
    streak = 0
    for c in recent:
        if c.completed == 0.0:
            streak += 1
        else:
            break
    return streak


def _weeks_at_current_level(user: models.User, db: Session) -> int:
    last = (
        db.query(models.DeviceAssignment)
        .filter(
            models.DeviceAssignment.user_id == user.id,
            models.DeviceAssignment.device_type == user.current_device,
        )
        .order_by(models.DeviceAssignment.assigned_at.desc())
        .first()
    )
    anchor = last.assigned_at if last else user.created_at
    delta = datetime.now(timezone.utc).replace(tzinfo=None) - anchor
    return max(0, delta.days // 7)


# ----------------------------------------------------------------------
# Pending reminders (used by the hourly scheduler)
# ----------------------------------------------------------------------


def get_pending_reminders(db: Session) -> list[tuple[models.User, models.Task, dict]]:
    """Tasks starting within the next 2 hours with no check-in yet.

    Returns triples of (user, task, payload). For Level-1 users, the payload
    echoes the implementation-intention fields so the notification can include
    them.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_end = now + timedelta(hours=REMINDER_WINDOW_HOURS)

    checked_in_ids = db.query(models.Checkin.task_id).subquery()
    due = (
        db.query(models.Task)
        .filter(
            models.Task.is_active.is_(True),
            models.Task.planned_start >= now,
            models.Task.planned_start <= window_end,
            models.Task.id.not_in(checked_in_ids),
        )
        .all()
    )

    out: list[tuple[models.User, models.Task, dict]] = []
    for task in due:
        user = db.query(models.User).filter(models.User.id == task.user_id).first()
        if not user:
            continue
        payload = {
            "task_name": task.name,
            "planned_start": task.planned_start,
        }
        if user.current_device >= 1:
            payload["impl_where"] = task.impl_where
            payload["impl_what_first"] = task.impl_what_first
        out.append((user, task, payload))
    return out
