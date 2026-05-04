"""Public surface for the device/anomaly pipeline.

The router and scheduler import these names; the implementations live in
``backend.response_logic``. Re-exports are resolved lazily so that the pure
ML modules (``backend.ml.bocd``, ``backend.ml.recommender``) remain importable
without the full backend stack (sqlalchemy, etc.).
"""
from typing import TYPE_CHECKING

__all__ = [
    "CheckinActions",
    "NotificationDescriptor",
    "RescheduleResponse",
    "TaskModification",
    "get_pending_reminders",
    "on_device_change",
    "process_checkin",
    "process_reschedule_attempt",
    "process_task_creation",
    "run_reevaluation",
]

if TYPE_CHECKING:  # pragma: no cover
    from backend.response_logic import (  # noqa: F401
        CheckinActions,
        NotificationDescriptor,
        RescheduleResponse,
        TaskModification,
        get_pending_reminders,
        on_device_change,
        process_checkin,
        process_reschedule_attempt,
        process_task_creation,
        run_reevaluation,
    )


def __getattr__(name: str):
    if name in __all__:
        from backend import response_logic  # local import — defers sqlalchemy load
        return getattr(response_logic, name)
    raise AttributeError(f"module 'backend.ml' has no attribute {name!r}")
