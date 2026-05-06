"""Translate backend domain types → ml.inference TaskFeatures.

The backend speaks human-readable strings + a 1–5 difficulty scale; the
ML inference layer speaks integer indices + difficulty ∈ [0, 1] +
log-normalized durations. This adapter is the only place the two
representations meet.

Note on schema version: the original placeholder spec said the task
feature vector was 13-dimensional. The actual trained model uses **10
dimensions** (1 difficulty + 4 category + 1 duration + 1 delay +
3 deadline-pressure). Callers should stop touching the raw vector and
go through ``encode_task`` below, which returns a ``TaskFeatures``
Pydantic model the inference API consumes directly.
"""
from __future__ import annotations

from typing import Literal

# Re-export the schema so backend code can type-annotate against it.
from ml.inference.inference_api import TaskFeatures

CATEGORIES: tuple[str, ...] = ("academic", "exercise", "work", "personal")
DEADLINE_OPTIONS: tuple[str, ...] = ("today", "this_week", "none")

# Map backend's "none" deadline → ML's "later" bucket (index 2).
_DEADLINE_TO_INDEX: dict[str, int] = {
    "today": 0,
    "this_week": 1,
    "none": 2,
}


def encode_task(
    *,
    category: Literal["academic", "exercise", "work", "personal"],
    deadline_pressure: Literal["today", "this_week", "none"],
    difficulty: int,
    planned_duration: int,
    days_until: int,
) -> TaskFeatures:
    """Build the inference-API task representation from backend domain types.

    Args:
        category: one of ``CATEGORIES``.
        deadline_pressure: one of ``DEADLINE_OPTIONS``.
        difficulty: integer 1–5; mapped to a float in [0, 1].
        planned_duration: minutes, must be positive.
        days_until: whole days from now until ``planned_start``, ≥ 0.

    Returns:
        A validated ``TaskFeatures`` Pydantic model.
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown category: {category!r}")
    if deadline_pressure not in DEADLINE_OPTIONS:
        raise ValueError(f"unknown deadline_pressure: {deadline_pressure!r}")
    if not 1 <= difficulty <= 5:
        raise ValueError(f"difficulty must be in [1, 5]: {difficulty}")

    return TaskFeatures(
        difficulty=(difficulty - 1) / 4.0,            # 1..5 → 0.0..1.0
        category_index=CATEGORIES.index(category),
        planned_duration_minutes=float(planned_duration),
        days_until_planned_start=int(days_until),
        deadline_pressure_index=_DEADLINE_TO_INDEX[deadline_pressure],
    )
