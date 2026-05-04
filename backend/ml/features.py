"""
Task feature engineering — IMPLEMENT THIS (Kaitlyn).

encode_task() must return a 13-dimensional numpy array.

Feature layout (13 dims):
  [0:4]  category one-hot    (academic, exercise, work, personal)
  [4:7]  deadline_pressure one-hot  (today, this_week, none)
  [7]    difficulty normalized to [0, 1]  → (difficulty - 1) / 4
  [8]    log(planned_duration + 1) normalized
  [9]    log(days_until + 1) normalized
  [10:13] reserved (zeros for now)
"""
from __future__ import annotations

import math
import numpy as np

CATEGORIES = ["academic", "exercise", "work", "personal"]
DEADLINE_OPTIONS = ["today", "this_week", "none"]


def encode_task(
    category: str,
    deadline_pressure: str,
    difficulty: int,
    planned_duration: int,
    days_until: int,
) -> np.ndarray:
    """
    Encode a task into a 13-dimensional feature vector.

    Args:
        category: one of "academic", "exercise", "work", "personal"
        deadline_pressure: one of "today", "this_week", "none"
        difficulty: integer 1–5
        planned_duration: minutes (> 0)
        days_until: days from now until planned_start (>= 0)

    Returns:
        np.ndarray of shape (13,), dtype float32
    """
    vec = np.zeros(13, dtype=np.float32)

    # category one-hot [0:4]
    if category in CATEGORIES:
        vec[CATEGORIES.index(category)] = 1.0

    # deadline one-hot [4:7]
    if deadline_pressure in DEADLINE_OPTIONS:
        vec[4 + DEADLINE_OPTIONS.index(deadline_pressure)] = 1.0

    # difficulty [7]
    vec[7] = (difficulty - 1) / 4.0

    # log-normalized planned duration [8]
    vec[8] = math.log1p(planned_duration) / math.log1p(480)  # normalize against 8h max

    # log-normalized days until [9]
    vec[9] = math.log1p(days_until) / math.log1p(30)  # normalize against 30-day horizon

    # dims [10:13] reserved — stay zero

    return vec
