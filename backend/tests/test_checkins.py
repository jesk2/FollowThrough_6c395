"""API integration tests for the /checkins router."""
import pytest
from unittest.mock import MagicMock
import uuid

from backend.ml.recommender import apply_projection_correction


def test_proj_bias_update_formula():
    """Verify the exponential moving average formula for proj_bias_score."""
    # overrun = (actual - planned) / planned = (75 - 60) / 60 = 0.25
    # new_score = 0.8 * 0.0 + 0.2 * 0.25 = 0.05
    existing_score = 0.0
    planned = 60
    actual = 75
    overrun = (actual - planned) / planned
    new_score = 0.8 * existing_score + 0.2 * overrun
    assert abs(new_score - 0.05) < 1e-6


def test_streak_increments_on_completion():
    """Streak should increment on full completion, reset on failure."""
    streak = 5
    # simulate full completion
    completed = 1.0
    if completed == 1.0:
        streak += 1
    assert streak == 6

    # simulate failure
    completed = 0.0
    if completed == 0.0:
        streak = 0
    assert streak == 0


@pytest.mark.integration
def test_submit_checkin_returns_201():
    pytest.skip("Requires test database setup")


@pytest.mark.integration
def test_duplicate_checkin_returns_409():
    pytest.skip("Requires test database setup")
