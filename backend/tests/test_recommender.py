"""Unit tests for the device recommender decision tree. (Nicole to expand)"""
import pytest

from backend.ml.recommender import apply_projection_correction


def test_projection_correction_no_bias():
    corrected = apply_projection_correction(60, 0.0)
    assert corrected == 60


def test_projection_correction_30_percent():
    corrected = apply_projection_correction(60, 0.3)
    assert corrected == 78  # 60 * 1.3


def test_projection_correction_rounds():
    # 45 * 1.3 = 58.5 → rounds to 58 or 59
    corrected = apply_projection_correction(45, 0.3)
    assert corrected in (58, 59)


# Placeholders — Nicole implements evaluate_device_level()
@pytest.mark.skip(reason="evaluate_device_level() not yet implemented")
def test_low_beta_low_bias_returns_level_0():
    from backend.ml.recommender import evaluate_device_level
    level = evaluate_device_level(beta_proxy=0.60, proj_bias_score=0.1, drift_flag=False, failure_streak=0)
    assert level == 0


@pytest.mark.skip(reason="evaluate_device_level() not yet implemented")
def test_high_proj_bias_returns_level_2():
    from backend.ml.recommender import evaluate_device_level
    level = evaluate_device_level(beta_proxy=0.65, proj_bias_score=0.4, drift_flag=False, failure_streak=0)
    assert level == 2


@pytest.mark.skip(reason="evaluate_device_level() not yet implemented")
def test_three_failures_escalates():
    from backend.ml.recommender import evaluate_device_level
    level = evaluate_device_level(beta_proxy=0.65, proj_bias_score=0.1, drift_flag=False, failure_streak=3)
    # should escalate beyond level 1
    assert level >= 2
