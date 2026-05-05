"""Unit tests for the device recommender decision tree."""
import pytest

from backend.ml.recommender import (
    RecommendationResult,
    UserProfile,
    apply_projection_correction,
    evaluate,
)


# ----------------------------------------------------------------------
# projection correction (existing pinned behaviour)
# ----------------------------------------------------------------------


def test_projection_correction_no_bias():
    assert apply_projection_correction(60, 0.0) == 60


def test_projection_correction_30_percent():
    assert apply_projection_correction(60, 0.3) == 78  # 60 * 1.3


def test_projection_correction_rounds():
    # 45 * 1.3 = 58.5 → rounds to 58 or 59
    assert apply_projection_correction(45, 0.3) in (58, 59)


def test_projection_correction_only_corrects_upward():
    # negative bias must not shrink the planned duration
    assert apply_projection_correction(60, -0.2) == 60


# ----------------------------------------------------------------------
# recommender helpers
# ----------------------------------------------------------------------


def _profile(**overrides) -> UserProfile:
    base = dict(
        beta_proxy=0.70,
        proj_bias_score=0.0,
        drift_flag="stable",
        recent_failure_streak=0,
        current_device=0,
        weeks_at_current_level=0,
        recent_completion_rate=0.5,
        total_checkins=50,
    )
    base.update(overrides)
    return UserProfile(**base)


# ----------------------------------------------------------------------
# beta-baseline mapping (priority 6)
# ----------------------------------------------------------------------


def test_low_beta_returns_level_0():
    res = evaluate(_profile(beta_proxy=0.60))
    assert res.recommended_device == 0


def test_mid_low_beta_returns_level_1():
    res = evaluate(_profile(beta_proxy=0.70))
    assert res.recommended_device == 1


def test_mid_high_beta_returns_level_3():
    res = evaluate(_profile(beta_proxy=0.80))
    assert res.recommended_device == 3


def test_high_beta_returns_level_4():
    res = evaluate(_profile(beta_proxy=0.90))
    assert res.recommended_device == 4


# ----------------------------------------------------------------------
# overrides (priorities 1–5)
# ----------------------------------------------------------------------


def test_cold_start_returns_level_0_regardless_of_beta():
    res = evaluate(_profile(beta_proxy=0.95, total_checkins=5))
    assert res.recommended_device == 0
    assert res.reason == "insufficient data"


def test_high_proj_bias_overrides_beta():
    res = evaluate(_profile(beta_proxy=0.65, proj_bias_score=0.4))
    assert res.recommended_device == 2
    assert res.reason == "high projection bias"


def test_failure_streak_escalates_one_level():
    res = evaluate(_profile(beta_proxy=0.70, current_device=1, recent_failure_streak=3))
    assert res.recommended_device == 2
    assert res.reason == "repeated failures"


def test_failure_streak_caps_at_level_4():
    res = evaluate(_profile(beta_proxy=0.90, current_device=4, recent_failure_streak=5))
    assert res.recommended_device == 4


def test_sustained_success_de_escalates():
    res = evaluate(_profile(
        beta_proxy=0.80,
        current_device=3,
        recent_completion_rate=0.95,
        weeks_at_current_level=3,
    ))
    assert res.recommended_device == 2
    assert res.reason == "sustained high performance"


def test_de_escalation_does_not_fire_on_confirmed_decline():
    res = evaluate(_profile(
        beta_proxy=0.80,
        current_device=3,
        recent_completion_rate=0.95,
        weeks_at_current_level=3,
        drift_flag="confirmed_decline",
    ))
    assert res.recommended_device == 4  # decline path escalates instead


def test_drift_improvement_de_escalates():
    res = evaluate(_profile(
        beta_proxy=0.70,
        current_device=2,
        drift_flag="confirmed_improvement",
    ))
    assert res.recommended_device == 1
    assert res.reason == "confirmed improvement"


def test_drift_decline_escalates():
    res = evaluate(_profile(
        beta_proxy=0.70,
        current_device=1,
        drift_flag="confirmed_decline",
    ))
    assert res.recommended_device == 2
    assert res.reason == "confirmed decline"


# ----------------------------------------------------------------------
# RecommendationResult metadata
# ----------------------------------------------------------------------


def test_result_changed_flag_true_when_device_differs():
    res = evaluate(_profile(beta_proxy=0.90, current_device=0))
    assert res.changed is True
    assert res.previous_device == 0


def test_result_changed_flag_false_when_recommendation_matches_current():
    res = evaluate(_profile(beta_proxy=0.60, current_device=0))
    assert res.changed is False
    assert res.previous_device == 0


# ----------------------------------------------------------------------
# anti-thrash: drop guard on baseline path
# ----------------------------------------------------------------------


def test_baseline_drop_blocked_without_tenure():
    """β-baseline can't drop a user who just escalated. Prevents thrashing
    where a failure-streak escalates to L+1 and the very next check-in's
    baseline calc immediately undoes it.
    """
    res = evaluate(_profile(beta_proxy=0.60, current_device=2, weeks_at_current_level=0))
    # Baseline says L0 but user just got to L2 — guard blocks the drop
    assert res.recommended_device == 2
    assert res.changed is False
    assert "drop guarded" in res.reason


def test_baseline_drop_allowed_after_tenure():
    """Same situation, but user has been at the level for ≥1 week."""
    res = evaluate(_profile(beta_proxy=0.60, current_device=2, weeks_at_current_level=1))
    assert res.recommended_device == 0
    assert res.changed is True


def test_drop_guard_does_not_block_increases():
    """Increases (escalations) are always permitted, regardless of tenure."""
    res = evaluate(_profile(beta_proxy=0.90, current_device=0, weeks_at_current_level=0))
    assert res.recommended_device == 4
    assert res.changed is True


def test_drift_de_escalation_bypasses_drop_guard():
    """Confirmed-improvement drift is one-shot; it should still de-escalate
    even with no tenure (unlike baseline drops).
    """
    res = evaluate(_profile(
        beta_proxy=0.70, current_device=2,
        drift_flag="confirmed_improvement", weeks_at_current_level=0,
    ))
    assert res.recommended_device == 1
    assert res.changed is True
