"""Unit tests for the BOCD anomaly detector."""
import numpy as np
import pytest

from backend.ml.bocd import BOCDDetector, load_detector, save_detector


# ----------------------------------------------------------------------
# construction & basic state
# ----------------------------------------------------------------------


def test_detector_creation_default_hazard():
    d = BOCDDetector()
    assert d.hazard_rate == pytest.approx(1 / 30)
    assert d.drift_status == "stable"
    assert len(d.run_length_probs) == 1
    assert d.run_length_probs[0] == pytest.approx(1.0)


def test_detector_creation_custom_hazard():
    d = BOCDDetector(hazard_rate=0.05)
    assert d.hazard_rate == pytest.approx(0.05)


def test_detector_reset():
    d = BOCDDetector()
    for _ in range(5):
        d.update(0.7)
    d.reset()
    assert len(d.run_length_probs) == 1
    assert d.drift_status == "stable"
    assert d.pending_flag_days == 0


# ----------------------------------------------------------------------
# update behaviour
# ----------------------------------------------------------------------


def test_update_returns_float_in_unit_interval():
    d = BOCDDetector()
    cp = d.update(0.7)
    assert isinstance(cp, float)
    assert 0.0 <= cp <= 1.0


def test_update_clips_extreme_values():
    """y=0.0 or 1.0 must not crash the Beta PDF eval."""
    d = BOCDDetector()
    cp_zero = d.update(0.0)
    cp_one = d.update(1.0)
    assert np.isfinite(cp_zero)
    assert np.isfinite(cp_one)


def test_run_length_distribution_grows_over_time():
    d = BOCDDetector()
    for _ in range(5):
        d.update(0.7)
    # length grows by 1 each update (initial 1 + 5 updates)
    assert len(d.run_length_probs) == 6
    assert d.alpha.shape == d.run_length_probs.shape
    assert d.b.shape == d.run_length_probs.shape


def test_run_length_distribution_sums_to_one():
    d = BOCDDetector()
    for y in [0.6, 0.7, 0.8, 0.5, 0.9]:
        d.update(y)
        assert d.run_length_probs.sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# spec scenarios
# ----------------------------------------------------------------------


def test_stable_series_does_not_flag():
    """30 days of uniform 0.85 should keep cp_prob below threshold."""
    d = BOCDDetector()
    cps = [d.update(0.85) for _ in range(30)]
    assert max(cps) < 0.5
    assert d.drift_status == "stable"


def test_step_change_triggers_changepoint_within_5_days():
    """0.7 for 20 days then 0.3 for 5 days should flag inside the post-drop window."""
    d = BOCDDetector()
    for _ in range(20):
        d.update(0.7)
    cp_after_drop = [d.update(0.3) for _ in range(5)]
    assert max(cp_after_drop) > 0.5


def test_transient_dip_classified_as_transient():
    """Three bad days flanked by stable days should not become a confirmed decline."""
    d = BOCDDetector()
    for _ in range(15):
        d.update(0.85)
    for _ in range(3):
        d.update(0.20)
    for _ in range(15):
        d.update(0.85)
    assert d.drift_status not in ("confirmed_decline", "confirmed_improvement")


def test_sustained_decline_classified_as_confirmed_decline():
    """An extended drop past the classification window should escalate."""
    d = BOCDDetector()
    for _ in range(20):
        d.update(0.85)
    for _ in range(10):
        d.update(0.20)
    assert d.drift_status == "confirmed_decline"


# ----------------------------------------------------------------------
# accessors
# ----------------------------------------------------------------------


def test_last_result_keys():
    d = BOCDDetector()
    d.update(0.7)
    res = d.last_result()
    assert set(res.keys()) >= {
        "changepoint_prob",
        "expected_run_length",
        "regime_completion_rate",
        "flagged",
        "drift_status",
    }


def test_expected_run_length_grows_under_stable_input():
    d = BOCDDetector()
    rls = []
    for _ in range(15):
        d.update(0.8)
        rls.append(d.expected_run_length())
    assert rls[-1] > rls[2]  # later observations, longer expected run


# ----------------------------------------------------------------------
# serialization
# ----------------------------------------------------------------------


def test_to_dict_from_dict_round_trip_state():
    d = BOCDDetector()
    for y in [0.7, 0.6, 0.8, 0.5, 0.9]:
        d.update(y)
    state = d.to_dict()
    restored = BOCDDetector.from_dict(state)

    assert np.allclose(restored.run_length_probs, d.run_length_probs)
    assert np.allclose(restored.alpha, d.alpha)
    assert np.allclose(restored.b, d.b)
    assert restored.drift_status == d.drift_status
    assert restored.pending_flag_days == d.pending_flag_days


def test_round_trip_produces_identical_next_update():
    """from_dict must yield a detector whose next .update() matches the original."""
    d = BOCDDetector()
    for y in [0.7, 0.6, 0.8]:
        d.update(y)
    restored = BOCDDetector.from_dict(d.to_dict())
    cp_orig = d.update(0.4)
    cp_restored = restored.update(0.4)
    assert cp_orig == pytest.approx(cp_restored)


# ----------------------------------------------------------------------
# persistence helpers
# ----------------------------------------------------------------------


class _FakeUser:
    """Stand-in for the ORM User row — only needs ``detector_state`` attribute."""
    def __init__(self):
        self.detector_state = None


def test_load_detector_for_new_user_returns_fresh():
    user = _FakeUser()
    d = load_detector(user)
    assert d.drift_status == "stable"
    assert len(d.run_length_probs) == 1


def test_save_then_load_detector_round_trip():
    user = _FakeUser()
    d = BOCDDetector()
    for y in [0.7, 0.5, 0.9]:
        d.update(y)
    save_detector(user, d)
    assert user.detector_state is not None

    restored = load_detector(user)
    assert np.allclose(restored.run_length_probs, d.run_length_probs)
    assert restored.drift_status == d.drift_status
