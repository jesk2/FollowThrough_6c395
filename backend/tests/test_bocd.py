"""Unit tests for the BOCD anomaly detector. (Nicole to expand)"""
import pytest

from backend.ml.bocd import BOCDDetector, get_or_create_detector


def test_detector_creation():
    d = BOCDDetector(hazard_rate=1 / 30)
    assert d.hazard_rate == pytest.approx(1 / 30)


def test_get_or_create_returns_same_instance():
    d1 = get_or_create_detector("user-abc")
    d2 = get_or_create_detector("user-abc")
    assert d1 is d2


def test_get_or_create_returns_different_for_different_users():
    d1 = get_or_create_detector("user-x")
    d2 = get_or_create_detector("user-y")
    assert d1 is not d2


def test_detector_reset():
    d = BOCDDetector()
    d.reset()
    assert len(d._run_length_dist) == 1
    assert d._t == 0


# Placeholder — Nicole will add update() tests once implemented
@pytest.mark.skip(reason="BOCDDetector.update() not yet implemented")
def test_stable_sequence_low_cp_prob():
    d = BOCDDetector()
    for _ in range(20):
        cp = d.update(0.85)
    assert cp < 0.3


@pytest.mark.skip(reason="BOCDDetector.update() not yet implemented")
def test_sudden_drop_triggers_changepoint():
    d = BOCDDetector()
    for _ in range(20):
        d.update(0.90)
    for _ in range(5):
        cp = d.update(0.10)
    assert cp > 0.5
