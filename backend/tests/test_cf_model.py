"""Unit tests for the CF adapter and the backend feature encoder.

These tests cover only the paths that don't require trained ML artifacts
(``cf_model.pt`` / ``ridge_probe.joblib`` / ``user_state.pt``), which are
gitignored and absent from CI.
"""
import pytest

from backend.ml.cf_model import K, get_user_embedding
from backend.ml.features import CATEGORIES, DEADLINE_OPTIONS, encode_task
from ml.inference.inference_api import TaskFeatures


def test_user_embedding_dim_is_eight():
    assert K == 8


def test_encode_task_returns_taskfeatures():
    feat = encode_task(
        category="academic",
        deadline_pressure="today",
        difficulty=3,
        planned_duration=60,
        days_until=0,
    )
    assert isinstance(feat, TaskFeatures)


def test_encode_task_difficulty_normalized():
    for diff in range(1, 6):
        feat = encode_task(
            category="work",
            deadline_pressure="this_week",
            difficulty=diff,
            planned_duration=45,
            days_until=2,
        )
        assert 0.0 <= feat.difficulty <= 1.0
    # endpoints
    lo = encode_task(category="work", deadline_pressure="none",
                     difficulty=1, planned_duration=30, days_until=0)
    hi = encode_task(category="work", deadline_pressure="none",
                     difficulty=5, planned_duration=30, days_until=0)
    assert lo.difficulty == pytest.approx(0.0)
    assert hi.difficulty == pytest.approx(1.0)


def test_encode_task_category_index_matches_order():
    for i, cat in enumerate(CATEGORIES):
        feat = encode_task(
            category=cat,
            deadline_pressure="none",
            difficulty=2,
            planned_duration=30,
            days_until=1,
        )
        assert feat.category_index == i


def test_encode_task_deadline_pressure_mapping():
    expected = {"today": 0, "this_week": 1, "none": 2}
    for label, idx in expected.items():
        feat = encode_task(
            category="personal",
            deadline_pressure=label,
            difficulty=1,
            planned_duration=15,
            days_until=0,
        )
        assert feat.deadline_pressure_index == idx
    assert set(DEADLINE_OPTIONS) == set(expected.keys())


def test_encode_task_rejects_bad_inputs():
    with pytest.raises(ValueError):
        encode_task(category="not_a_cat", deadline_pressure="today",  # type: ignore[arg-type]
                    difficulty=3, planned_duration=30, days_until=0)
    with pytest.raises(ValueError):
        encode_task(category="academic", deadline_pressure="someday",  # type: ignore[arg-type]
                    difficulty=3, planned_duration=30, days_until=0)
    with pytest.raises(ValueError):
        encode_task(category="academic", deadline_pressure="today",
                    difficulty=0, planned_duration=30, days_until=0)
    with pytest.raises(ValueError):
        encode_task(category="academic", deadline_pressure="today",
                    difficulty=6, planned_duration=30, days_until=0)


def test_get_user_embedding_none_id_returns_none():
    assert get_user_embedding(None) is None
