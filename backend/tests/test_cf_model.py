"""Unit tests for the CF model forward pass and embedding management. (Kaitlyn to expand)"""
import numpy as np
import pytest

from backend.ml.cf_model import K, get_user_embedding, set_user_embedding
from backend.ml.features import encode_task


def test_encode_task_shape():
    vec = encode_task("academic", "today", 3, 60, 0)
    assert vec.shape == (13,)
    assert vec.dtype == np.float32


def test_encode_task_category_onehot():
    vec = encode_task("exercise", "none", 1, 30, 5)
    # exercise is index 1 in CATEGORIES
    assert vec[1] == 1.0
    assert vec[0] == 0.0
    assert vec[2] == 0.0
    assert vec[3] == 0.0


def test_encode_task_difficulty_range():
    for diff in range(1, 6):
        vec = encode_task("work", "this_week", diff, 45, 2)
        assert 0.0 <= vec[7] <= 1.0


def test_user_embedding_roundtrip():
    emb = np.random.randn(K).astype(np.float32)
    set_user_embedding("test-user-123", emb)
    retrieved = get_user_embedding("test-user-123")
    assert retrieved is not None
    np.testing.assert_array_equal(retrieved, emb)


def test_unknown_user_embedding_returns_none():
    result = get_user_embedding("user-that-does-not-exist")
    assert result is None
