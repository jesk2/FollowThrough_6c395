"""API integration tests for the /tasks router."""
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid

from backend.main import app

client = TestClient(app)

MOCK_USER_ID = str(uuid.uuid4())
MOCK_TOKEN = "mock.jwt.token"


def mock_get_current_user():
    user = MagicMock()
    user.id = uuid.UUID(MOCK_USER_ID)
    user.email = "test@example.com"
    user.current_device = 0
    user.proj_bias_score = 0.0
    user.beta_proxy = 0.70
    user.streak = 0
    return user


def make_task_payload(overrides=None):
    payload = {
        "name": "Study for exam",
        "category": "academic",
        "difficulty": 3,
        "deadline_pressure": "today",
        "planned_start": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
        "planned_duration": 60,
    }
    if overrides:
        payload.update(overrides)
    return payload


@pytest.mark.integration
def test_create_task_requires_auth():
    resp = client.post("/tasks", json=make_task_payload())
    assert resp.status_code == 403


@pytest.mark.integration
def test_create_task_level1_requires_impl_fields():
    """At device level 1, impl_where and impl_what_first must be provided."""
    user = mock_get_current_user()
    user.current_device = 1

    from backend.dependencies import get_current_user

    # FastAPI captures Depends(...) targets at import time, so patching the
    # module attribute does nothing — must use dependency_overrides.
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        resp = client.post("/tasks", json=make_task_payload())
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    # should reject because impl_where and impl_what_first are missing
    assert resp.status_code == 422


@pytest.mark.integration
def test_projection_correction_applied_at_level2():
    """At device level 2 with proj_bias_score > 0.3, corrected_duration should differ from planned."""
    user = mock_get_current_user()
    user.current_device = 2
    user.proj_bias_score = 0.4  # 40% overrun history

    with patch("backend.routers.tasks.get_current_user", return_value=user):
        # corrected_duration = 60 * 1.4 = 84
        payload = make_task_payload()
        # We can't easily mock the DB here without a test DB — mark as skip for now
        pytest.skip("Requires test database setup")
