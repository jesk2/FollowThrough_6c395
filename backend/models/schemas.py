"""
Pydantic request/response schemas.
These are the source of truth for the frontend API contract — share with Audrey.
"""
from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class TaskCreate(BaseModel):
    name: str
    category: Literal["academic", "exercise", "work", "personal"]
    difficulty: int = Field(..., ge=1, le=5)
    deadline_pressure: Literal["today", "this_week", "none"]
    planned_start: datetime
    planned_duration: int = Field(..., gt=0, description="minutes")
    # Level 1 — implementation intention (required when user.current_device >= 1)
    impl_where: Optional[str] = None
    impl_what_first: Optional[str] = None


class TaskResponse(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    category: str
    difficulty: int
    deadline_pressure: str
    planned_start: datetime
    planned_duration: int
    corrected_duration: Optional[int]
    impl_where: Optional[str]
    impl_what_first: Optional[str]
    created_at: datetime
    is_active: bool
    locked: bool = False
    has_checkin: bool = False

    model_config = {"from_attributes": True}


class RescheduleRequest(BaseModel):
    new_start: datetime
    confirmation_token: Optional[str] = None


class RescheduleResponseSchema(BaseModel):
    allowed: bool
    requires_confirmation: bool
    confirmation_token: Optional[str] = None
    reason: Optional[str] = None
    task: Optional[TaskResponse] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int


# ---------------------------------------------------------------------------
# Checkins
# ---------------------------------------------------------------------------

class CheckinCreate(BaseModel):
    task_id: UUID
    completed: Literal[0.0, 0.5, 1.0]
    actual_duration: Optional[int] = Field(None, gt=0, description="minutes")
    failure_reason: Optional[
        Literal["ran_out_of_time", "forgot", "chose_not_to", "external_blocker"]
    ] = None


class CheckinResponse(BaseModel):
    id: UUID
    task_id: UUID
    user_id: UUID
    completed: float
    actual_duration: Optional[int]
    failure_reason: Optional[str]
    checked_in_at: datetime

    model_config = {"from_attributes": True}


class CheckinHistoryResponse(BaseModel):
    checkins: list[CheckinResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class DeviceAssignmentResponse(BaseModel):
    id: UUID
    device_type: int
    beta_at_assignment: float
    assigned_at: datetime
    pre_completion_rate: Optional[float]

    model_config = {"from_attributes": True}


class DriftEventResponse(BaseModel):
    id: UUID
    detected_at: datetime
    drift_type: str
    direction: str
    beta_before: float
    beta_after: Optional[float]

    model_config = {"from_attributes": True}


class ProfileResponse(BaseModel):
    id: UUID
    email: str
    beta_proxy: float
    proj_bias_score: float
    current_device: int
    device_label: str
    streak: int
    completion_rate_14d: float
    drift_status: Literal["stable", "potential", "confirmed_decline", "confirmed_improvement"]
    created_at: datetime

    model_config = {"from_attributes": True}


class EmbeddingPoint(BaseModel):
    x: float
    y: float


class EmbeddingResponse(BaseModel):
    user: EmbeddingPoint
    # anonymized population sample for the scatter plot
    population: list[EmbeddingPoint]


# ---------------------------------------------------------------------------
# Notifications (internal)
# ---------------------------------------------------------------------------

class ReminderRequest(BaseModel):
    user_id: UUID
    task_id: UUID
