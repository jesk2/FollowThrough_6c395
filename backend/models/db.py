import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # behavioral profile
    beta_proxy = Column(Float, default=0.70, nullable=False)
    proj_bias_score = Column(Float, default=0.0, nullable=False)
    embedding_id = Column(Integer, nullable=True)  # index into CF embedding table
    current_device = Column(Integer, default=0, nullable=False)  # 0–4
    streak = Column(Integer, default=0, nullable=False)
    detector_state = Column(JSONB, nullable=True)

    tasks = relationship("Task", back_populates="user")
    checkins = relationship("Checkin", back_populates="user")
    device_assignments = relationship("DeviceAssignment", back_populates="user")
    drift_events = relationship("DriftEvent", back_populates="user")
    reschedule_events = relationship("RescheduleEvent", back_populates="user")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)  # academic | exercise | work | personal
    difficulty = Column(Integer, nullable=False)  # 1–5
    deadline_pressure = Column(String, nullable=False)  # today | this_week | none
    planned_start = Column(DateTime, nullable=False)
    planned_duration = Column(Integer, nullable=False)  # minutes
    corrected_duration = Column(Integer, nullable=True)  # set when proj_bias correction applied
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    locked = Column(Boolean, default=False, nullable=False)

    # Level 1 implementation intention fields
    impl_where = Column(String, nullable=True)
    impl_what_first = Column(String, nullable=True)

    user = relationship("User", back_populates="tasks")
    checkins = relationship("Checkin", back_populates="task")
    reschedule_events = relationship("RescheduleEvent", back_populates="task")


class Checkin(Base):
    __tablename__ = "checkins"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    completed = Column(Float, nullable=False)  # 0.0 | 0.5 | 1.0
    actual_duration = Column(Integer, nullable=True)  # minutes; null if completed=0
    failure_reason = Column(String, nullable=True)  # ran_out_of_time | forgot | chose_not_to | external_blocker
    checked_in_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    task = relationship("Task", back_populates="checkins")
    user = relationship("User", back_populates="checkins")


class DeviceAssignment(Base):
    """Audit log — one row every time a user's commitment device changes."""
    __tablename__ = "device_assignments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    device_type = Column(Integer, nullable=False)  # 0–4
    beta_at_assignment = Column(Float, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    pre_completion_rate = Column(Float, nullable=True)  # 14-day rate before assignment; for lift calc

    user = relationship("User", back_populates="device_assignments")


class DriftEvent(Base):
    """Log of BOCD-detected behavioral changes."""
    __tablename__ = "drift_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    drift_type = Column(String, nullable=False)  # level_shift | transient
    direction = Column(String, nullable=False)  # decline | improvement
    beta_before = Column(Float, nullable=False)
    beta_after = Column(Float, nullable=True)  # filled in after re-estimation

    user = relationship("User", back_populates="drift_events")


class RescheduleEvent(Base):
    """Audit log of confirmed reschedules of locked tasks."""
    __tablename__ = "reschedule_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    original_start = Column(DateTime, nullable=False)
    rescheduled_to = Column(DateTime, nullable=False)
    confirmed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    task = relationship("Task", back_populates="reschedule_events")
    user = relationship("User", back_populates="reschedule_events")
