"""Initial schema — all five tables.

Revision ID: 001
Revises:
Create Date: 2026-05-03
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("beta_proxy", sa.Float(), nullable=False, server_default="0.70"),
        sa.Column("proj_bias_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("embedding_id", sa.Integer(), nullable=True),
        sa.Column("current_device", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("streak", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "tasks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("deadline_pressure", sa.String(), nullable=False),
        sa.Column("planned_start", sa.DateTime(), nullable=False),
        sa.Column("planned_duration", sa.Integer(), nullable=False),
        sa.Column("corrected_duration", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("impl_where", sa.String(), nullable=True),
        sa.Column("impl_what_first", sa.String(), nullable=True),
    )
    op.create_index("ix_tasks_user_id", "tasks", ["user_id"])

    op.create_table(
        "checkins",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("task_id", UUID(as_uuid=True), sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("completed", sa.Float(), nullable=False),
        sa.Column("actual_duration", sa.Integer(), nullable=True),
        sa.Column("failure_reason", sa.String(), nullable=True),
        sa.Column("checked_in_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_checkins_task_id", "checkins", ["task_id"])
    op.create_index("ix_checkins_user_id", "checkins", ["user_id"])

    op.create_table(
        "device_assignments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("device_type", sa.Integer(), nullable=False),
        sa.Column("beta_at_assignment", sa.Float(), nullable=False),
        sa.Column("assigned_at", sa.DateTime(), nullable=False),
        sa.Column("pre_completion_rate", sa.Float(), nullable=True),
    )
    op.create_index("ix_device_assignments_user_id", "device_assignments", ["user_id"])

    op.create_table(
        "drift_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("detected_at", sa.DateTime(), nullable=False),
        sa.Column("drift_type", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("beta_before", sa.Float(), nullable=False),
        sa.Column("beta_after", sa.Float(), nullable=True),
    )
    op.create_index("ix_drift_events_user_id", "drift_events", ["user_id"])


def downgrade() -> None:
    op.drop_table("drift_events")
    op.drop_table("device_assignments")
    op.drop_table("checkins")
    op.drop_table("tasks")
    op.drop_table("users")
