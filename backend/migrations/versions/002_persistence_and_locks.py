"""Persistence for BOCD detector state, Level-4 lock flag, reschedule audit log.

Revision ID: 002
Revises: 001
Create Date: 2026-05-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("detector_state", JSONB(), nullable=True),
    )
    op.add_column(
        "tasks",
        sa.Column("locked", sa.Boolean(), nullable=False, server_default="false"),
    )

    op.create_table(
        "reschedule_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("task_id", UUID(as_uuid=True), sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("original_start", sa.DateTime(), nullable=False),
        sa.Column("rescheduled_to", sa.DateTime(), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_reschedule_events_user_id", "reschedule_events", ["user_id"])
    op.create_index("ix_reschedule_events_task_id", "reschedule_events", ["task_id"])


def downgrade() -> None:
    op.drop_index("ix_reschedule_events_task_id", table_name="reschedule_events")
    op.drop_index("ix_reschedule_events_user_id", table_name="reschedule_events")
    op.drop_table("reschedule_events")
    op.drop_column("tasks", "locked")
    op.drop_column("users", "detector_state")
