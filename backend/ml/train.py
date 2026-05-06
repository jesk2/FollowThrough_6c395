"""Backend-facing training entry points — adapters over ``ml.training``.

Two modes:
    pretrain()            — full pass on synthetic data; produces the
                            artifacts in ``ml/artifacts/``. Run from the
                            repo root with ``python -m ml.training.pretrain``.
    incremental_update()  — 5–10 gradient steps on a single check-in,
                            freezing the task tower. Called from the
                            check-in router on every event.

The legacy ``incremental_update(user_id: str, task_features: ndarray, ...)``
signature is preserved so router code doesn't need to change. Internally
we look up the integer ``embedding_id`` via ``cf_model._user_store`` and
delegate to ``inference_api.incremental_update_api``.

For new code, prefer the typed entry point ``incremental_update_for_id``
which takes the already-resolved ``embedding_id`` and a Pydantic
``TaskFeatures`` directly.
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np

from ml.inference.inference_api import (
    TaskFeatures,
    incremental_update_api,
)

logger = logging.getLogger(__name__)


def pretrain(*_args, **_kwargs) -> None:
    """Run the full synthetic-data pretraining pipeline.

    Delegates to :mod:`ml.training.pretrain`. Should normally be invoked
    out-of-band (CLI: ``python -m ml.training.pretrain``) rather than
    from a request path — pretraining takes ~45 s on an H100 and writes
    artifacts under ``ml/artifacts/``.
    """
    from ml.training.pretrain import run_pretraining

    run_pretraining()


def incremental_update_for_id(
    embedding_id: int,
    features: TaskFeatures,
    completed: float,
) -> None:
    """Run one online update for an existing ``embedding_id``.

    Args:
        embedding_id: integer index from ``users.embedding_id``.
        features: typed task representation (see ``backend.ml.features.encode_task``).
        completed: 0.0 | 0.5 | 1.0. Passed through to BCE-with-logits.
            (Soft labels are accepted; the inference layer logs a warning.)
    """
    incremental_update_api(
        embedding_id=int(embedding_id),
        features=features,
        completed=float(completed),
    )


def incremental_update(
    user_id: Union[str, int],
    task_features: TaskFeatures,
    completed: float,
    n_steps: int = 10,  # accepted but ignored — see ml/training/incremental_update.py
) -> None:
    """Legacy adapter — preserved for the existing router call site.

    Args:
        user_id: should now be the integer ``embedding_id`` from
            ``users.embedding_id``. UUID strings are rejected with a
            clear error so callers update to the new convention.
        task_features: a ``TaskFeatures`` Pydantic model from
            ``backend.ml.features.encode_task``.
        completed: 0.0 | 0.5 | 1.0.
        n_steps: ignored. Step count is owned by ``IncrementalUpdateConfig``
            in the inference layer (currently 5).
    """
    if isinstance(user_id, str) and not user_id.isdigit():
        raise TypeError(
            "incremental_update now expects an integer embedding_id, "
            "not a UUID string. Pass user.embedding_id from the User row "
            "(allocate it via initialize_new_user if None)."
        )
    if isinstance(task_features, np.ndarray):
        raise TypeError(
            "task_features must be a TaskFeatures Pydantic model from "
            "backend.ml.features.encode_task. The 13-dim numpy stub is gone."
        )

    incremental_update_for_id(int(user_id), task_features, completed)


def full_retrain_all_users() -> None:
    """Hook for the weekly APScheduler job; not yet wired up.

    The plan is to retrain the entire CF model on accumulated real
    check-in data, updating the task tower as well. For now this is
    a no-op so the scheduler import doesn't break.
    """
    logger.info("full_retrain_all_users called — currently a no-op (TODO)")
