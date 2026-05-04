"""
Training logic — IMPLEMENT THIS (Kaitlyn).

Two modes:
  1. pretrain()          — full pass on synthetic data to initialize model weights
  2. incremental_update() — 5–10 gradient steps on a single new observation,
                            freezing the task encoder, only updating user embedding + bias

The backend calls incremental_update() from the checkins router on every check-in.
Do not change that function signature.
"""
from __future__ import annotations

import numpy as np


def pretrain(observations) -> None:
    """
    Train the CF model on the full synthetic dataset.

    Args:
        observations: list of SyntheticObservation from synthetic.generate_dataset()

    Saves model weights to disk so they survive server restarts.
    """
    raise NotImplementedError("Implement pretraining loop — Kaitlyn")


def incremental_update(
    user_id: str,
    task_features: np.ndarray,
    completed: float,
    n_steps: int = 10,
) -> None:
    """
    Update the user's embedding given one new observation.

    Rules:
      - Freeze the task encoder (shared weights) — only update user embedding + user bias.
      - Run n_steps gradient steps with binary cross-entropy loss.
      - Persist the updated embedding via cf_model.set_user_embedding().

    Args:
        user_id: string UUID of the user
        task_features: 13-dim feature vector from features.encode_task()
        completed: 0.0, 0.5, or 1.0
        n_steps: number of gradient steps (default 10)
    """
    raise NotImplementedError("Implement incremental update — Kaitlyn")


def full_retrain_all_users() -> None:
    """
    Full retrain on all accumulated real data, updating task encoder as well.
    Called by the weekly APScheduler job. Kaitlyn implements this.
    """
    raise NotImplementedError("Implement full retrain — Kaitlyn")
