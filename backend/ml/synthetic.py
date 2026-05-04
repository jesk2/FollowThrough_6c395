"""
Synthetic data generation — IMPLEMENT THIS (Kaitlyn).

Generates 500 synthetic users × 50 tasks = 25,000 training examples.

User beta values are drawn from Beta(7, 3) — mean ~0.70, matching published literature.
Completion probability for each task uses the quasi-hyperbolic model:
    p = base_rate * difficulty_factor * beta^days_until
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterator


@dataclass
class SyntheticObservation:
    user_id: int
    beta_true: float
    task_features: np.ndarray  # shape (13,)
    completed: float           # 0.0 or 1.0


def generate_dataset(
    n_users: int = 500,
    tasks_per_user: int = 50,
    seed: int = 42,
) -> list[SyntheticObservation]:
    """
    Generate the pretraining dataset.

    Returns a list of SyntheticObservation, one per (user, task) pair.
    """
    raise NotImplementedError("Implement synthetic data generation — Kaitlyn")


def generate_user_betas(n_users: int, seed: int = 42) -> np.ndarray:
    """Draw beta values from Beta(7, 3). Returns shape (n_users,)."""
    rng = np.random.default_rng(seed)
    return rng.beta(7, 3, size=n_users)
