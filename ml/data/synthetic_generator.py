"""Synthetic dataset for FollowThrough's behavioral-tracking model.

Generates a population of users with a latent present-bias parameter
``β ~ Beta(α, β)`` and a corpus of tasks per user. Task completion is
sampled from a quasi-hyperbolic-discounted Bernoulli model so the
downstream CF model has a learnable signal that ties user identity to
behavioral patterns conditional on task properties.

The generator returns:
    - per-user β values
    - per-task feature matrix (rows aligned with `user_ids`)
    - per-task binary completion labels
    - normalization statistics for the log-transformed features

Normalization stats are persisted alongside the dataset so identical
transformations can be applied to brand-new tasks at inference time.

Feature schema (column order matters — keep in sync with
``encode_task_features``):
    [0]      difficulty                     ∈ [0, 1]
    [1..4]   category one-hot               (4 categories)
    [5]      planned_duration   (log-normalized minutes)
    [6]      days_until_planned_start (log-normalized days)
    [7..9]   deadline_pressure one-hot      (today / this week / later)

Spec note: the product brief mentions a "13-dimensional" task vector
in the CF section, but the explicit feature breakdown above sums to 10.
We treat the explicit breakdown as the source of truth and expose
``TASK_FEATURE_DIM`` as a single constant consumed everywhere downstream
so the dimensionality stays consistent if the schema is later extended.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# --- Schema constants -------------------------------------------------------

NUM_CATEGORIES: int = 4
NUM_DEADLINE_BUCKETS: int = 3

TASK_FEATURE_DIM: int = (
    1                          # difficulty
    + NUM_CATEGORIES           # category one-hot
    + 1                        # planned_duration (normalized)
    + 1                        # days_until_planned_start (normalized)
    + NUM_DEADLINE_BUCKETS     # deadline_pressure one-hot
)  # = 10

# Deadline-bucket indices (kept symbolic so callers don't pass magic ints).
DEADLINE_TODAY: int = 0
DEADLINE_THIS_WEEK: int = 1
DEADLINE_LATER: int = 2


# --- Distributional defaults ------------------------------------------------

DEFAULT_BETA_ALPHA: float = 7.0
DEFAULT_BETA_BETA: float = 3.0

# Raw planned_duration ~ LogNormal(μ=3.5, σ=0.8) minutes
DURATION_MU: float = 3.5
DURATION_SIGMA: float = 0.8

# Raw days_until_planned_start ~ Poisson(λ=2)
DELAY_LAMBDA: float = 2.0

# Label model: P(y=1) = clip(base · (1 − λ_d · difficulty) · β^d, 0, 1)
LABEL_BASE_PROB: float = 0.9
LABEL_DIFFICULTY_SCALE: float = 0.5  # λ in (1 − λ · difficulty)


# --- Public dataclasses -----------------------------------------------------


@dataclass
class FeatureStats:
    """Normalization statistics for the log-transformed features.

    These must be re-applied verbatim at inference time so unseen tasks
    land in the same input space the model was trained on.
    """

    duration_log_mean: float
    duration_log_std: float
    delay_log_mean: float
    delay_log_std: float

    def to_dict(self) -> dict:
        return {
            "duration_log_mean": self.duration_log_mean,
            "duration_log_std": self.duration_log_std,
            "delay_log_mean": self.delay_log_mean,
            "delay_log_std": self.delay_log_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        return cls(
            duration_log_mean=float(d["duration_log_mean"]),
            duration_log_std=float(d["duration_log_std"]),
            delay_log_mean=float(d["delay_log_mean"]),
            delay_log_std=float(d["delay_log_std"]),
        )


@dataclass
class SyntheticDataset:
    """Container for a generated dataset; arrays are aligned by row index."""

    user_ids: np.ndarray          # (N_tasks,) int64
    betas: np.ndarray             # (N_users,) float32
    task_features: np.ndarray     # (N_tasks, TASK_FEATURE_DIM) float32
    labels: np.ndarray            # (N_tasks,) float32 in {0, 1}
    raw_delays: np.ndarray        # (N_tasks,) int64 — preserved for analysis
    feature_stats: FeatureStats


# --- Internals --------------------------------------------------------------


def _sample_deadline_pressure(d_raw: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a one-hot deadline-pressure vector conditioned on raw delay.

    Probabilities (today / this_week / later):
        d == 0       → (0.80, 0.15, 0.05)
        1 ≤ d ≤ 6    → (0.00, 0.80, 0.20)
        d > 6        → (0.00, 0.00, 1.00)
    """
    if d_raw == 0:
        probs = (0.80, 0.15, 0.05)
    elif 1 <= d_raw <= 6:
        probs = (0.00, 0.80, 0.20)
    else:
        probs = (0.00, 0.00, 1.00)

    bucket = int(rng.choice(NUM_DEADLINE_BUCKETS, p=probs))
    one_hot = np.zeros(NUM_DEADLINE_BUCKETS, dtype=np.float32)
    one_hot[bucket] = 1.0
    return one_hot


# --- Public API -------------------------------------------------------------


def generate_dataset(
    num_users: int = 500,
    tasks_per_user: int = 50,
    seed: Optional[int] = 42,
    beta_alpha: float = DEFAULT_BETA_ALPHA,
    beta_beta: float = DEFAULT_BETA_BETA,
    label_base: float = LABEL_BASE_PROB,
    label_difficulty_scale: float = LABEL_DIFFICULTY_SCALE,
) -> SyntheticDataset:
    """Generate a synthetic dataset of users and tasks.

    Args:
        num_users: Number of synthetic users.
        tasks_per_user: Tasks generated per user.
        seed: RNG seed for reproducibility.
        beta_alpha, beta_beta: Beta-distribution shape parameters for β.
        label_base: Base success probability cap (β=1, difficulty=0, d=0).
        label_difficulty_scale: λ in (1 − λ · difficulty).

    Returns:
        A populated ``SyntheticDataset``.
    """
    if num_users <= 0:
        raise ValueError(f"num_users must be positive: {num_users}")
    if tasks_per_user <= 0:
        raise ValueError(f"tasks_per_user must be positive: {tasks_per_user}")

    rng = np.random.default_rng(seed)
    n_tasks = num_users * tasks_per_user

    # 1. Per-user β
    betas = rng.beta(beta_alpha, beta_beta, size=num_users).astype(np.float32)

    # 2. Per-task raw values
    user_ids = np.repeat(np.arange(num_users, dtype=np.int64), tasks_per_user)
    difficulty = rng.uniform(0.0, 1.0, size=n_tasks).astype(np.float32)
    category_idx = rng.integers(0, NUM_CATEGORIES, size=n_tasks)
    duration_raw = rng.lognormal(mean=DURATION_MU, sigma=DURATION_SIGMA, size=n_tasks)
    delay_raw = rng.poisson(DELAY_LAMBDA, size=n_tasks).astype(np.int64)

    # 3. Compute (and persist) normalization stats from this dataset
    duration_log = np.log(duration_raw)
    delay_log = np.log(delay_raw + 1.0)  # +1 to keep d=0 finite
    stats = FeatureStats(
        duration_log_mean=float(duration_log.mean()),
        duration_log_std=float(duration_log.std() + 1e-8),
        delay_log_mean=float(delay_log.mean()),
        delay_log_std=float(delay_log.std() + 1e-8),
    )
    duration_norm = (
        (duration_log - stats.duration_log_mean) / stats.duration_log_std
    ).astype(np.float32)
    delay_norm = (
        (delay_log - stats.delay_log_mean) / stats.delay_log_std
    ).astype(np.float32)

    # 4. One-hots
    category_onehot = np.zeros((n_tasks, NUM_CATEGORIES), dtype=np.float32)
    category_onehot[np.arange(n_tasks), category_idx] = 1.0

    deadline_onehot = np.stack(
        [_sample_deadline_pressure(int(d), rng) for d in delay_raw], axis=0
    ).astype(np.float32)

    # 5. Assemble feature matrix in the documented column order
    features = np.concatenate(
        [
            difficulty[:, None],
            category_onehot,
            duration_norm[:, None],
            delay_norm[:, None],
            deadline_onehot,
        ],
        axis=1,
    ).astype(np.float32)

    if features.shape != (n_tasks, TASK_FEATURE_DIM):
        raise AssertionError(
            f"Feature matrix shape mismatch: expected {(n_tasks, TASK_FEATURE_DIM)}, "
            f"got {features.shape}"
        )

    # 6. Labels: quasi-hyperbolic-discounted Bernoulli
    user_beta = betas[user_ids]
    p_success = (
        label_base
        * (1.0 - label_difficulty_scale * difficulty)
        * np.power(user_beta, delay_raw.astype(np.float32))
    )
    p_success = np.clip(p_success, 0.0, 1.0)
    labels = (rng.uniform(0.0, 1.0, size=n_tasks) < p_success).astype(np.float32)

    logger.info(
        "Generated %d tasks across %d users (positive rate=%.3f, β mean=%.3f).",
        n_tasks,
        num_users,
        float(labels.mean()),
        float(betas.mean()),
    )

    return SyntheticDataset(
        user_ids=user_ids,
        betas=betas,
        task_features=features,
        labels=labels,
        raw_delays=delay_raw,
        feature_stats=stats,
    )


def encode_task_features(
    *,
    difficulty: float,
    category_index: int,
    planned_duration_minutes: float,
    days_until_planned_start: int,
    deadline_pressure_index: int,
    stats: FeatureStats,
) -> np.ndarray:
    """Encode a single raw task into the model's feature vector.

    Used at inference time on new, unseen tasks. The normalization stats
    must be the ones produced during dataset construction (loaded from
    ``feature_stats.json``).

    Returns:
        ``np.ndarray`` of shape ``(TASK_FEATURE_DIM,)`` and dtype float32.
    """
    if not (0 <= category_index < NUM_CATEGORIES):
        raise ValueError(f"category_index out of range: {category_index}")
    if not (0 <= deadline_pressure_index < NUM_DEADLINE_BUCKETS):
        raise ValueError(
            f"deadline_pressure_index out of range: {deadline_pressure_index}"
        )
    if planned_duration_minutes <= 0:
        raise ValueError(
            f"planned_duration_minutes must be positive: {planned_duration_minutes}"
        )
    if days_until_planned_start < 0:
        raise ValueError(
            f"days_until_planned_start must be ≥ 0: {days_until_planned_start}"
        )
    if not (0.0 <= difficulty <= 1.0):
        raise ValueError(f"difficulty must be in [0, 1]: {difficulty}")

    duration_norm = (
        np.log(planned_duration_minutes) - stats.duration_log_mean
    ) / stats.duration_log_std
    delay_norm = (
        np.log(days_until_planned_start + 1.0) - stats.delay_log_mean
    ) / stats.delay_log_std

    cat_onehot = np.zeros(NUM_CATEGORIES, dtype=np.float32)
    cat_onehot[category_index] = 1.0
    deadline_onehot = np.zeros(NUM_DEADLINE_BUCKETS, dtype=np.float32)
    deadline_onehot[deadline_pressure_index] = 1.0

    vec = np.concatenate(
        [
            np.array([difficulty], dtype=np.float32),
            cat_onehot,
            np.array([duration_norm], dtype=np.float32),
            np.array([delay_norm], dtype=np.float32),
            deadline_onehot,
        ]
    ).astype(np.float32)

    if vec.shape != (TASK_FEATURE_DIM,):
        raise AssertionError(
            f"Encoded feature vector shape mismatch: expected ({TASK_FEATURE_DIM},), "
            f"got {vec.shape}"
        )
    return vec
