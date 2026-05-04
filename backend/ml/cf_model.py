"""
Collaborative filtering model — IMPLEMENT THIS (Kaitlyn).

Architecture:
  - User embedding table: k=8 dimensional vector per user
  - Task encoder: linear layer mapping 13-dim task features → k dims
  - P(complete) = sigmoid(u_i · v_j + b_user + b_task)

The backend calls get_user_embedding() and get_population_embeddings_2d().
Do not change those function signatures.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Model state — replace with real PyTorch model + weight loading
# ---------------------------------------------------------------------------

# Maps user_id (str) → numpy embedding vector of shape (k,)
_user_embeddings: dict[str, np.ndarray] = {}

K = 8  # embedding dimension


def get_user_embedding(user_id: str) -> Optional[np.ndarray]:
    """Return the current k-dimensional embedding for user_id, or None if not yet fitted."""
    return _user_embeddings.get(user_id)


def set_user_embedding(user_id: str, embedding: np.ndarray) -> None:
    """Persist an updated embedding for user_id (called by train.py after gradient steps)."""
    assert embedding.shape == (K,), f"Expected shape ({K},), got {embedding.shape}"
    _user_embeddings[user_id] = embedding


def predict_completion(user_id: str, task_features: np.ndarray) -> float:
    """
    Return P(user completes task) in [0, 1].

    Args:
        user_id: string UUID of the user
        task_features: 13-dim feature vector from features.encode_task()

    Returns:
        Scalar completion probability.
    """
    raise NotImplementedError("Implement CompletionModel forward pass — Kaitlyn")


def get_population_embeddings_2d(
    focal_user_id: str,
    focal_embedding: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Return the focal user's 2D PCA projection and an anonymized sample of the population.

    Args:
        focal_user_id: the user whose position to return
        focal_embedding: their current embedding (already fetched by caller)

    Returns:
        (user_2d, pop_2d) where user_2d is shape (2,) and pop_2d is list of (2,) arrays.
    """
    raise NotImplementedError("Implement PCA projection — Kaitlyn")
