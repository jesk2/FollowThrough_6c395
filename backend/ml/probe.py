"""
Linear probe: user embedding → beta proxy — IMPLEMENT THIS (Kaitlyn).

After pretraining, fit a Ridge regression from synthetic user embeddings to their known
true beta values. This probe is fixed after pretraining — it does not get retrained
as real users arrive.

The backend calls get_beta_proxy() from the checkins router.
Do not change that function signature.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# Fitted Ridge regression coefficients — loaded at startup after pretraining
_probe_weights: Optional[np.ndarray] = None  # shape (k,)
_probe_bias: float = 0.0


def fit_probe(embeddings: np.ndarray, beta_values: np.ndarray) -> None:
    """
    Fit the Ridge regression probe on synthetic data.

    Args:
        embeddings: shape (n_users, k) — synthetic user embeddings after pretraining
        beta_values: shape (n_users,)  — corresponding true beta values
    """
    raise NotImplementedError("Implement probe fitting — Kaitlyn")


def get_beta_proxy(user_embedding: np.ndarray) -> float:
    """
    Map a user embedding to an interpretable scalar beta proxy in [0, 1].

    Args:
        user_embedding: shape (k,)

    Returns:
        Scalar in [0, 1], clipped to valid range.
    """
    if _probe_weights is None:
        # probe not fitted yet — return population mean
        return 0.70
    raw = float(np.dot(_probe_weights, user_embedding) + _probe_bias)
    return float(np.clip(raw, 0.0, 1.0))
