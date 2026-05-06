"""Backend-facing β-probe interface — adapter over ``ml.inference``.

The Ridge probe was fit during pretraining (see ``ml/training/pretrain.py``)
and shipped as ``ml/artifacts/ridge_probe.joblib``. The inference layer
loads it lazily on first use; this module just exposes the call
shape the routers expect.

There are two ways to ask for a β estimate:

    get_beta_for_user(embedding_id)   — preferred; uses the live store
    get_beta_proxy(user_embedding)    — legacy stub-shaped signature

Use the first whenever the caller has the user's ``embedding_id`` (which
is everywhere — it's on the User row). The second exists only because
the original placeholder stub took a raw embedding vector; we keep it
working for backward compatibility but it always returns the same value
as ``get_beta_for_user`` would for the matching id.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ml.inference.inference_api import _ensure_loaded, get_user_state

logger = logging.getLogger(__name__)

POPULATION_MEAN_BETA: float = 0.70  # fallback when probe is unavailable


def _probe():
    """Return the loaded BetaProbe (private accessor)."""
    from ml.inference import inference_api as api
    _ensure_loaded()
    assert api._PROBE is not None
    return api._PROBE


def get_beta_for_user(embedding_id: Optional[int]) -> float:
    """β proxy for a known user. Returns the population mean if id is missing."""
    if embedding_id is None:
        return POPULATION_MEAN_BETA
    try:
        return get_user_state(int(embedding_id))
    except KeyError:
        logger.warning(
            "Unknown embedding_id=%s in get_beta_for_user; returning population mean",
            embedding_id,
        )
        return POPULATION_MEAN_BETA


def get_beta_proxy(user_embedding: np.ndarray) -> float:
    """Map a raw embedding vector → β proxy via the Ridge probe.

    Kept for backward compatibility with the original stub signature.
    Prefer ``get_beta_for_user(embedding_id)`` when you have the id.
    """
    if user_embedding is None:
        return POPULATION_MEAN_BETA
    try:
        return float(_probe().predict(np.asarray(user_embedding, dtype=np.float32))[0])
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Probe predict failed (%s); returning population mean", exc)
        return POPULATION_MEAN_BETA
