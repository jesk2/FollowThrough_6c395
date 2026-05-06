"""Backend-facing CF interface — implemented as a thin adapter over
``ml.inference.inference_api``.

What the routers expect from this module:
    get_user_embedding(embedding_id)              -> np.ndarray | None
    predict_completion(embedding_id, features)    -> float
    get_population_embeddings_2d(embedding_id, e) -> (user_2d, pop_2d)

The inference layer is the source of truth for model weights, the
user-state table, feature normalization stats, and concurrency safety.
This module's job is to translate between the backend's call shapes
and that layer.

ID convention: backend code passes the *integer* ``embedding_id`` from
``users.embedding_id`` — not the UUID. New users must allocate one via
``ml.inference.inference_api.initialize_new_user`` before the first
prediction call (see ``backend/routers/checkins.py``).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from ml.inference.inference_api import (
    TaskFeatures,
    _ensure_loaded,
    predict_task,
)

logger = logging.getLogger(__name__)

# 8-dim user embedding, fixed by the trained model architecture.
K: int = 8

# Lazy PCA cache for the /profile/embedding endpoint. Recomputed on
# first call after model load; reused thereafter.
_pca_cache: Optional[dict] = None


def _user_store():
    """Return the loaded ml.inference user store (private accessor)."""
    from ml.inference import inference_api as api
    _ensure_loaded()
    assert api._STORE is not None  # populated by _ensure_loaded
    return api._STORE


def get_user_embedding(embedding_id: Optional[int]) -> Optional[np.ndarray]:
    """Return the current k-dim embedding for ``embedding_id``.

    Returns ``None`` if the id is missing or unknown — callers should
    interpret that as "user has not been allocated yet" and skip the
    prediction (or run ``initialize_new_user`` first).
    """
    if embedding_id is None:
        return None
    try:
        emb, _ = _user_store().get(int(embedding_id))
        return emb
    except KeyError:
        logger.warning("Unknown embedding_id=%s; returning None", embedding_id)
        return None


def predict_completion(embedding_id: int, features: TaskFeatures) -> float:
    """Return ``P(complete)`` ∈ [0, 1] for a (user, task) pair."""
    return predict_task(int(embedding_id), features)


def get_population_embeddings_2d(
    focal_embedding_id: int,
    focal_embedding: np.ndarray,
    sample_size: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Project the focal user's emb + an anonymized sample to 2D via PCA.

    The PCA is fit once over the entire user-state matrix and cached
    in this module. Subsequent calls re-use it, so the same user keeps
    landing at the same coordinates across requests. The cache is
    invalidated when the population grows (see ``invalidate_pca_cache``).
    """
    global _pca_cache

    store = _user_store()
    all_embs = store.embeddings.cpu().numpy()  # (N, K)

    if _pca_cache is None or _pca_cache["n_users"] != all_embs.shape[0]:
        pca = PCA(n_components=2, random_state=seed)
        proj = pca.fit_transform(all_embs)
        _pca_cache = {
            "n_users": all_embs.shape[0],
            "pca": pca,
            "all_2d": proj,
        }
        logger.info("Refit PCA for /profile/embedding: %d users", all_embs.shape[0])

    pca: PCA = _pca_cache["pca"]
    all_2d: np.ndarray = _pca_cache["all_2d"]

    if 0 <= int(focal_embedding_id) < all_2d.shape[0]:
        user_2d = all_2d[int(focal_embedding_id)]
    else:
        user_2d = pca.transform(focal_embedding[None, :])[0]

    rng = np.random.default_rng(seed)
    n = all_2d.shape[0]
    if n <= sample_size:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)
    pop_2d = [all_2d[i] for i in idx if i != int(focal_embedding_id)]

    return user_2d, pop_2d


def invalidate_pca_cache() -> None:
    """Drop the cached PCA. Call after batch sign-ups so new users land correctly."""
    global _pca_cache
    _pca_cache = None
