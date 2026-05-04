"""Linear probe mapping CF user embeddings → β proxy.

The CF model's 8-dim user embeddings are uninterpretable on their own.
We fit a linear probe on ``(embedding, β)`` pairs from the synthetic
data so that, at inference time, we can report a stable, calibrated
estimate of the user's present-bias.

Why Ridge (L2) and not plain OLS:
    Online incremental updates shift each user's embedding a little
    after every observed task outcome. Without an L2 penalty, the
    learned linear map can have very large coefficients in directions
    of low embedding variance, which makes the β output flap around
    every time the embedding moves. The L2 penalty trades a bit of
    in-sample R² for a much smoother, stable inference signal.

Outputs are clipped to ``(0, 1]`` because β has that domain in the
quasi-hyperbolic discounting model — values outside the unit interval
have no behavioral interpretation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


DEFAULT_RIDGE_ALPHA: float = 1.0
_BETA_FLOOR: float = 1e-4  # avoids exactly-zero β which would break β^d


class BetaProbe:
    """Wraps a sklearn ``Ridge`` regressor predicting β from user embeddings."""

    def __init__(self, alpha: float = DEFAULT_RIDGE_ALPHA) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative: {alpha}")
        self.alpha: float = float(alpha)
        self.model: Ridge = Ridge(alpha=self.alpha)
        self._fitted: bool = False

    # --- training ----------------------------------------------------------

    def fit(self, embeddings: np.ndarray, betas: np.ndarray) -> "BetaProbe":
        """Fit the probe on a frozen embedding matrix.

        Args:
            embeddings: ``(N, D)`` user-embedding matrix.
            betas:      ``(N,)`` ground-truth β values.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        if betas.ndim != 1 or betas.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "betas must be 1D and match embeddings.shape[0]: "
                f"emb={embeddings.shape}, β={betas.shape}"
            )

        self.model.fit(embeddings, betas)
        self._fitted = True

        train_score = float(self.model.score(embeddings, betas))
        logger.info(
            "Ridge probe fitted on %d users; in-sample R²=%.4f",
            embeddings.shape[0],
            train_score,
        )
        return self

    # --- inference ---------------------------------------------------------

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        """Return β proxy ∈ ``(0, 1]`` for one or more embeddings.

        Accepts either a single ``(D,)`` vector or a batched ``(N, D)``
        matrix; always returns a 1-D array.
        """
        if not self._fitted:
            raise RuntimeError("BetaProbe.predict called before fit/load.")

        x = embedding
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim != 2:
            raise ValueError(f"embedding must be 1D or 2D, got shape {x.shape}")

        raw = self.model.predict(x)
        return np.clip(raw, _BETA_FLOOR, 1.0).astype(np.float32)

    # --- persistence -------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the probe to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"alpha": self.alpha, "model": self.model, "fitted": self._fitted},
            path,
        )
        logger.info("Saved Ridge probe to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BetaProbe":
        """Load a previously saved probe."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Probe artifact not found: {path}")

        payload = joblib.load(path)
        probe = cls(alpha=float(payload["alpha"]))
        probe.model = payload["model"]
        probe._fitted = bool(payload["fitted"])
        logger.info("Loaded Ridge probe from %s", path)
        return probe

    # --- introspection -----------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted
