"""Backend-facing inference contract for FollowThrough.

This module is the single entry point the backend should depend on.
It hides:
    - PyTorch tensors (callers see Python floats / Pydantic models)
    - artifact-loading details
    - the mutable user-embedding store

Public API (matches the spec exactly):
    initialize_new_user()                            -> int
    get_user_state(embedding_id)                     -> float    # β proxy
    predict_task(embedding_id, features)             -> float    # P(complete)
    incremental_update_api(embedding_id, features, completed) -> None

All four are thread-safe (the underlying store uses an internal lock and
the model is read-only at inference time).

Artifact directory:
    By default we read/write under ``ml/artifacts``. Tests can call
    ``configure_artifact_dir`` to redirect to a temp directory before
    calling any of the four public functions.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from ml.data.synthetic_generator import (
    NUM_CATEGORIES,
    NUM_DEADLINE_BUCKETS,
    FeatureStats,
    encode_task_features,
)
from ml.models.cf_model import CollaborativeFilteringModel
from ml.models.linear_probe import BetaProbe
from ml.training.incremental_update import (
    IncrementalUpdateConfig,
    update_user_state,
)
from ml.training.pretrain import (
    CF_MODEL_FILE,
    DEFAULT_ARTIFACT_DIR,
    FEATURE_STATS_FILE,
    PROBE_FILE,
    USER_STATE_FILE,
)

logger = logging.getLogger(__name__)


# --- Pydantic contract ------------------------------------------------------


class TaskFeatures(BaseModel):
    """Strict, raw-domain task representation accepted by the inference API.

    Every field is in its raw / pre-encoding domain — the inference layer
    applies the exact same normalizations the model was trained with,
    using the persisted ``FeatureStats``.
    """

    difficulty: float = Field(
        ..., ge=0.0, le=1.0, description="Subjective difficulty in [0, 1]."
    )
    category_index: int = Field(
        ...,
        ge=0,
        lt=NUM_CATEGORIES,
        description=f"Integer category index in [0, {NUM_CATEGORIES}).",
    )
    planned_duration_minutes: float = Field(
        ..., gt=0.0, description="Planned duration in minutes (positive)."
    )
    days_until_planned_start: int = Field(
        ..., ge=0, description="Whole days until planned start (≥ 0)."
    )
    deadline_pressure_index: int = Field(
        ...,
        ge=0,
        lt=NUM_DEADLINE_BUCKETS,
        description="0 = today, 1 = this week, 2 = later.",
    )


class PredictionResult(BaseModel):
    """Rich prediction container.

    Not directly returned by ``predict_task`` (which returns a bare float
    to match the spec'd signature) but useful for callers that want both
    the completion probability and the user's current β proxy in one go.
    """

    completion_probability: float = Field(..., ge=0.0, le=1.0)
    beta_proxy: float = Field(..., gt=0.0, le=1.0)


# --- Internal user-state store ---------------------------------------------


class _UserStore:
    """Thread-safe, disk-backed store of dynamic per-user state.

    Pretraining produces an ``N × D`` embedding matrix and an ``N``-vector
    of biases; new sign-ups grow these by appending a row initialized to
    the population mean. Every mutation is immediately persisted so the
    next process restart picks up the up-to-date table.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

        if not self.path.exists():
            raise FileNotFoundError(
                f"User-state artifact not found: {self.path}. "
                "Run ml.training.pretrain first."
            )

        payload = torch.load(self.path, map_location="cpu")
        self.embeddings: torch.Tensor = payload["embeddings"].float()
        self.biases: torch.Tensor = payload["biases"].float()
        self.population_mean_emb: torch.Tensor = payload[
            "population_mean_emb"
        ].float()
        self.population_mean_bias: float = float(payload["population_mean_bias"])
        self._next_user_id: int = int(payload["next_user_id"])

    # --- mutators ----------------------------------------------------------

    def add_new_user(self) -> int:
        """Append a row initialized to the population mean. Returns its id."""
        with self._lock:
            new_id = self._next_user_id
            self._next_user_id += 1
            new_row = self.population_mean_emb.unsqueeze(0).clone()
            self.embeddings = torch.cat([self.embeddings, new_row], dim=0)
            self.biases = torch.cat(
                [self.biases, torch.tensor([self.population_mean_bias])], dim=0
            )
            self._save_locked()
            logger.info("Initialized new user with id=%d (population mean).", new_id)
            return new_id

    def set(self, user_id: int, emb: np.ndarray, bias: float) -> None:
        with self._lock:
            self._check_id_locked(user_id)
            self.embeddings[user_id] = torch.from_numpy(emb.astype(np.float32))
            self.biases[user_id] = float(bias)
            self._save_locked()

    # --- accessors ---------------------------------------------------------

    def get(self, user_id: int) -> Tuple[np.ndarray, float]:
        with self._lock:
            self._check_id_locked(user_id)
            return (
                self.embeddings[user_id].cpu().numpy().astype(np.float32),
                float(self.biases[user_id].item()),
            )

    @property
    def emb_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def num_users(self) -> int:
        return int(self.embeddings.shape[0])

    # --- helpers -----------------------------------------------------------

    def _check_id_locked(self, user_id: int) -> None:
        if not (0 <= user_id < self.embeddings.shape[0]):
            raise KeyError(
                f"Unknown user_id={user_id} "
                f"(valid range 0..{self.embeddings.shape[0] - 1})"
            )

    def _save_locked(self) -> None:
        torch.save(
            {
                "embeddings": self.embeddings,
                "biases": self.biases,
                "population_mean_emb": self.population_mean_emb,
                "population_mean_bias": self.population_mean_bias,
                "next_user_id": self._next_user_id,
            },
            self.path,
        )


# --- Singleton infrastructure ----------------------------------------------

_LOCK = threading.Lock()
_MODEL: Optional[CollaborativeFilteringModel] = None
_PROBE: Optional[BetaProbe] = None
_FEATURE_STATS: Optional[FeatureStats] = None
_STORE: Optional[_UserStore] = None
_ARTIFACT_DIR: Path = Path(DEFAULT_ARTIFACT_DIR)


def configure_artifact_dir(path: Union[Path, str]) -> None:
    """Override the artifact directory.

    Must be called before any of the four public API functions; resets
    the cached singletons.
    """
    global _ARTIFACT_DIR, _MODEL, _PROBE, _FEATURE_STATS, _STORE
    with _LOCK:
        _ARTIFACT_DIR = Path(path)
        _MODEL = None
        _PROBE = None
        _FEATURE_STATS = None
        _STORE = None
        logger.info("Inference artifact directory set to %s", _ARTIFACT_DIR)


def _ensure_loaded() -> None:
    """Lazy-load the CF model, Ridge probe, feature stats, and user store."""
    global _MODEL, _PROBE, _FEATURE_STATS, _STORE

    if (
        _MODEL is not None
        and _PROBE is not None
        and _FEATURE_STATS is not None
        and _STORE is not None
    ):
        return

    with _LOCK:
        if _MODEL is None:
            cf_path = _ARTIFACT_DIR / CF_MODEL_FILE
            if not cf_path.exists():
                raise FileNotFoundError(
                    f"CF artifact missing at {cf_path}. "
                    "Run ml.training.pretrain first."
                )
            payload = torch.load(cf_path, map_location="cpu")
            cfg = payload["config"]
            model = CollaborativeFilteringModel(
                num_users=cfg["num_users"],
                task_feature_dim=cfg["task_feature_dim"],
                user_emb_dim=cfg["user_emb_dim"],
                task_hidden_dim=cfg.get("task_hidden_dim", 16),
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            _MODEL = model
            logger.info("Loaded CF model from %s", cf_path)

        if _PROBE is None:
            _PROBE = BetaProbe.load(_ARTIFACT_DIR / PROBE_FILE)

        if _FEATURE_STATS is None:
            stats_path = _ARTIFACT_DIR / FEATURE_STATS_FILE
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"Feature-stats artifact missing at {stats_path}."
                )
            with open(stats_path) as f:
                _FEATURE_STATS = FeatureStats.from_dict(json.load(f))
            logger.info("Loaded feature stats from %s", stats_path)

        if _STORE is None:
            _STORE = _UserStore(_ARTIFACT_DIR / USER_STATE_FILE)
            logger.info(
                "Loaded user store with %d users from %s",
                _STORE.num_users,
                _ARTIFACT_DIR / USER_STATE_FILE,
            )


def _encode(features: TaskFeatures) -> np.ndarray:
    assert _FEATURE_STATS is not None  # guaranteed by _ensure_loaded
    return encode_task_features(
        difficulty=features.difficulty,
        category_index=features.category_index,
        planned_duration_minutes=features.planned_duration_minutes,
        days_until_planned_start=features.days_until_planned_start,
        deadline_pressure_index=features.deadline_pressure_index,
        stats=_FEATURE_STATS,
    )


# --- Public API -------------------------------------------------------------


def initialize_new_user() -> int:
    """Allocate a new user embedding initialized to the population mean.

    Returns:
        The newly assigned ``embedding_id`` (an integer). The id is
        monotonically increasing and persistent across process restarts.
    """
    _ensure_loaded()
    assert _STORE is not None
    return _STORE.add_new_user()


def get_user_state(embedding_id: int) -> float:
    """Return the Ridge-probe estimate of β ∈ ``(0, 1]`` for an existing user."""
    _ensure_loaded()
    assert _STORE is not None and _PROBE is not None
    emb, _ = _STORE.get(embedding_id)
    beta = float(_PROBE.predict(emb)[0])
    return beta


def predict_task(embedding_id: int, features: TaskFeatures) -> float:
    """Return the predicted completion probability ∈ ``[0, 1]`` for a task."""
    _ensure_loaded()
    assert _MODEL is not None and _STORE is not None

    emb, bias = _STORE.get(embedding_id)
    feat_vec = _encode(features)

    with torch.no_grad():
        emb_t = torch.from_numpy(emb).float().unsqueeze(0)
        bias_t = torch.tensor([bias], dtype=torch.float32)
        feat_t = torch.from_numpy(feat_vec).float().unsqueeze(0)
        logit = _MODEL.score(emb_t, bias_t, feat_t)
        prob = torch.sigmoid(logit).item()

    return float(prob)


def incremental_update_api(
    embedding_id: int,
    features: TaskFeatures,
    completed: float,
) -> None:
    """Run an online update for a single user given one observed outcome.

    Args:
        embedding_id: Existing user id.
        features: The task that was attempted.
        completed: 1.0 if the user completed it, 0.0 otherwise.
    """
    _ensure_loaded()
    assert _MODEL is not None and _STORE is not None

    if completed not in (0.0, 1.0):
        raise ValueError(f"completed must be 0.0 or 1.0; got {completed!r}")

    emb, bias = _STORE.get(embedding_id)
    feat_vec = _encode(features)

    new_emb, new_bias = update_user_state(
        model=_MODEL,
        user_emb=emb,
        user_bias=bias,
        task_features=feat_vec,
        completed=float(completed),
        cfg=IncrementalUpdateConfig(),
    )
    _STORE.set(embedding_id, new_emb, new_bias)
