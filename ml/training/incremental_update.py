"""Online incremental update for a single user's embedding.

After a real task completes, we want the user's β estimate to move in
the direction implied by the observed outcome. We do this by running a
handful of gradient steps on **only** that user's 8-D embedding and
scalar bias — the task encoder, task-bias head, and every other user's
embedding stay frozen.

Why not refit the whole model?
    - Updates need to be cheap (sub-millisecond order per event).
    - The shared task representation should stay globally consistent.
    - Identifiability of β through the linear probe depends on the
      embedding space remaining stable.

Two entry points are exposed:
    ``update_user_state`` — operates on a pre-loaded model, used by the
    inference singleton in ``ml.inference.inference_api``.

    ``update_user_state_from_disk`` — convenience wrapper that loads the
    CF model from the standard artifact path. Useful for one-off scripts
    or tests; production callers should reuse a loaded model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn

from ml.models.cf_model import CollaborativeFilteringModel

logger = logging.getLogger(__name__)


@dataclass
class IncrementalUpdateConfig:
    """Hyperparameters for online updates.

    The defaults are tuned empirically: ``lr=0.05, num_steps=8`` (the
    naive choice) overshoots and *degrades* held-out AUC vs. the
    population-mean baseline. Gentler updates close the gap to the
    pretrained-with-full-history embedding within ~30 events.
    """

    num_steps: int = 5         # spec: 5–10
    lr: float = 0.01           # step size on the user's params
    weight_decay: float = 1e-4 # mild L2 keeps embeddings near origin


# --- Core update -----------------------------------------------------------


def update_user_state(
    model: CollaborativeFilteringModel,
    user_emb: np.ndarray,
    user_bias: float,
    task_features: np.ndarray,
    completed: float,
    cfg: IncrementalUpdateConfig = IncrementalUpdateConfig(),
) -> Tuple[np.ndarray, float]:
    """Run a few SGD steps on a single user's embedding & bias.

    The model itself is treated as read-only. We freeze its task tower
    as a defensive guard, and we score using the externally-supplied
    user vectors (via ``model.score``) so the model's internal user
    table is also untouched.

    Args:
        model: A loaded ``CollaborativeFilteringModel`` (will be set to eval).
        user_emb: Current 8-D user embedding (copied internally).
        user_bias: Current scalar user bias.
        task_features: 1-D feature vector for the observed task.
        completed: 1.0 if the user completed the task, 0.0 otherwise.
            Soft targets in [0, 1] are accepted (BCE supports them) but
            warned about, since the production pipeline emits binary labels.
        cfg: Hyperparameters.

    Returns:
        ``(new_user_emb, new_user_bias)``. The caller is responsible for
        persisting these back into its store.
    """
    if user_emb.ndim != 1:
        raise ValueError(f"user_emb must be 1D, got shape {user_emb.shape}")
    if task_features.ndim != 1:
        raise ValueError(f"task_features must be 1D, got shape {task_features.shape}")
    if cfg.num_steps <= 0:
        raise ValueError(f"num_steps must be positive: {cfg.num_steps}")
    if not (0.0 <= float(completed) <= 1.0):
        raise ValueError(f"completed must be in [0, 1]: {completed}")
    if completed not in (0.0, 1.0):
        logger.warning(
            "Non-binary completion label %.3f — treating as soft target.",
            float(completed),
        )

    device = next(model.parameters()).device

    # Freeze the entire model. We then score using local Parameters that
    # live outside the model's graph, so backprop only touches them.
    model.eval()
    model.freeze_task_tower()
    for p in (model.user_emb.weight, model.user_bias.weight):
        p.requires_grad_(False)

    emb_param = nn.Parameter(
        torch.from_numpy(user_emb.astype(np.float32)).to(device).clone()
    )
    bias_param = nn.Parameter(
        torch.tensor(float(user_bias), device=device, dtype=torch.float32)
    )

    feat_tensor = (
        torch.from_numpy(task_features.astype(np.float32)).to(device).unsqueeze(0)
    )  # (1, F)
    label_tensor = torch.tensor(
        [float(completed)], device=device, dtype=torch.float32
    )

    optimizer = torch.optim.SGD(
        [emb_param, bias_param],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    for step in range(cfg.num_steps):
        optimizer.zero_grad()
        logits = model.score(
            user_emb=emb_param.unsqueeze(0),     # (1, D)
            user_bias=bias_param.unsqueeze(0),   # (1,)
            task_features=feat_tensor,           # (1, F)
        )
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()
        logger.debug("incremental step=%d loss=%.4f", step, float(loss.item()))

    new_emb = emb_param.detach().cpu().numpy().astype(np.float32)
    new_bias = float(bias_param.detach().cpu().item())
    return new_emb, new_bias


# --- Disk-backed convenience wrapper ---------------------------------------


def update_user_state_from_disk(
    cf_model_path: Path,
    user_emb: np.ndarray,
    user_bias: float,
    task_features: np.ndarray,
    completed: float,
    cfg: IncrementalUpdateConfig = IncrementalUpdateConfig(),
) -> Tuple[np.ndarray, float]:
    """Load the CF model from disk, then delegate to ``update_user_state``.

    This loads the model on every call, which is fine for ad-hoc use but
    wasteful in a hot path. Production callers should keep a singleton
    model and use ``update_user_state`` directly (see ``inference_api``).
    """
    cf_model_path = Path(cf_model_path)
    if not cf_model_path.exists():
        raise FileNotFoundError(f"CF model artifact not found: {cf_model_path}")

    payload = torch.load(cf_model_path, map_location="cpu")
    cfg_meta = payload["config"]
    model = CollaborativeFilteringModel(
        num_users=cfg_meta["num_users"],
        task_feature_dim=cfg_meta["task_feature_dim"],
        user_emb_dim=cfg_meta["user_emb_dim"],
        task_hidden_dim=cfg_meta.get("task_hidden_dim", 16),
    )
    model.load_state_dict(payload["state_dict"])
    return update_user_state(
        model=model,
        user_emb=user_emb,
        user_bias=user_bias,
        task_features=task_features,
        completed=completed,
        cfg=cfg,
    )
