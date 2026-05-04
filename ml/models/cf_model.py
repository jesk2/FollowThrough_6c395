"""Collaborative-filtering model for task-completion prediction.

The model factorizes the per-(user, task) score as

    ŷ = σ(u · t + b_u + b_t)

where:
    u  ∈ R^D   learnable per-user embedding (D = 8 by default)
    t  ∈ R^D   produced by a shallow MLP over task features
    b_u        per-user scalar bias
    b_t        scalar emitted by a head on top of the task MLP

The forward pass returns **logits** so callers can use
``torch.nn.BCEWithLogitsLoss`` (numerically stabler than sigmoid + BCE).
Apply a sigmoid externally to get a probability.

Two scoring entry points are exposed:
    - ``forward(user_ids, task_features)`` — used during pretraining,
      sources user vectors from the model's own ``nn.Embedding`` table.
    - ``score(user_emb, user_bias, task_features)`` — used at inference
      and during incremental updates, accepts user vectors from outside
      the model (so a dynamically resizing user store can drive scoring).
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


DEFAULT_USER_EMB_DIM: int = 8
DEFAULT_TASK_HIDDEN_DIM: int = 16


class CollaborativeFilteringModel(nn.Module):
    """Two-tower CF model with a task-feature encoder and a user-id table."""

    def __init__(
        self,
        num_users: int,
        task_feature_dim: int,
        user_emb_dim: int = DEFAULT_USER_EMB_DIM,
        task_hidden_dim: int = DEFAULT_TASK_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if num_users <= 0:
            raise ValueError(f"num_users must be positive: {num_users}")
        if task_feature_dim <= 0:
            raise ValueError(f"task_feature_dim must be positive: {task_feature_dim}")
        if user_emb_dim <= 0:
            raise ValueError(f"user_emb_dim must be positive: {user_emb_dim}")

        self.num_users: int = num_users
        self.task_feature_dim: int = task_feature_dim
        self.user_emb_dim: int = user_emb_dim
        self.task_hidden_dim: int = task_hidden_dim

        # User-side parameters.
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)

        # Task-side encoder: shared trunk + two heads (embedding + bias).
        self.task_trunk = nn.Sequential(
            nn.Linear(task_feature_dim, task_hidden_dim),
            nn.ReLU(),
        )
        self.task_emb_head = nn.Linear(task_hidden_dim, user_emb_dim)
        self.task_bias_head = nn.Linear(task_hidden_dim, 1)

        self._init_parameters()

    # --- construction helpers ---------------------------------------------

    def _init_parameters(self) -> None:
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        for layer in (self.task_emb_head, self.task_bias_head):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # The trunk's Linear is left at PyTorch default init (Kaiming-like)
        # which is appropriate for the ReLU activation that follows it.

    # --- core ops ----------------------------------------------------------

    def encode_task(
        self, task_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map raw task features → ``(task_embedding, task_bias)``."""
        if task_features.shape[-1] != self.task_feature_dim:
            raise ValueError(
                f"task_features has last-dim {task_features.shape[-1]}, "
                f"expected {self.task_feature_dim}"
            )
        h = self.task_trunk(task_features)
        t_emb = self.task_emb_head(h)
        t_bias = self.task_bias_head(h).squeeze(-1)
        return t_emb, t_bias

    def score(
        self,
        user_emb: torch.Tensor,        # (B, D)
        user_bias: torch.Tensor,       # (B,)
        task_features: torch.Tensor,   # (B, F)
    ) -> torch.Tensor:
        """Compute logits given externally supplied user vectors.

        Used by the incremental-update path so the user state can live in a
        dynamically-resizable store outside the model.
        """
        if user_emb.shape[-1] != self.user_emb_dim:
            raise ValueError(
                f"user_emb has last-dim {user_emb.shape[-1]}, "
                f"expected {self.user_emb_dim}"
            )
        if user_bias.shape != user_emb.shape[:-1]:
            raise ValueError(
                f"user_bias shape {tuple(user_bias.shape)} must match "
                f"user_emb leading dims {tuple(user_emb.shape[:-1])}"
            )

        t_emb, t_bias = self.encode_task(task_features)
        return (user_emb * t_emb).sum(dim=-1) + user_bias + t_bias

    def forward(
        self, user_ids: torch.Tensor, task_features: torch.Tensor
    ) -> torch.Tensor:
        """Return logits for a batch of (user_id, task_features) pairs."""
        u_emb = self.user_emb(user_ids)
        u_bias = self.user_bias(user_ids).squeeze(-1)
        return self.score(u_emb, u_bias, task_features)

    # --- mutation helpers used by incremental-update path -----------------

    def freeze_task_tower(self) -> None:
        """Disable gradients on the task encoder and task bias head.

        The user-side parameters are left untouched. Used as a safety
        guard before incremental updates so that — even if a caller
        accidentally hands the model's own ``user_emb`` table to the
        optimizer — the task tower cannot drift.
        """
        for module in (self.task_trunk, self.task_emb_head, self.task_bias_head):
            for p in module.parameters():
                p.requires_grad_(False)

    # --- read-only accessors ----------------------------------------------

    def get_user_embedding_matrix(self) -> torch.Tensor:
        """Return a detached copy of the (num_users, D) embedding matrix."""
        return self.user_emb.weight.detach().clone()

    def get_user_bias_vector(self) -> torch.Tensor:
        """Return a detached copy of the (num_users,) bias vector."""
        return self.user_bias.weight.detach().clone().squeeze(-1)
