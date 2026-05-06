"""Offline pretraining entry point.

Pipeline:
    1. Generate synthetic dataset (users, tasks, β, labels).
    2. Train the CF model with ``BCEWithLogitsLoss`` over an 80/20 split,
       AdamW + cosine LR schedule + per-group weight decay (heavier on
       the user-side parameters because they are easier to overfit with
       only ~50 observations per user).
    3. Extract the frozen user embeddings.
    4. Fit the Ridge probe on ``(embedding, β)``.
    5. Persist all artifacts: model state-dict, probe, feature stats,
       a snapshot of the user-state table the inference layer grows as
       new users register, and the train/validation loss curves as both
       a JSON history and a PNG plot.

Run as a module:
    python -m ml.training.pretrain
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.data.synthetic_generator import (
    LABEL_BASE_PROB,
    LABEL_DIFFICULTY_SCALE,
    TASK_FEATURE_DIM,
    SyntheticDataset,
    generate_dataset,
)
from ml.models.cf_model import CollaborativeFilteringModel
from ml.models.linear_probe import BetaProbe

logger = logging.getLogger(__name__)


# --- Artifact paths (single source of truth, imported by inference too) ----

DEFAULT_ARTIFACT_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"
CF_MODEL_FILE: str = "cf_model.pt"
PROBE_FILE: str = "ridge_probe.joblib"
FEATURE_STATS_FILE: str = "feature_stats.json"
USER_STATE_FILE: str = "user_state.pt"
TRAIN_HISTORY_FILE: str = "train_history.json"
LOSS_CURVE_FILE: str = "loss_curve.png"


# --- Training configuration -------------------------------------------------


@dataclass
class TrainConfig:
    """Hyperparameters for a single pretraining run.

    The two weight-decay knobs let us regularize the user-side parameters
    (which see only ``tasks_per_user`` observations each) more strongly
    than the task tower (which sees the full dataset). Without this,
    the user embeddings overfit within the first 2–3 epochs and the
    validation loss starts climbing.
    """

    num_users: int = 3000
    tasks_per_user: int = 150
    task_hidden_dim: int = 32         # wider task tower (was 16); plenty of data to support it
    val_fraction: float = 0.2
    batch_size: int = 4096            # H100-sized
    epochs: int = 60                  # upper bound; early stopping usually trips first
    lr: float = 3e-3
    user_weight_decay: float = 1e-3   # AdamW WD on user_emb / user_bias
    task_weight_decay: float = 1e-5   # AdamW WD on task tower + heads
    early_stop_patience: int = 6      # stop after N epochs without val-loss improvement
    early_stop_min_delta: float = 5e-5
    seed: int = 42
    ridge_alpha: float = 1.0


@dataclass
class TrainHistory:
    """Per-epoch training metrics, used for diagnostics and plotting."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


# --- Internals --------------------------------------------------------------


def _split_indices(
    n: int, val_frac: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Random 80/20 split of row indices."""
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1): {val_frac}")
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    return perm[n_val:], perm[:n_val]


def _make_loader(
    dataset: SyntheticDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    user_ids = torch.from_numpy(dataset.user_ids[indices]).long()
    features = torch.from_numpy(dataset.task_features[indices]).float()
    labels = torch.from_numpy(dataset.labels[indices]).float()
    return DataLoader(
        TensorDataset(user_ids, features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def _build_optimizer(
    model: CollaborativeFilteringModel, cfg: TrainConfig
) -> torch.optim.Optimizer:
    """AdamW with two parameter groups so we can regularize differently.

    The user embeddings and user-bias table are the most over-parameterized
    pieces (``num_users × user_emb_dim`` parameters each seen by only
    ``tasks_per_user`` observations), so they get a much heavier L2 prior
    than the task tower.
    """
    user_param_ids = {
        id(p)
        for p in (model.user_emb.weight, model.user_bias.weight)
    }
    user_params, task_params = [], []
    for p in model.parameters():
        (user_params if id(p) in user_param_ids else task_params).append(p)

    return torch.optim.AdamW(
        [
            {"params": user_params, "weight_decay": cfg.user_weight_decay},
            {"params": task_params, "weight_decay": cfg.task_weight_decay},
        ],
        lr=cfg.lr,
    )


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _evaluate_loss(
    model: CollaborativeFilteringModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute the mean BCE loss over a loader. No grad, eval mode."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for user_ids, features, labels in loader:
            user_ids = user_ids.to(device)
            features = features.to(device)
            labels = labels.to(device)
            logits = model(user_ids, features)
            loss = criterion(logits, labels)
            total += float(loss.item()) * labels.size(0)
            count += labels.size(0)
    return total / max(count, 1)


def _train_cf(
    model: CollaborativeFilteringModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> TrainHistory:
    """Train the CF model in-place. Returns the per-epoch history.

    Schedule: cosine-annealed LR over ``cfg.epochs``. The best
    validation checkpoint is captured and restored at the end of
    training (no early stopping — cosine annealing benefits from
    completing its full schedule).
    """
    optimizer = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
    criterion = nn.BCEWithLogitsLoss()

    history = TrainHistory()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_since_best = 0

    # Epoch-0 baseline: how the random-init model scores before any update.
    # This is the "ground floor" of the training curve and makes the actual
    # drop from training visible (otherwise the curve looks like it only
    # moves a tiny amount because epoch 1 is already most of the way down).
    init_train_loss = _evaluate_loss(model, train_loader, criterion, device)
    init_val_loss = _evaluate_loss(model, val_loader, criterion, device)
    history.train_loss.append(init_train_loss)
    history.val_loss.append(init_val_loss)
    history.lr.append(float(cfg.lr))
    logger.info(
        "epoch=00  (random init)  train_loss=%.4f  val_loss=%.4f",
        init_train_loss,
        init_val_loss,
    )

    for epoch in range(1, cfg.epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for user_ids, features, labels in train_loader:
            user_ids = user_ids.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(user_ids, features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * labels.size(0)
            train_count += labels.size(0)

        train_loss = train_loss_sum / max(train_count, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for user_ids, features, labels in val_loader:
                user_ids = user_ids.to(device)
                features = features.to(device)
                labels = labels.to(device)
                logits = model(user_ids, features)
                loss = criterion(logits, labels)
                val_loss_sum += float(loss.item()) * labels.size(0)
                val_count += labels.size(0)
        val_loss = val_loss_sum / max(val_count, 1)

        current_lr = float(scheduler.get_last_lr()[0])
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr.append(current_lr)

        logger.info(
            "epoch=%02d  lr=%.5f  train_loss=%.4f  val_loss=%.4f",
            epoch,
            current_lr,
            train_loss,
            val_loss,
        )

        if val_loss < history.best_val_loss - cfg.early_stop_min_delta:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (best val=%.4f at epoch %d).",
                    epoch,
                    history.best_val_loss,
                    history.best_epoch,
                )
                break

        scheduler.step()

    logger.info(
        "Training done. Best val_loss=%.4f at epoch %d.",
        history.best_val_loss,
        history.best_epoch,
    )
    model.load_state_dict(best_state)
    return history


def _bayes_optimal_bce(dataset: SyntheticDataset) -> float:
    """Mean per-task binary entropy under the generator's true P(y=1).

    This is the information-theoretic floor on validation BCE: no model
    can do better in expectation, because the labels themselves are
    Bernoulli draws from these probabilities.
    """
    betas = dataset.betas[dataset.user_ids]
    difficulty = dataset.task_features[:, 0]
    delay = dataset.raw_delays.astype(np.float32)
    p = LABEL_BASE_PROB * (1.0 - LABEL_DIFFICULTY_SCALE * difficulty) * np.power(
        betas, delay
    )
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(h.mean())


def _save_loss_curves(
    history: TrainHistory,
    png_path: Path,
    bayes_floor: Optional[float] = None,
) -> None:
    # Lazy-import matplotlib here so that ``import ml.training.pretrain``
    # is cheap on the inference path (the backend container imports a
    # few constants from this module — it must not pay matplotlib's
    # startup cost just to do that).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    """Render the train/val loss curves (and LR overlay) to a PNG.

    If ``bayes_floor`` is provided (the entropy of P(y|x) under the data
    generator), it is drawn as a horizontal reference so it's obvious how
    close the model is to the information-theoretic minimum.
    """
    # history index 0 is the random-init baseline; subsequent indices are
    # post-epoch-N evaluations.
    epochs = list(range(0, len(history.train_loss)))

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(epochs, history.train_loss, label="Train", linewidth=2)
    ax_loss.plot(epochs, history.val_loss, label="Validation", linewidth=2)
    if bayes_floor is not None:
        ax_loss.axhline(
            bayes_floor,
            color="tab:red",
            linestyle=":",
            linewidth=1.5,
            label=f"Bayes-optimal floor ({bayes_floor:.4f})",
        )
    if history.best_epoch > 0:
        ax_loss.axvline(
            history.best_epoch,
            color="grey",
            linestyle="--",
            alpha=0.6,
            label=f"Best epoch ({history.best_epoch})",
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE-with-logits loss")
    ax_loss.set_title("CF model training curves")
    ax_loss.grid(True, alpha=0.3)

    # Secondary axis for LR.
    ax_lr = ax_loss.twinx()
    ax_lr.plot(
        epochs,
        history.lr,
        label="LR",
        color="tab:green",
        alpha=0.4,
        linewidth=1,
    )
    ax_lr.set_ylabel("Learning rate", color="tab:green")
    ax_lr.tick_params(axis="y", labelcolor="tab:green")

    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_r, labels_r = ax_lr.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved loss curve to %s", png_path)


# --- Public entry point -----------------------------------------------------


def run_pretraining(
    cfg: TrainConfig = TrainConfig(),
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    device: Optional[torch.device] = None,
) -> Path:
    """Run the full pretraining pipeline. Returns the artifact directory."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Pretraining on device=%s", device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 1. Synthetic data
    dataset = generate_dataset(
        num_users=cfg.num_users,
        tasks_per_user=cfg.tasks_per_user,
        seed=cfg.seed,
    )

    # 2. CF model
    model = CollaborativeFilteringModel(
        num_users=cfg.num_users,
        task_feature_dim=TASK_FEATURE_DIM,
        task_hidden_dim=cfg.task_hidden_dim,
    ).to(device)

    print(f"Collaborative filtering model number of parameters: {_count_parameters(model)}")

    # 3. Train/val split + training
    rng = np.random.default_rng(cfg.seed)
    train_idx, val_idx = _split_indices(
        dataset.user_ids.shape[0], cfg.val_fraction, rng
    )
    train_loader = _make_loader(dataset, train_idx, cfg.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_idx, cfg.batch_size, shuffle=False)
    history = _train_cf(model, train_loader, val_loader, cfg, device)

    # Compute the Bayes-optimal BCE floor analytically from the synthetic
    # generator's P(y=1|features). Useful as a visual ceiling on training.
    bayes_floor = _bayes_optimal_bce(dataset)
    logger.info(
        "Bayes-optimal BCE floor: %.4f (best val=%.4f, headroom=%.4f).",
        bayes_floor,
        history.best_val_loss,
        history.best_val_loss - bayes_floor,
    )

    # Persist training diagnostics so the eval notebook can show them.
    history_path = artifact_dir / TRAIN_HISTORY_FILE
    history_payload = asdict(history)
    history_payload["bayes_optimal_bce"] = bayes_floor
    with open(history_path, "w") as f:
        json.dump(history_payload, f, indent=2)
    logger.info("Saved training history to %s", history_path)
    _save_loss_curves(history, artifact_dir / LOSS_CURVE_FILE, bayes_floor=bayes_floor)

    # 4. Fit Ridge probe on frozen embeddings
    user_emb = model.get_user_embedding_matrix().cpu().numpy()
    user_bias = model.get_user_bias_vector().cpu().numpy()
    probe = BetaProbe(alpha=cfg.ridge_alpha).fit(user_emb, dataset.betas)

    # 5. Persist artifacts
    cf_path = artifact_dir / CF_MODEL_FILE
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "num_users": cfg.num_users,
                "task_feature_dim": TASK_FEATURE_DIM,
                "user_emb_dim": model.user_emb_dim,
                "task_hidden_dim": model.task_hidden_dim,
            },
        },
        cf_path,
    )
    logger.info("Saved CF model to %s", cf_path)

    probe.save(artifact_dir / PROBE_FILE)

    stats_path = artifact_dir / FEATURE_STATS_FILE
    with open(stats_path, "w") as f:
        json.dump(dataset.feature_stats.to_dict(), f, indent=2)
    logger.info("Saved feature stats to %s", stats_path)

    # Snapshot the user-state table the inference layer will grow as new
    # users register. We also persist the population mean so newly created
    # users can be initialized to a sensible prior.
    user_state_path = artifact_dir / USER_STATE_FILE
    torch.save(
        {
            "embeddings": torch.from_numpy(user_emb).float(),
            "biases": torch.from_numpy(user_bias).float(),
            "population_mean_emb": torch.from_numpy(user_emb.mean(axis=0)).float(),
            "population_mean_bias": float(user_bias.mean()),
            "next_user_id": int(cfg.num_users),
        },
        user_state_path,
    )
    logger.info("Saved user-state snapshot to %s", user_state_path)

    return artifact_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_pretraining()
