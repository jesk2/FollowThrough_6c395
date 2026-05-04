"""Build evaluation_plots.ipynb programmatically.

Run from the repo root:

    PYTHONPATH=. .venv/bin/python ml/notebooks/_build_notebook.py

This script is here purely to keep the notebook source under version
control as readable Python rather than as raw JSON. After running it,
``evaluation_plots.ipynb`` is the canonical artifact — feel free to
delete this builder if you prefer.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent / "evaluation_plots.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip("\n"))


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------

cells: list[nbf.NotebookNode] = []

cells.append(md(r'''
# FollowThrough — ML Evaluation Report

This notebook generates the publication-quality plots used in the project
writeup. All figures are derived from the **pretrained CF model**, the
**Ridge-regression β-probe**, and the **synthetic dataset** that produced
them. Run `python -m ml.training.pretrain` from the repo root first so
the artifacts in `ml/artifacts/` are populated.

**The five plots:**

| # | Plot | Question it answers |
|---|---|---|
| 1 | Behavioral manifold (PCA) | Did the embedding space recover the latent β dimension? |
| 2 | β-proxy calibration | Does the linear probe yield interpretable β estimates? |
| 3 | Reliability diagram | When the model says 70%, does it happen 70% of the time? |
| 4 | Learning curve | Does the per-user state get sharper as more tasks are observed? |
| 5 | Feature-impact comparison | Do predictions move intuitively across user types and task properties? |

Each figure is also written to `ml/notebooks/figures/` as a standalone
PNG so the project writeup can embed them without re-running anything.
'''))

cells.append(md("## 0. Setup"))

cells.append(code(r'''
# Standard imports.
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Ensure the project root is on sys.path regardless of where the kernel
# was started (notebook dir, repo root, etc.). Walk up until we find the
# `ml/__init__.py` marker.
def _find_project_root() -> Path:
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        if (cand / "ml" / "__init__.py").exists() and cand.name != "ml":
            return cand
    raise RuntimeError(f"Could not locate project root from {here}")

PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import (
    brier_score_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

# FollowThrough imports.
from ml.data.synthetic_generator import (
    NUM_DEADLINE_BUCKETS,
    TASK_FEATURE_DIM,
    FeatureStats,
    encode_task_features,
    generate_dataset,
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
    TrainConfig,
    USER_STATE_FILE,
)

# Plotting style — clean, publication-ready defaults.
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.titleweight": "semibold",
})
'''))

cells.append(code(r'''
# Paths. Override `ARTIFACT_DIR` if you trained into a non-default location.
ARTIFACT_DIR = Path(DEFAULT_ARTIFACT_DIR)
FIGURE_DIR = PROJECT_ROOT / "ml" / "notebooks" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Pretraining config — must match what produced the artifacts in ARTIFACT_DIR
# so the regenerated dataset (β values, labels) is bit-identical.
PRETRAIN_CFG = TrainConfig()

print("Artifact dir:", ARTIFACT_DIR)
print("Figure dir:  ", FIGURE_DIR)
print("Pretrain cfg:", PRETRAIN_CFG)
'''))

cells.append(md("### 0.1 Load artifacts"))

cells.append(code(r'''
def load_cf_model(artifact_dir: Path) -> CollaborativeFilteringModel:
    """Load the pretrained CF model into eval mode on CPU."""
    payload = torch.load(artifact_dir / CF_MODEL_FILE, map_location="cpu")
    cfg = payload["config"]
    model = CollaborativeFilteringModel(
        num_users=cfg["num_users"],
        task_feature_dim=cfg["task_feature_dim"],
        user_emb_dim=cfg["user_emb_dim"],
        task_hidden_dim=cfg.get("task_hidden_dim", 16),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_feature_stats(artifact_dir: Path) -> FeatureStats:
    with open(artifact_dir / FEATURE_STATS_FILE) as f:
        return FeatureStats.from_dict(json.load(f))


def load_user_state(artifact_dir: Path) -> dict:
    return torch.load(artifact_dir / USER_STATE_FILE, map_location="cpu")


model = load_cf_model(ARTIFACT_DIR)
probe = BetaProbe.load(ARTIFACT_DIR / PROBE_FILE)
feature_stats = load_feature_stats(ARTIFACT_DIR)
user_state = load_user_state(ARTIFACT_DIR)

print(f"CF model: {sum(p.numel() for p in model.parameters()):,} params, "
      f"user_emb_dim={model.user_emb_dim}, task_feature_dim={model.task_feature_dim}")
print(f"Ridge probe alpha={probe.alpha}, fitted={probe.is_fitted}")
print(f"User store: {tuple(user_state['embeddings'].shape)} embeddings")
'''))

cells.append(md("### 0.2 Re-create the synthetic dataset"))

cells.append(md(r'''
The dataset itself is deterministic given the seed in `TrainConfig`, so
re-generating from the same seed gives bit-identical user IDs, β values,
features, and labels. We use the same dataset throughout the report so
ground-truth β is available for every plot.
'''))

cells.append(code(r'''
dataset = generate_dataset(
    num_users=PRETRAIN_CFG.num_users,
    tasks_per_user=PRETRAIN_CFG.tasks_per_user,
    seed=PRETRAIN_CFG.seed,
)

# Convenient handles.
user_ids_all = dataset.user_ids
task_features_all = dataset.task_features
labels_all = dataset.labels
betas_true = dataset.betas
raw_delays_all = dataset.raw_delays

print(f"Tasks: {task_features_all.shape[0]:,}, "
      f"users: {betas_true.shape[0]:,}, "
      f"positive rate: {labels_all.mean():.3f}")
'''))

cells.append(md("### 0.3 Helper functions"))

cells.append(code(r'''
@torch.no_grad()
def predict_probs(
    model: CollaborativeFilteringModel,
    user_ids: np.ndarray,
    task_features: np.ndarray,
) -> np.ndarray:
    """Vectorized P(complete) over a batch of (user_id, task) pairs."""
    logits = model(
        torch.from_numpy(user_ids).long(),
        torch.from_numpy(task_features).float(),
    )
    return torch.sigmoid(logits).numpy()


@torch.no_grad()
def predict_probs_with_emb(
    model: CollaborativeFilteringModel,
    user_emb: np.ndarray,
    user_bias: float,
    task_features: np.ndarray,
) -> np.ndarray:
    """P(complete) when the user vector lives outside the model table.

    Used for evaluating an incrementally-updated user against a held-out
    set of tasks.
    """
    feats = task_features
    if feats.ndim == 1:
        feats = feats[None]
    n = feats.shape[0]
    emb_t = torch.from_numpy(user_emb.astype(np.float32)).unsqueeze(0).expand(n, -1)
    bias_t = torch.tensor([float(user_bias)] * n, dtype=torch.float32)
    feat_t = torch.from_numpy(feats.astype(np.float32))
    logits = model.score(emb_t, bias_t, feat_t)
    return torch.sigmoid(logits).numpy()


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Persist a figure to FIGURE_DIR with a consistent stem."""
    out = FIGURE_DIR / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return out
'''))

# ----- Plot 1: PCA manifold ------------------------------------------------

cells.append(md(r'''
## 1. The Behavioral Manifold

**Question:** does the model's user-embedding layer organize people along
their latent present-bias dimension, or is the geometry random?

**Method:** project the 8-D user embeddings to 2-D with PCA, color each
user by their true synthetic β. If the model picked up on β, the colors
should form a smooth gradient along some direction in the PC1–PC2 plane.
'''))

cells.append(code(r'''
user_emb = user_state["embeddings"].numpy()  # (N_users, D)
assert user_emb.shape[0] == betas_true.shape[0]

pca = PCA(n_components=2, random_state=PRETRAIN_CFG.seed)
emb_2d = pca.fit_transform(user_emb)
explained = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}, "
      f"sum={explained.sum():.3f}")
'''))

cells.append(code(r'''
def plot_behavioral_manifold(
    emb_2d: np.ndarray,
    betas: np.ndarray,
    explained: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=betas,
        cmap="viridis",
        s=22,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"Present Bias ($\beta$)")

    ax.set_xlabel(f"Principal Component 1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"Principal Component 2 ({explained[1]*100:.1f}% var)")
    ax.set_title("CF user-embedding manifold colored by true β")
    return fig


fig = plot_behavioral_manifold(emb_2d, betas_true, explained)
save_figure(fig, "01_behavioral_manifold")
plt.show()
'''))

# ----- Plot 2: probe calibration -------------------------------------------

cells.append(md(r'''
## 2. β-Proxy Calibration

**Question:** does running an embedding through the Ridge probe recover
its owner's true β?

**Method:** scatter true β (x) against probe-predicted β (y) for every
user. A perfect probe would put every point on the y = x diagonal.
Annotate with R² and MSE.
'''))

cells.append(code(r'''
beta_pred = probe.predict(user_emb)  # (N_users,)
r2 = r2_score(betas_true, beta_pred)
mse = mean_squared_error(betas_true, beta_pred)
print(f"Probe R² = {r2:.4f}, MSE = {mse:.5f}")


def plot_beta_calibration(
    betas_true: np.ndarray,
    betas_pred: np.ndarray,
    r2: float,
    mse: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6.5))

    # Scatter + linear fit through the cloud.
    sns.regplot(
        x=betas_true,
        y=betas_pred,
        ax=ax,
        scatter_kws={"s": 22, "alpha": 0.55, "edgecolors": "white", "linewidths": 0.3},
        line_kws={"color": "C1", "linewidth": 2, "label": "OLS fit"},
    )
    # Perfect-calibration diagonal.
    lo = min(betas_true.min(), betas_pred.min()) - 0.02
    hi = max(betas_true.max(), betas_pred.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5,
            label="Ideal (y = x)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"True $\beta$")
    ax.set_ylabel(r"Predicted $\beta$ (probe)")
    ax.set_title(
        f"Linear-probe calibration  ·  $R^2$ = {r2:.3f}  ·  MSE = {mse:.4f}"
    )
    ax.legend(loc="upper left")
    return fig


fig = plot_beta_calibration(betas_true, beta_pred, r2, mse)
save_figure(fig, "02_beta_calibration")
plt.show()
'''))

# ----- Plot 3: reliability diagram -----------------------------------------

cells.append(md(r'''
## 3. Reliability Diagram (Probability Calibration)

**Question:** when the model emits a 0.7, does the user complete the
task ~70% of the time?

**Method:** bin the model's predicted probabilities and plot the mean
predicted vs. the empirical positive rate within each bin. Overlay the
y = x reference. Brier score in the title.
'''))

cells.append(code(r'''
preds_all = predict_probs(model, user_ids_all, task_features_all)
brier = brier_score_loss(labels_all, preds_all)
fop, mpv = calibration_curve(labels_all, preds_all, n_bins=12, strategy="quantile")
print(f"Brier score = {brier:.4f}  (lower is better; perfect = 0, naive ≈ {labels_all.mean()*(1-labels_all.mean()):.3f})")
'''))

cells.append(code(r'''
def plot_reliability_diagram(
    fop: np.ndarray,
    mpv: np.ndarray,
    preds: np.ndarray,
    brier: float,
) -> plt.Figure:
    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1,
        figsize=(7, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    ax_main.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5,
                 label="Perfectly calibrated")
    ax_main.plot(mpv, fop, marker="o", linewidth=2, color="C0",
                 label="Model")
    ax_main.set_ylabel("Empirical fraction of positives")
    ax_main.set_title(f"Reliability diagram  ·  Brier = {brier:.4f}")
    ax_main.legend(loc="upper left")
    ax_main.set_ylim(-0.02, 1.02)

    ax_hist.hist(preds, bins=30, color="C0", alpha=0.7)
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(0, 1)
    return fig


fig = plot_reliability_diagram(fop, mpv, preds_all, brier)
save_figure(fig, "03_reliability_diagram")
plt.show()
'''))

# ----- Plot 4: learning curve ----------------------------------------------

cells.append(md(r'''
## 4. Learning Curve — Personalization through Incremental Updates

**Question:** does the per-user state get sharper as the user logs more
tasks? This is the central claim of the system: each completion event
should refine the user's embedding and improve next-task predictions.

**Method:** for a sample of pretrained users we

1. reset the user's embedding to the population mean (a fresh user state),
2. simulate observing the first *k* tasks of that user's history and
   apply `incremental_update_api` after each one,
3. score the held-out last 30 tasks and compute AUC-ROC.

We sweep *k* ∈ {0, 5, 10, 20, 40, 80, 120} and report two metrics
side-by-side:

1. **Held-out AUC-ROC** (per spec). Bracketed by:
   - **Population-mean baseline** — AUC for an unpersonalized (cold-start) user
   - **Pretrained per-user ceiling** — AUC for the embedding learned from
     a user's full history during pretraining
   With a strong task tower the predictive headroom is small in absolute
   terms (the global task signal already explains most of the variance),
   so AUC is a relatively insensitive demonstration.

2. **|β̂ − β_true|** — the deviation between the Ridge-probe's β estimate
   and the user's true synthetic β. This is much more sensitive to
   personalization: it directly measures whether the embedding has been
   pulled toward this user's individual location in latent space.
'''))

cells.append(code(r'''
def simulate_learning_curve(
    *,
    model: CollaborativeFilteringModel,
    probe: BetaProbe,
    betas_true: np.ndarray,
    population_mean_emb: np.ndarray,
    population_mean_bias: float,
    pretrained_embeddings: np.ndarray,
    pretrained_biases: np.ndarray,
    user_ids_all: np.ndarray,
    task_features_all: np.ndarray,
    labels_all: np.ndarray,
    selected_users: np.ndarray,
    history_size: int = 120,
    test_size: int = 30,
    checkpoints: tuple = (0, 5, 10, 20, 40, 80, 120),
    rng: np.random.Generator | None = None,
) -> dict:
    """Run the personalization simulation and return three aligned arrays.

    Returns a dict with:
        ``aucs``         : (n_users, n_checkpoints) AUCs after k updates
        ``pop_aucs``     : (n_users,) AUCs using the bare population mean
        ``ceiling_aucs`` : (n_users,) AUCs using the pretrained embedding
        ``skipped``      : list of user ids dropped (degenerate test labels)

    Users whose held-out test set is degenerate (all-positive or
    all-negative labels) are skipped — AUC is undefined there.
    """
    rng = rng or np.random.default_rng(0)
    update_cfg = IncrementalUpdateConfig()

    n = selected_users.shape[0]
    aucs = np.full((n, len(checkpoints)), np.nan)
    beta_errs = np.full((n, len(checkpoints)), np.nan)
    pop_aucs = np.full(n, np.nan)
    ceiling_aucs = np.full(n, np.nan)
    skipped: list = []

    for u_idx, user_id in enumerate(selected_users):
        mask = user_ids_all == user_id
        u_features = task_features_all[mask]
        u_labels = labels_all[mask]

        # Shuffle so the "history" isn't biased by per-user task ordering.
        order = rng.permutation(u_features.shape[0])
        u_features = u_features[order]
        u_labels = u_labels[order]

        if u_features.shape[0] < history_size + test_size:
            skipped.append(int(user_id))
            continue

        history_feat = u_features[:history_size]
        history_lab = u_labels[:history_size]
        test_feat = u_features[history_size : history_size + test_size]
        test_lab = u_labels[history_size : history_size + test_size]

        if len(np.unique(test_lab)) < 2:
            skipped.append(int(user_id))
            continue

        # Bounds: the population-mean baseline (no personalization) and
        # the pretrained embedding (full-history personalization).
        pop_probs = predict_probs_with_emb(
            model, population_mean_emb, population_mean_bias, test_feat
        )
        pop_aucs[u_idx] = roc_auc_score(test_lab, pop_probs)

        ceil_probs = predict_probs_with_emb(
            model,
            pretrained_embeddings[user_id],
            float(pretrained_biases[user_id]),
            test_feat,
        )
        ceiling_aucs[u_idx] = roc_auc_score(test_lab, ceil_probs)

        # Walk the checkpoints in order, incrementally adding history.
        emb = population_mean_emb.copy()
        bias = population_mean_bias
        prev_cp = 0
        for cp_idx, cp in enumerate(checkpoints):
            for i in range(prev_cp, cp):
                emb, bias = update_user_state(
                    model=model,
                    user_emb=emb,
                    user_bias=bias,
                    task_features=history_feat[i],
                    completed=float(history_lab[i]),
                    cfg=update_cfg,
                )
            prev_cp = cp

            test_probs = predict_probs_with_emb(model, emb, bias, test_feat)
            aucs[u_idx, cp_idx] = roc_auc_score(test_lab, test_probs)
            beta_pred = float(probe.predict(emb)[0])
            beta_errs[u_idx, cp_idx] = abs(beta_pred - float(betas_true[user_id]))

    return {
        "aucs": aucs,
        "beta_errs": beta_errs,
        "pop_aucs": pop_aucs,
        "ceiling_aucs": ceiling_aucs,
        "skipped": skipped,
    }


population_mean_emb = user_state["population_mean_emb"].numpy()
population_mean_bias = float(user_state["population_mean_bias"])

CHECKPOINTS = (0, 5, 10, 20, 40, 80, 120)
HISTORY_SIZE = 120
TEST_SIZE = 30

rng = np.random.default_rng(7)
candidate_users = np.arange(PRETRAIN_CFG.num_users)
selected_users = rng.choice(candidate_users, size=80, replace=False)

pretrained_embeddings = user_state["embeddings"].numpy()
pretrained_biases = user_state["biases"].numpy()

sim = simulate_learning_curve(
    model=model,
    probe=probe,
    betas_true=betas_true,
    population_mean_emb=population_mean_emb,
    population_mean_bias=population_mean_bias,
    pretrained_embeddings=pretrained_embeddings,
    pretrained_biases=pretrained_biases,
    user_ids_all=user_ids_all,
    task_features_all=task_features_all,
    labels_all=labels_all,
    selected_users=selected_users,
    history_size=HISTORY_SIZE,
    test_size=TEST_SIZE,
    checkpoints=CHECKPOINTS,
    rng=rng,
)

aucs_matrix = sim["aucs"]
beta_errs = sim["beta_errs"]
pop_aucs = sim["pop_aucs"]
ceiling_aucs = sim["ceiling_aucs"]

valid = ~np.isnan(aucs_matrix[:, 0])
print(f"Evaluated {valid.sum()} users; "
      f"skipped {len(sim['skipped'])} (degenerate test labels).")
print(f"Population-mean AUC = {np.nanmean(pop_aucs):.4f}")
print(f"Pretrained-emb AUC  = {np.nanmean(ceiling_aucs):.4f}")
print(f"After {CHECKPOINTS[-1]} updates: AUC = {np.nanmean(aucs_matrix[:, -1]):.4f}, "
      f"|β err| = {np.nanmean(beta_errs[:, -1]):.4f}")
print(f"At  0 updates (cold start): |β err| = {np.nanmean(beta_errs[:, 0]):.4f}")
'''))

cells.append(code(r'''
def _mean_and_ci(matrix: np.ndarray):
    n = (~np.isnan(matrix)).sum(axis=0)
    mean = np.nanmean(matrix, axis=0)
    sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, 1.96 * sem


def plot_learning_curve(
    aucs: np.ndarray,
    beta_errs: np.ndarray,
    pop_aucs: np.ndarray,
    ceiling_aucs: np.ndarray,
    checkpoints: tuple,
) -> plt.Figure:
    fig, (ax_auc, ax_beta) = plt.subplots(
        1, 2, figsize=(13, 5.2), sharex=True
    )

    # --- Panel 1: held-out AUC with bracketed reference lines ---
    mean_auc, ci_auc = _mean_and_ci(aucs)
    pop_mean = float(np.nanmean(pop_aucs))
    ceil_mean = float(np.nanmean(ceiling_aucs))

    ax_auc.plot(checkpoints, mean_auc, marker="o", linewidth=2.4, color="C2",
                label="Incremental updates")
    ax_auc.fill_between(
        checkpoints, mean_auc - ci_auc, mean_auc + ci_auc,
        alpha=0.22, color="C2", label="95% CI",
    )
    ax_auc.axhline(pop_mean, color="C3", linestyle="--", linewidth=1.5,
                   label=f"Population baseline ({pop_mean:.3f})")
    ax_auc.axhline(ceil_mean, color="C0", linestyle="--", linewidth=1.5,
                   label=f"Pretrained ceiling ({ceil_mean:.3f})")
    ax_auc.set_xlabel("Number of historical check-ins observed")
    ax_auc.set_ylabel("Held-out AUC-ROC")
    ax_auc.set_title("Predictive accuracy")
    ax_auc.legend(loc="lower right", fontsize=9)
    ax_auc.set_xticks(checkpoints)

    # --- Panel 2: |β_predicted - β_true| via the Ridge probe ---
    mean_err, ci_err = _mean_and_ci(beta_errs)
    ax_beta.plot(checkpoints, mean_err, marker="o", linewidth=2.4, color="C4",
                 label="Incremental updates")
    ax_beta.fill_between(
        checkpoints, mean_err - ci_err, mean_err + ci_err,
        alpha=0.22, color="C4", label="95% CI",
    )
    ax_beta.set_xlabel("Number of historical check-ins observed")
    ax_beta.set_ylabel(r"$|\hat\beta - \beta_\text{true}|$")
    ax_beta.set_title("β-recovery error (probe applied to user emb)")
    ax_beta.legend(loc="upper right", fontsize=9)
    ax_beta.set_xticks(checkpoints)

    fig.suptitle(
        "Personalization signal vs. number of observed events",
        y=1.02, fontsize=13, fontweight="semibold",
    )
    fig.tight_layout()
    return fig


fig = plot_learning_curve(aucs_matrix, beta_errs, pop_aucs, ceiling_aucs, CHECKPOINTS)
save_figure(fig, "04_learning_curve")
plt.show()
'''))

# ----- Plot 5: feature impact ----------------------------------------------

cells.append(md(r'''
## 5. Feature-Impact Analysis

**Question:** do predictions move in the *right direction* when we
manipulate task features, and does the model treat high-β and low-β
users differently?

**Method:** pick a high-β profile (95th-percentile β) and a low-β
profile (5th-percentile β). Build a panel of "scenario tasks" that vary
one feature at a time (difficulty, deadline pressure, days-out) while
holding the others fixed. Compare predicted P(complete) for the two
user profiles in a grouped bar chart.
'''))

cells.append(code(r'''
def make_scenario_features(
    *,
    difficulty: float,
    category_index: int = 0,
    planned_duration_minutes: float = 30.0,
    days_until_planned_start: int = 1,
    deadline_pressure_index: int = 1,
    stats: FeatureStats = feature_stats,
) -> np.ndarray:
    return encode_task_features(
        difficulty=difficulty,
        category_index=category_index,
        planned_duration_minutes=planned_duration_minutes,
        days_until_planned_start=days_until_planned_start,
        deadline_pressure_index=deadline_pressure_index,
        stats=stats,
    )


# A small panel of scenarios. Each row varies a single dimension from a
# common baseline, so the bar chart isolates that dimension's effect.
SCENARIOS = [
    ("Easy · today",            dict(difficulty=0.2, days_until_planned_start=0, deadline_pressure_index=0)),
    ("Easy · this week (3d)",   dict(difficulty=0.2, days_until_planned_start=3, deadline_pressure_index=1)),
    ("Easy · later (10d)",      dict(difficulty=0.2, days_until_planned_start=10, deadline_pressure_index=2)),
    ("Hard · today",            dict(difficulty=0.8, days_until_planned_start=0, deadline_pressure_index=0)),
    ("Hard · this week (3d)",   dict(difficulty=0.8, days_until_planned_start=3, deadline_pressure_index=1)),
    ("Hard · later (10d)",      dict(difficulty=0.8, days_until_planned_start=10, deadline_pressure_index=2)),
]


def pick_user_by_beta_quantile(quantile: float) -> int:
    """Return the user id whose true β sits at the requested quantile."""
    target = float(np.quantile(betas_true, quantile))
    return int(np.argmin(np.abs(betas_true - target)))


low_user = pick_user_by_beta_quantile(0.05)
high_user = pick_user_by_beta_quantile(0.95)
print(f"Low-β user:  id={low_user}, β={betas_true[low_user]:.3f}")
print(f"High-β user: id={high_user}, β={betas_true[high_user]:.3f}")
'''))

cells.append(code(r'''
def predict_for_user_scenarios(
    user_id: int,
    scenarios: list,
) -> np.ndarray:
    feats = np.stack([make_scenario_features(**kw) for _, kw in scenarios])
    user_ids = np.full(len(scenarios), user_id, dtype=np.int64)
    return predict_probs(model, user_ids, feats)


probs_low = predict_for_user_scenarios(low_user, SCENARIOS)
probs_high = predict_for_user_scenarios(high_user, SCENARIOS)

impact_df = pd.DataFrame(
    {
        "scenario": [s for s, _ in SCENARIOS],
        f"Low-β user (β={betas_true[low_user]:.2f})": probs_low,
        f"High-β user (β={betas_true[high_user]:.2f})": probs_high,
    }
)
impact_df
'''))

cells.append(code(r'''
def plot_feature_impact(impact_df: pd.DataFrame) -> plt.Figure:
    melted = impact_df.melt(
        id_vars="scenario",
        var_name="user profile",
        value_name="P(complete)",
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=melted,
        x="scenario",
        y="P(complete)",
        hue="user profile",
        palette=["C3", "C0"],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Predicted P(complete)")
    ax.set_title("Feature impact on completion probability  ·  high-β vs low-β user")
    ax.legend(title=None, loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return fig


fig = plot_feature_impact(impact_df)
save_figure(fig, "05_feature_impact")
plt.show()
'''))

cells.append(md(r'''
---

### Summary of saved figures

```
ml/notebooks/figures/
├── 01_behavioral_manifold.png
├── 02_beta_calibration.png
├── 03_reliability_diagram.png
├── 04_learning_curve.png
└── 05_feature_impact.png
```

These can be embedded directly into the project writeup without re-running
the notebook.
'''))

# ---------------------------------------------------------------------------
# Assemble & write
# ---------------------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (FollowThrough)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10",
    },
}

NOTEBOOK_PATH.write_text(nbf.writes(nb))
print(f"Wrote notebook with {len(cells)} cells to {NOTEBOOK_PATH}")
