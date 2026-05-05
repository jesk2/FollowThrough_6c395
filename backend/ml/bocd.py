"""
Bayesian Online Changepoint Detection (BOCD) for daily completion-rate streams.

Reference: Adams & MacKay 2007. One detector instance per user; state is persisted
to ``users.detector_state`` (JSONB) so it survives restarts.

The observation model is Beta-distributed completion rates. For each run-length
hypothesis r we keep Beta sufficient stats (alpha[r], b[r]); the predictive
likelihood of a new observation y is Beta(y; alpha[r], b[r]).

Drift status vocabulary:
  - "stable"                  : no flag pending
  - "potential"               : a changepoint was flagged within the last 5 days
  - "confirmed_decline"       : level shift, post-flag mean < pre-flag - 0.15
  - "confirmed_improvement"   : level shift, post-flag mean > pre-flag + 0.15
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist


CP_THRESHOLD = 0.5            # flag when P(short run) > this
SHORT_RUN_K = 3               # define "short run" as r <= K
WARMUP_STEPS = 5              # don't flag during warmup where short-run is trivially likely
CONSEC_HIGH_CP_REQUIRED = 2   # require N consecutive elevated days before flagging
CLASSIFICATION_WINDOW = 7     # days after a flag before classifying it
LEVEL_SHIFT_CP_MEAN = 0.3     # mean cp_prob in window must exceed this for level shift
PRE_FLAG_WINDOW = 14          # days of pre-flag history used for direction comparison
DIRECTION_TAIL = 3            # use last N post-flag obs for direction — short enough to
                              # let recovery dominate after a transient dip, long enough
                              # to smooth single-day flukes
SHIFT_MAGNITUDE = 0.20        # post-pre mean delta required for direction
PRIOR_ALPHA = 3.0             # Beta prior — mean ~0.67
PRIOR_B = 1.5
EPS = 1e-3                    # clip y away from 0 / 1 for Beta PDF stability

# NOTE on the changepoint metric:
# The spec called for cp_prob = run_length_probs[0], but with a constant hazard
# rate H that quantity equals H exactly after normalization (provably, by
# inspecting the Adams-MacKay recursion) — so it can never cross a 0.5 threshold.
# Instead we use P(r_t <= K) as the changepoint signal: after a real regime
# shift, posterior mass collapses onto small r, which is exactly what we want
# to detect. Threshold semantics ("flag if cp_prob > 0.5") still apply.


class BOCDDetector:
    def __init__(self, hazard_rate: float = 1 / 30):
        self.hazard_rate = hazard_rate
        self.run_length_probs: np.ndarray = np.array([1.0])
        self.alpha: np.ndarray = np.array([PRIOR_ALPHA])
        self.b: np.ndarray = np.array([PRIOR_B])

        self.t: int = 0
        self.consecutive_high_cp: int = 0  # gates the stable -> potential transition
        self.pending_flag_days: int = 0
        self.drift_status: str = "stable"
        self.last_update_date: Optional[str] = None  # ISO date string

        # rolling buffers for transient/level-shift classification
        self.pre_flag_rates: list[float] = []   # last PRE_FLAG_WINDOW pre-flag rates
        self.post_flag_rates: list[float] = []  # rates since the most recent flag
        self.post_flag_cp_probs: list[float] = []

        # most recent update result, exposed via last_result()
        self._last_changepoint_prob: float = 0.0
        self._last_expected_run_length: float = 0.0
        self._last_regime_completion_rate: float = PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_B)
        self._last_flagged: bool = False

    # ------------------------------------------------------------------
    # core update
    # ------------------------------------------------------------------

    def update(self, y: float) -> float:
        """Ingest one daily completion rate; return the changepoint probability.

        Side effects: advances run-length distribution, updates Beta sufficient
        stats, refreshes drift_status / pending_flag_days, and stores the latest
        derived quantities for retrieval via :meth:`last_result`.
        """
        # scalar float in [EPS, 1-EPS]
        y_clipped = float(np.clip(y, EPS, 1.0 - EPS))

        # predictive likelihood of y under each run-length hypothesis
        # vector of float, length = len(self.run_length_probs); pred_probs[r] = Beta PDF at y for hypothesis r
        pred_probs = beta_dist.pdf(y_clipped, self.alpha, self.b)

        # joint = P(y | r_{t-1}, x_{1:t-1}) * P(r_{t-1} | x_{1:t-1})
        # vector of float, same length as pred_probs; unnormalized joint over old run-length hypotheses
        joint = pred_probs * self.run_length_probs

        # scalar float: total mass routed to r_t = 0 (changepoint bucket)
        cp_prob = float(np.sum(joint) * self.hazard_rate)
        # vector of float, same length as joint; growth[r] = mass routed to r_t = r+1
        growth = joint * (1.0 - self.hazard_rate)

        new_run_length_probs = np.concatenate(([cp_prob], growth))
        total = new_run_length_probs.sum() # for normalization
        if total <= 0 or not np.isfinite(total):
            # numerical collapse — reset to point mass at r=0
            new_run_length_probs = np.array([1.0])
        else:
            new_run_length_probs = new_run_length_probs / total

        # update sufficient stats: prepend fresh prior for r=0, then add (y, 1-y)
        # new_alpha, new_b: vector of float, length = len(self.alpha)+1; Beta sufficient stats per new run-length hypothesis
        new_alpha = np.concatenate(([PRIOR_ALPHA], self.alpha + y_clipped))
        new_b = np.concatenate(([PRIOR_B], self.b + (1.0 - y_clipped)))

        self.run_length_probs = new_run_length_probs
        self.alpha = new_alpha
        self.b = new_b
        self.t += 1

        # Changepoint score: posterior probability that the run length is short.
        # Suppressed during warmup, where short-run mass is trivially high.
        k = min(SHORT_RUN_K + 1, len(self.run_length_probs))
        cp_score = float(self.run_length_probs[:k].sum())
        if self.t <= WARMUP_STEPS:
            cp_score = 0.0

        expected_rl = float(np.dot(np.arange(len(self.run_length_probs)), self.run_length_probs))
        regime_rate = self._compute_regime_rate()
        flagged = cp_score > CP_THRESHOLD

        self._advance_classification(y_clipped, cp_score, flagged)

        self._last_changepoint_prob = cp_score
        self._last_expected_run_length = expected_rl
        self._last_regime_completion_rate = regime_rate
        self._last_flagged = flagged
        return cp_score # high is signal that smth changed

    def _compute_regime_rate(self) -> float:
        """E[completion rate] under the current run-length posterior."""
        means = self.alpha / (self.alpha + self.b)
        return float(np.dot(self.run_length_probs, means))

    def _advance_classification(self, y: float, cp_prob: float, flagged: bool) -> None:
        if self.drift_status == "stable":
            self.pre_flag_rates.append(y)
            if len(self.pre_flag_rates) > PRE_FLAG_WINDOW:
                self.pre_flag_rates.pop(0)

            # Sustained-elevation gate: require CONSEC_HIGH_CP_REQUIRED days in a
            # row above threshold before transitioning. A single anomalous day
            # in a stable regime briefly spikes cp_prob; we don't want that to
            # be treated as a candidate regime shift.
            if flagged:
                self.consecutive_high_cp += 1
            else:
                self.consecutive_high_cp = 0

            if self.consecutive_high_cp >= CONSEC_HIGH_CP_REQUIRED:
                self.drift_status = "potential"
                self.pending_flag_days = 1
                self.consecutive_high_cp = 0
                self.post_flag_rates = [y]
                self.post_flag_cp_probs = [cp_prob]
            return

        if self.drift_status == "potential":
            self.pending_flag_days += 1
            self.post_flag_rates.append(y)
            self.post_flag_cp_probs.append(cp_prob)

            if self.pending_flag_days >= CLASSIFICATION_WINDOW:
                mean_cp = float(np.mean(self.post_flag_cp_probs))
                # Compare the recent post-flag state (last DIRECTION_TAIL obs)
                # to pre-flag baseline. Using a wider tail than the unit-test
                # version because under realistic noise a 2-obs tail is
                # dominated by single days.
                tail = self.post_flag_rates[-DIRECTION_TAIL:] or self.post_flag_rates
                recent_post = float(np.mean(tail))
                pre_mean = (
                    float(np.mean(self.pre_flag_rates)) if self.pre_flag_rates else recent_post
                )
                delta = recent_post - pre_mean

                if mean_cp > LEVEL_SHIFT_CP_MEAN and abs(delta) > SHIFT_MAGNITUDE:
                    self.drift_status = (
                        "confirmed_improvement" if delta > 0 else "confirmed_decline"
                    )
                else:
                    # transient — return to stable, fold post-flag rates into history
                    self.drift_status = "stable"
                    for r in self.post_flag_rates:
                        self.pre_flag_rates.append(r)
                        if len(self.pre_flag_rates) > PRE_FLAG_WINDOW:
                            self.pre_flag_rates.pop(0)
                self.pending_flag_days = 0
            return

        # confirmed_decline / confirmed_improvement: stays sticky until reset()

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    def expected_run_length(self) -> float:
        return self._last_expected_run_length

    def regime_completion_rate(self) -> float:
        return self._last_regime_completion_rate

    def get_drift_status(self) -> str:
        return self.drift_status

    def last_result(self) -> dict:
        return {
            "changepoint_prob": self._last_changepoint_prob,
            "expected_run_length": self._last_expected_run_length,
            "regime_completion_rate": self._last_regime_completion_rate,
            "flagged": self._last_flagged,
            "drift_status": self.drift_status,
        }

    def reset(self) -> None:
        """Reset run-length distribution and classification state."""
        self.run_length_probs = np.array([1.0])
        self.alpha = np.array([PRIOR_ALPHA])
        self.b = np.array([PRIOR_B])
        self.t = 0
        self.consecutive_high_cp = 0
        self.pending_flag_days = 0
        self.drift_status = "stable"
        self.pre_flag_rates = []
        self.post_flag_rates = []
        self.post_flag_cp_probs = []
        self._last_changepoint_prob = 0.0
        self._last_expected_run_length = 0.0
        self._last_regime_completion_rate = PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_B)
        self._last_flagged = False

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "hazard_rate": self.hazard_rate,
            "run_length_probs": self.run_length_probs.tolist(),
            "alpha": self.alpha.tolist(),
            "b": self.b.tolist(),
            "t": self.t,
            "consecutive_high_cp": self.consecutive_high_cp,
            "pending_flag_days": self.pending_flag_days,
            "drift_status": self.drift_status,
            "last_update_date": self.last_update_date,
            "pre_flag_rates": list(self.pre_flag_rates),
            "post_flag_rates": list(self.post_flag_rates),
            "post_flag_cp_probs": list(self.post_flag_cp_probs),
            "last_changepoint_prob": self._last_changepoint_prob,
            "last_expected_run_length": self._last_expected_run_length,
            "last_regime_completion_rate": self._last_regime_completion_rate,
            "last_flagged": self._last_flagged,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BOCDDetector":
        det = cls(hazard_rate=d["hazard_rate"])
        det.run_length_probs = np.asarray(d["run_length_probs"], dtype=float)
        det.alpha = np.asarray(d["alpha"], dtype=float)
        det.b = np.asarray(d["b"], dtype=float)
        det.t = int(d.get("t", len(det.run_length_probs) - 1))
        det.consecutive_high_cp = int(d.get("consecutive_high_cp", 0))
        det.pending_flag_days = int(d["pending_flag_days"])
        det.drift_status = d["drift_status"]
        det.last_update_date = d.get("last_update_date")
        det.pre_flag_rates = list(d.get("pre_flag_rates", []))
        det.post_flag_rates = list(d.get("post_flag_rates", []))
        det.post_flag_cp_probs = list(d.get("post_flag_cp_probs", []))
        det._last_changepoint_prob = float(d.get("last_changepoint_prob", 0.0))
        det._last_expected_run_length = float(d.get("last_expected_run_length", 0.0))
        det._last_regime_completion_rate = float(
            d.get("last_regime_completion_rate", PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_B))
        )
        det._last_flagged = bool(d.get("last_flagged", False))
        return det


# ----------------------------------------------------------------------
# persistence helpers — read/write detector state on a User row
# ----------------------------------------------------------------------

def load_detector(user) -> BOCDDetector:
    """Reconstruct the detector for a user (or build a fresh one if no state yet)."""
    state = getattr(user, "detector_state", None)
    if not state:
        return BOCDDetector()
    return BOCDDetector.from_dict(state)


def save_detector(user, detector: BOCDDetector) -> None:
    """Serialize detector state onto the user. Caller commits the session."""
    user.detector_state = detector.to_dict()


def should_tick_today(detector: BOCDDetector, today: date) -> bool:
    """True if the detector hasn't been ticked yet for the given calendar day."""
    if detector.last_update_date is None:
        return True
    return detector.last_update_date != today.isoformat()


def mark_ticked(detector: BOCDDetector, on: date) -> None:
    detector.last_update_date = on.isoformat()
