"""
Commitment device recommender — IMPLEMENT THIS (Nicole).

Decision tree operating on (beta_proxy, proj_bias_score, drift_flag, failure_streak).

Device levels:
  0 — Salience Nudge:          beta < 0.65, proj_bias < 0.2
  1 — Implementation Intention: beta 0.65–0.75
  2 — Planning Correction:      proj_bias > 0.3 (regardless of beta)
  3 — Virtual Stakes:           beta 0.75–0.85
  4 — Precommitment Lock:       beta > 0.85 OR 3+ consecutive Level-3 failures

Escalation override: 3+ consecutive failures at any level → escalate one level.
De-escalation: 90%+ completion rate sustained 2 weeks → drop one level.

The backend calls evaluate_device_level() from the checkins router.
Do not change the function signatures.
"""
from __future__ import annotations


def evaluate_device_level(
    beta_proxy: float,
    proj_bias_score: float,
    drift_flag: bool,
    failure_streak: int,
) -> int:
    """
    Return the recommended commitment device level (0–4).

    Args:
        beta_proxy: estimated present-bias parameter in [0, 1]
        proj_bias_score: projection bias (fractional overrun, e.g. 0.3 = 30% over)
        drift_flag: True if BOCD detected a changepoint this cycle
        failure_streak: number of consecutive 0.0-completion check-ins

    Returns:
        int in [0, 4]
    """
    raise NotImplementedError("Implement device decision tree — Nicole")


def apply_projection_correction(planned_duration: int, proj_bias_score: float) -> int:
    """
    Return a corrected task duration accounting for the user's projection bias.

    Formula: corrected = planned_duration * (1 + proj_bias_score)
    Applied when user is at Level 2 and proj_bias_score > 0.3.

    Args:
        planned_duration: original duration in minutes
        proj_bias_score: fractional overrun factor (e.g. 0.3 → 30% inflation)

    Returns:
        Corrected duration in minutes (rounded to nearest integer).
    """
    return round(planned_duration * (1 + proj_bias_score))
