"""
Commitment-device recommender — decision tree over the user's behavioral profile.

Device levels (ints, matching ``users.current_device``):
  0 — Salience Nudge:           reminder notification 2h before a task
  1 — Implementation Intention: requires when/where/first-action at task creation
  2 — Planning Correction:      auto-inflates planned_duration by proj_bias_score
  3 — Virtual Stakes:           failure deducts streak; loss-framed notification
  4 — Precommitment Lock:       reschedule-within-24h requires friction confirmation

The decision is a strict priority ladder; see :func:`evaluate`.
"""
from __future__ import annotations

from dataclasses import dataclass


LEVEL_MIN = 0
LEVEL_MAX = 4

COLD_START_MIN_CHECKINS = 10
FAILURE_STREAK_THRESHOLD = 3
DE_ESCALATION_RATE = 0.90
DE_ESCALATION_WEEKS = 2
PROJ_BIAS_THRESHOLD = 0.3
MIN_STAY_WEEKS_FOR_DROP = 1  # baseline drops require ≥ 1 week at current level (anti-thrash)

BETA_LEVEL_1 = 0.65
BETA_LEVEL_3 = 0.75
BETA_LEVEL_4 = 0.85


@dataclass
class UserProfile:
    beta_proxy: float
    proj_bias_score: float
    drift_flag: str            # one of BOCDDetector.get_drift_status() values
    recent_failure_streak: int
    current_device: int        # 0–4
    weeks_at_current_level: int
    recent_completion_rate: float
    total_checkins: int


@dataclass
class RecommendationResult:
    recommended_device: int
    reason: str
    changed: bool
    previous_device: int


def _clamp(level: int) -> int:
    return max(LEVEL_MIN, min(LEVEL_MAX, level))


def _beta_baseline(beta_proxy: float) -> int:
    if beta_proxy < BETA_LEVEL_1:
        return 0
    if beta_proxy < BETA_LEVEL_3:
        return 1
    if beta_proxy < BETA_LEVEL_4:
        return 3
    return 4


def evaluate(profile: UserProfile) -> RecommendationResult:
    """Return the recommended commitment-device level for the user.

    Priority order (first match wins):
      1. Cold start (< 10 check-ins)               → Level 0
      2. Failure escalation (>= 3 consecutive)     → escalate from current device
      3. Sustained-success de-escalation
         (>= 90% for >= 2 weeks, no decline drift) → de-escalate from current device
      4. Drift response (improvement → de-esc, decline → esc)
      5. High projection bias (>0.3)               → Level 2
      6. Beta-based baseline                       → 0/1/3/4 from beta_proxy
    """
    prev = profile.current_device

    # 1. cold start
    if profile.total_checkins < COLD_START_MIN_CHECKINS:
        return RecommendationResult(
            recommended_device=0,
            reason="insufficient data",
            changed=(prev != 0),
            previous_device=prev,
        )

    # 2. escalation override
    if profile.recent_failure_streak >= FAILURE_STREAK_THRESHOLD:
        new = _clamp(prev + 1)
        return RecommendationResult(
            recommended_device=new,
            reason="repeated failures",
            changed=(new != prev),
            previous_device=prev,
        )

    # 3. de-escalation check
    if (
        profile.recent_completion_rate >= DE_ESCALATION_RATE
        and profile.weeks_at_current_level >= DE_ESCALATION_WEEKS
        and profile.drift_flag != "confirmed_decline"
    ):
        new = _clamp(prev - 1)
        return RecommendationResult(
            recommended_device=new,
            reason="sustained high performance",
            changed=(new != prev),
            previous_device=prev,
        )

    # 4. drift response
    if profile.drift_flag == "confirmed_improvement":
        new = _clamp(prev - 1)
        return RecommendationResult(
            recommended_device=new,
            reason="confirmed improvement",
            changed=(new != prev),
            previous_device=prev,
        )
    if profile.drift_flag == "confirmed_decline":
        new = _clamp(prev + 1)
        return RecommendationResult(
            recommended_device=new,
            reason="confirmed decline",
            changed=(new != prev),
            previous_device=prev,
        )

    # 5. projection-bias override
    if profile.proj_bias_score > PROJ_BIAS_THRESHOLD:
        # Drop-guarded: don't slide *down* to L2 from a higher level without tenure
        if 2 < prev and profile.weeks_at_current_level < MIN_STAY_WEEKS_FOR_DROP:
            return RecommendationResult(
                recommended_device=prev,
                reason="proj bias (drop guarded)",
                changed=False,
                previous_device=prev,
            )
        return RecommendationResult(
            recommended_device=2,
            reason="high projection bias",
            changed=(prev != 2),
            previous_device=prev,
        )

    # 6. beta-based baseline (drop-guarded: prevents thrashing when β fluctuates
    # near a threshold or a failure-streak escalation gets undone by the next
    # noisy success). Increases pass through immediately.
    new = _beta_baseline(profile.beta_proxy)
    if new < prev and profile.weeks_at_current_level < MIN_STAY_WEEKS_FOR_DROP:
        return RecommendationResult(
            recommended_device=prev,
            reason="beta baseline (drop guarded)",
            changed=False,
            previous_device=prev,
        )
    return RecommendationResult(
        recommended_device=new,
        reason="beta baseline",
        changed=(new != prev),
        previous_device=prev,
    )


def apply_projection_correction(planned_duration: int, proj_bias_score: float) -> int:
    """Inflate ``planned_duration`` by the user's projection bias.

    Only corrects upward — if the user under-runs (negative bias), the original
    estimate stands.
    """
    corrected = round(planned_duration * (1 + proj_bias_score))
    return max(planned_duration, corrected)
