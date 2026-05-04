"""
Bayesian Online Changepoint Detection (BOCD) — IMPLEMENT THIS (Nicole).

Reference: Adams & MacKay 2007.

One BOCDDetector instance per user.  The backend calls update() on every check-in
with that day's completion rate, and reads the returned changepoint probability to
decide whether to log a DriftEvent and re-evaluate the commitment device.

State: the run-length distribution P(r_t | x_{1:t}).
  - At each timestep, update via Bayes' rule using the observation.
  - Hazard rate = 1/30 (prior: regime change ~monthly).
  - When cp_prob > 0.5, the backend logs a potential DriftEvent.

Classification (done inside the detector after enough observations):
  - level_shift:  cp_prob stays elevated for 5+ consecutive days → durable change.
  - transient:    cp_prob subsides within 5 days → temporary disruption.

Do not change the class interface — the backend imports get_or_create_detector() and
calls detector.update().
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# Per-user detector registry (in-memory; state reconstructed from checkin history on restart)
_detectors: dict[str, "BOCDDetector"] = {}


class BOCDDetector:
    def __init__(self, hazard_rate: float = 1 / 30):
        self.hazard_rate = hazard_rate
        # run-length distribution — starts as a point mass at r=0
        self._run_length_dist: np.ndarray = np.array([1.0])
        self._t: int = 0
        # track recent cp_probs for classification
        self._recent_cp_probs: list[float] = []

    def update(self, daily_completion_rate: float) -> float:
        """
        Ingest today's completion rate and return the probability of a changepoint.

        Args:
            daily_completion_rate: float in [0, 1]

        Returns:
            Changepoint probability (scalar in [0, 1]).
        """
        raise NotImplementedError("Implement BOCD update step — Nicole")

    def classify(self) -> Optional[str]:
        """
        After update(), classify the event if cp_prob > 0.5.

        Returns:
            "level_shift" if elevated for 5+ consecutive days,
            "transient" if it subsided,
            None if no changepoint detected.
        """
        raise NotImplementedError("Implement BOCD classification — Nicole")

    def reset(self) -> None:
        """Reset run-length distribution (call after a confirmed level_shift)."""
        self._run_length_dist = np.array([1.0])
        self._t = 0
        self._recent_cp_probs = []


def get_or_create_detector(user_id: str) -> BOCDDetector:
    """Return the existing detector for this user, or create a new one."""
    if user_id not in _detectors:
        _detectors[user_id] = BOCDDetector()
    return _detectors[user_id]
