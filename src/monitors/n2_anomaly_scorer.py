"""N2 — Anomaly scorer.

Scores how far each step's embedding drifts from a "clean" centroid learned
from clean runs. Strength: needs no rule, catches unusual patterns.
Weakness (reproduced here too): a low-drift ("stealthy") step scores only
mildly above baseline, so N2 alone cannot separate "unusual but fine" from
"unusual and bad" without a hand-tuned threshold — exactly the trade-off
the paper's fusion-rule comparison is about.

Uses squared Mahalanobis-style distance (diagonal covariance) from the
clean-run centroid, which is the standard, cheap anomaly score for
fixed-length embeddings when no labeled anomaly examples are used to fit
the scorer itself (only clean runs are used to fit it, by design — the
detector must not have seen attacks at fit time).
"""

from __future__ import annotations

import numpy as np

from src.monitors.base import MonitorNode
from src.traces import Run


class AnomalyScorerNode(MonitorNode):
    name = "N2_anomaly_scorer"

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, clean_runs: list[Run]) -> None:
        vectors = [step.embedding for run in clean_runs for step in run.steps]
        arr = np.array(vectors)
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0) + 1e-6  # avoid div-by-zero

    def score_run(self, run: Run) -> list[float]:
        if self.mean is None:
            raise RuntimeError("AnomalyScorerNode.fit() must be called on clean runs first")
        scores = []
        for step in run.steps:
            v = np.array(step.embedding)
            z = (v - self.mean) / self.std
            scores.append(float(np.mean(z ** 2)))  # mean squared z-score
        return scores


class CusumNode(AnomalyScorerNode):
    """N2c — persistence variant of N2 (Page's CUSUM, 1954).

    Instead of thresholding each step's anomaly score alone, accumulate the
    excess of the score over a reference level k:  S_t = max(0, S_{t-1} +
    (score_t - k)).  A single spike decays; a sustained mild drift piles up.
    This is the natural defence against the 'stealthy' style, and it is the
    defender move that the 'low_and_slow' attacker is built to answer.
    """
    name = "N2c_cusum"

    def __init__(self, k: float = 1.15):
        super().__init__()
        self.k = k

    def score_run(self, run: Run) -> list[float]:
        raw = super().score_run(run)
        s, out = 0.0, []
        for r in raw:
            s = max(0.0, s + (r - self.k))
            out.append(s)
        return out
