"""Common interface every monitor node (N1-N4) implements.

A monitor's job is: given a Run, produce one suspicion score per step
(a float in roughly [0, inf), higher = more suspicious). It does not decide
alarm/no-alarm itself — that threshold is fusion's job (fusion.py), so the
same score can be recombined under different fusion rules without
recomputing anything.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.traces import Run


class MonitorNode(ABC):
    name: str = "unnamed"

    @abstractmethod
    def score_run(self, run: Run) -> list[float]:
        """Return one suspicion score per step, same length as run.steps."""
        raise NotImplementedError

    def score_dataset(self, runs: list[Run]) -> dict[str, list[float]]:
        return {run.run_id: self.score_run(run) for run in runs}
