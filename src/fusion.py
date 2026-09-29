"""fusion.py — E3: turn per-step suspicion scores into one alarm.

Every rule has the same signature: given per-node scores for ONE run
(node -> list[float]) and per-node thresholds, return the first alarm step
(or None) plus how many expensive N3 calls it needed. Each rule also
declares `.nodes`, the nodes it depends on, so calibration.py can tune
exactly those nodes to hit the shared false-alarm cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

NodeScores = dict[str, list[float]]
Thresholds = dict[str, float]

N1, N2, N2C, N3, N4 = ("N1_rule_matcher", "N2_anomaly_scorer", "N2c_cusum",
                       "N3_llm_judge_heuristic", "N4_provenance_monitor")


def _first_over(scores: list[float], thr: float) -> Optional[int]:
    for i, s in enumerate(scores):
        if s >= thr:
            return i
    return None


@dataclass
class FusionResult:
    alarm_step: Optional[int]
    n3_calls: int


class FusionRule:
    name = "base"
    nodes: list[str] = []

    def apply(self, node_scores: NodeScores, thr: Thresholds, n_steps: int) -> FusionResult:
        raise NotImplementedError


class SingleNode(FusionRule):
    def __init__(self, node: str):
        self.node, self.nodes, self.name = node, [node], "best_single"

    def apply(self, ns, thr, n_steps):
        a = _first_over(ns[self.node], thr[self.node])
        return FusionResult(a, n_steps if self.node == N3 else 0)


class AnyNodeOR(FusionRule):
    def __init__(self, nodes: list[str], name: str = "or_all"):
        self.nodes, self.name = list(nodes), name

    def apply(self, ns, thr, n_steps):
        firsts = [a for n in self.nodes if (a := _first_over(ns[n], thr[n])) is not None]
        return FusionResult(min(firsts) if firsts else None,
                            n_steps if N3 in self.nodes else 0)


class MajorityVote(FusionRule):
    def __init__(self, nodes: list[str], k: int = 2, name: str = "majority_2of4"):
        self.nodes, self.k, self.name = list(nodes), k, name

    def apply(self, ns, thr, n_steps):
        for t in range(n_steps):
            votes = sum(1 for n in self.nodes if ns[n][t] >= thr[n])
            if votes >= self.k:
                return FusionResult(t, n_steps if N3 in self.nodes else 0)
        return FusionResult(None, n_steps if N3 in self.nodes else 0)


class Cascade(FusionRule):
    """Cheap-first cascade.
    * Direct nodes (N1 rules, N4 provenance) alarm on their own.
    * The anomaly scorer N2 only *screens*: when it exceeds a relaxed
      'call' level (margin x its threshold) the costly judge N3 is asked,
      and an alarm needs N3 to agree.
    Cost: N3 runs only on screened steps. Risk: if N3 is fooled, everything
    that relied on N2's screening slips through (shared-blind-spot effect)."""
    name = "cascade"

    def __init__(self, direct=(N1, N4), screen=N2, confirm=N3, margin: float = 0.5):
        self.direct, self.screen, self.confirm, self.margin = list(direct), screen, confirm, margin
        self.nodes = [*self.direct, screen, confirm]

    def apply(self, ns, thr, n_steps):
        calls = 0
        for t in range(n_steps):
            if any(ns[n][t] >= thr[n] for n in self.direct):
                return FusionResult(t, calls)
            if ns[self.screen][t] >= self.margin * thr[self.screen]:
                calls += 1
                if ns[self.confirm][t] >= thr[self.confirm]:
                    return FusionResult(t, calls)
        return FusionResult(None, calls)


def make_rules(best_single_node: str) -> dict[str, FusionRule]:
    return {
        "best_single": SingleNode(best_single_node),
        "or_all": AnyNodeOR([N1, N2, N3, N4], "or_all"),
        "or_cusum": AnyNodeOR([N1, N2C, N3, N4], "or_cusum"),
        "majority_2of4": MajorityVote([N1, N2, N3, N4], 2),
        "cascade": Cascade(),
    }
