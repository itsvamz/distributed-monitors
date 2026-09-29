"""calibration.py — make every rule spend the SAME false-alarm budget.

Step 1 (node level): the loosest threshold at which a node alone alarms on
at most p of the clean runs.
Step 2 (rule level): for a fusion rule, search the per-node budget p so the
COMBINED rule alarms on at most `cap` of the clean runs. This is what makes
'any-node OR' automatically use stricter per-node thresholds than 'majority
vote' (the trade-off stated in the design), instead of assuming it.
Only clean runs are used; attacks never influence thresholds.
"""

from __future__ import annotations

NodeScores = dict[str, dict[str, list[float]]]   # node -> run_id -> scores
P_GRID = [0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.035, 0.025,
          0.02, 0.015, 0.01, 0.0075, 0.005, 0.0025, 0.0]


def calibrate_thresholds(clean_scores: NodeScores, fpr_cap: float = 0.05) -> dict[str, float]:
    thresholds = {}
    for node, per_run in clean_scores.items():
        maxima = sorted(max(v) for v in per_run.values())
        n = len(maxima)
        chosen = maxima[-1] + 1e-6                     # default: never alarms
        for t in sorted(set(round(m, 6) for m in maxima)):
            if sum(1 for m in maxima if m >= t) / n <= fpr_cap:
                chosen = t
                break
        thresholds[node] = chosen
    return thresholds


def rule_fpr(rule, thresholds, clean_runs, scores) -> float:
    alarmed = sum(1 for r in clean_runs
                  if rule.apply(scores[r.run_id], thresholds, len(r.steps)).alarm_step is not None)
    return alarmed / max(len(clean_runs), 1)


def calibrate_rule(rule, clean_runs, scores, cap: float = 0.05):
    """Return (thresholds, achieved_fpr) with achieved_fpr <= cap."""
    per_node = {n: {r.run_id: scores[r.run_id][n] for r in clean_runs} for n in rule.nodes}
    for p in P_GRID:
        thr = calibrate_thresholds(per_node, p)
        f = rule_fpr(rule, thr, clean_runs, scores)
        if f <= cap:
            return thr, f
    thr = calibrate_thresholds(per_node, 0.0)
    return thr, rule_fpr(rule, thr, clean_runs, scores)
