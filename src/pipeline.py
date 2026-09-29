"""pipeline.py — E0 -> E1 -> E2 -> E3 -> E5 for one seed, plus multi-seed
aggregation with bootstrap confidence intervals."""

from __future__ import annotations

import random

import numpy as np

from src.calibration import calibrate_rule, calibrate_thresholds
from src.fusion import N1, N2, N2C, N3, N4, make_rules
from src.game import best_pure_strategy_value, solve_zero_sum_game
from src.metrics import (bootstrap_ci, caught_in_time, detection_delay, lead_time,
                         memory_funnel_under_rule, node_caught_in_time,
                         random_baseline_lead_time, retry_success_forgetful,
                         retry_success_sharing, shared_blind_spot)
from src.monitors import (AnomalyScorerNode, CusumNode, HeuristicJudge,
                          ProvenanceMonitorNode, RuleMatcherNode)
from src.traces import STYLES, Run, generate_dataset

FPR_CAP = 0.05
BASE_NODES = [N1, N2, N3, N4]


def build_monitors(clean_runs):
    n2, n2c = AnomalyScorerNode(), CusumNode()
    n2.fit(clean_runs); n2c.fit(clean_runs)          # clean data only
    mons = [RuleMatcherNode(), n2, n2c, HeuristicJudge(), ProvenanceMonitorNode()]
    return {m.name: m for m in mons}


def score_all(runs):
    clean = [r for r in runs if r.label == "clean"]
    mons = build_monitors(clean)
    return {r.run_id: {n: m.score_run(r) for n, m in mons.items()} for r in runs}


def evaluate(seed=0, n_clean=200, n_per_attack=40, cap=FPR_CAP, retry_ks=(1, 2, 5, 10)):
    runs = generate_dataset(n_clean=n_clean, n_per_attack=n_per_attack, seed=seed)
    scores = score_all(runs)
    clean = [r for r in runs if r.label == "clean"]
    harmful = [r for r in runs if r.label == "harmful"]
    by_style = {s: [r for r in harmful if r.attack_style == s] for s in STYLES}

    # ---- E1/E2: node-level view at the same cap -------------------------
    node_thr = calibrate_thresholds(
        {n: {r.run_id: scores[r.run_id][n] for r in clean} for n in BASE_NODES + [N2C]}, cap)
    node_style = {n: {s: float(np.mean([node_caught_in_time(scores[r.run_id][n], node_thr[n], r.harm_step)
                                        for r in by_style[s]])) for s in STYLES} for n in BASE_NODES + [N2C]}
    hit = {n: [node_caught_in_time(scores[r.run_id][n], node_thr[n], r.harm_step) for r in harmful]
           for n in BASE_NODES}
    blind = {f"{a}|{b}": shared_blind_spot(hit[a], hit[b]) for a in BASE_NODES for b in BASE_NODES if a != b}
    best_single = max(BASE_NODES, key=lambda n: np.mean(hit[n]))

    # ---- E3: fusion rules at the same rule-level cap ---------------------
    rules = make_rules(best_single)
    report, alarms = {}, {}
    for name, rule in rules.items():
        thr, fpr = calibrate_rule(rule, clean, scores, cap)
        per_style, alarms[name] = {}, {}
        for s in STYLES:
            res = [rule.apply(scores[r.run_id], thr, len(r.steps)) for r in by_style[s]]
            alarms[name][s] = [x.alarm_step for x in res]
            caught = [caught_in_time(r.harm_step, x.alarm_step) for r, x in zip(by_style[s], res)]
            delays = [detection_delay(x.alarm_step, r.onset_step) for r, x in zip(by_style[s], res)]
            leads = [lead_time(r.harm_step, x.alarm_step) for r, x in zip(by_style[s], res)]
            per_style[s] = {
                "attacker_success": 1 - float(np.mean(caught)),
                "mean_delay": _mean([d for d in delays if d is not None]),
                "mean_lead": _mean([l for l in leads if l is not None and l > 0]),
                "n3_calls": float(np.mean([x.n3_calls for x in res])),
            }
        clean_calls = float(np.mean([rule.apply(scores[r.run_id], thr, len(r.steps)).n3_calls for r in clean]))
        report[name] = {"fpr": fpr, "thresholds": thr, "styles": per_style, "n3_calls_clean": clean_calls}

    # ---- E4 (memory funnel) ---------------------------------------------
    funnel = {name: memory_funnel_under_rule(by_style["delayed_memory"], alarms[name]["delayed_memory"])
              for name in rules}

    # ---- E5: solve the game -----------------------------------------------
    names = list(rules)
    payoff = np.array([[report[n]["styles"][s]["attacker_success"] for s in STYLES] for n in names])
    eq = solve_zero_sum_game(payoff)
    bi, bv = best_pure_strategy_value(payoff)
    v = eq["game_value"]
    retry = {"forgetful": {k: retry_success_forgetful(v, k) for k in retry_ks},
             "sharing": {k: retry_success_sharing(v, k) for k in retry_ks}}
    rb = random_baseline_lead_time(by_style["naive"][0].harm_step, len(by_style["naive"][0].steps),
                                   cap, random.Random(seed))
    return {
        "seed": seed, "n_runs": len(runs), "n_clean": len(clean), "n_harmful": len(harmful),
        "styles": STYLES, "rules": names, "best_single": best_single, "node_thresholds": node_thr,
        "node_style_catch": node_style, "blind_spots": blind, "report": report, "funnel": funnel,
        "payoff": payoff.tolist(), "security_value": v, "defender_mix": eq["defender_mix"].tolist(),
        "best_response": [STYLES[j] for j in eq["best_response_cols"]],
        "best_pure_rule": names[bi], "best_pure_value": bv, "retry": retry,
        "random_baseline_lead": rb,
    }


def timing_curve(seed=0, caps=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3), n_clean=200, n_per_attack=40):
    """AMOC-style data: for each rule and false-alarm cap, the caught-in-time
    rate and mean detection delay over all harmful runs."""
    runs = generate_dataset(n_clean=n_clean, n_per_attack=n_per_attack, seed=seed)
    scores = score_all(runs)
    clean = [r for r in runs if r.label == "clean"]
    harmful = [r for r in runs if r.label == "harmful"]
    hit = {n: [node_caught_in_time(scores[r.run_id][n], calibrate_thresholds(
        {n: {c.run_id: scores[c.run_id][n] for c in clean}}, 0.05)[n], r.harm_step) for r in harmful] for n in BASE_NODES}
    rules = make_rules(max(BASE_NODES, key=lambda n: np.mean(hit[n])))
    out = {}
    for name, rule in rules.items():
        pts = []
        for cap in caps:
            thr, fpr = calibrate_rule(rule, clean, scores, cap)
            res = [rule.apply(scores[r.run_id], thr, len(r.steps)) for r in harmful]
            caught = [caught_in_time(r.harm_step, x.alarm_step) for r, x in zip(harmful, res)]
            dl = [detection_delay(x.alarm_step, r.onset_step) for r, x in zip(harmful, res) if x.alarm_step is not None]
            pts.append({"cap": cap, "fpr": fpr, "caught": float(np.mean(caught)), "delay": _mean(dl)})
        out[name] = pts
    return out


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def run_many(seeds, **kw):
    """Aggregate evaluate() over seeds with bootstrap 95% CIs."""
    results = [evaluate(seed=s, **kw) for s in seeds]
    rules, styles = results[0]["rules"], results[0]["styles"]
    payoff = np.array([r["payoff"] for r in results])              # seeds x rules x styles
    agg = {"seeds": list(seeds), "rules": rules, "styles": styles,
           "payoff_mean": payoff.mean(0).tolist(),
           "payoff_ci": [[bootstrap_ci(payoff[:, i, j])[1:] for j in range(len(styles))] for i in range(len(rules))],
           "security_value": bootstrap_ci([r["security_value"] for r in results]),
           "best_pure_value": bootstrap_ci([r["best_pure_value"] for r in results]),
           "mixing_gain": bootstrap_ci([r["best_pure_value"] - r["security_value"] for r in results]),
           "defender_mix_mean": np.mean([r["defender_mix"] for r in results], axis=0).tolist(),
           "best_response_counts": _count([b for r in results for b in r["best_response"]]),
           "best_pure_rule_counts": _count([r["best_pure_rule"] for r in results]),
           "fpr_mean": {n: float(np.mean([r["report"][n]["fpr"] for r in results])) for n in rules},
           "n3_calls": {n: {"clean": float(np.mean([r["report"][n]["n3_calls_clean"] for r in results])),
                            **{s: float(np.mean([r["report"][n]["styles"][s]["n3_calls"] for r in results])) for s in styles}}
                        for n in rules},
           "mean_lead": {n: {s: _mean([r["report"][n]["styles"][s]["mean_lead"] for r in results
                                       if r["report"][n]["styles"][s]["mean_lead"] is not None]) for s in styles} for n in rules},
           "mean_delay": {n: {s: _mean([r["report"][n]["styles"][s]["mean_delay"] for r in results
                                        if r["report"][n]["styles"][s]["mean_delay"] is not None]) for s in styles} for n in rules},
           "blind_spots": {k: _mean([r["blind_spots"][k] for r in results if r["blind_spots"][k] is not None])
                           for k in results[0]["blind_spots"]},
           "node_style_catch": {n: {s: float(np.mean([r["node_style_catch"][n][s] for r in results])) for s in styles}
                                for n in results[0]["node_style_catch"]},
           "funnel": {n: {k: float(np.mean([r["funnel"][n][k] for r in results])) for k in ("written", "retrieved", "acted_on")}
                      for n in rules},
           "retry": {m: {k: bootstrap_ci([r["retry"][m][k] for r in results]) for k in results[0]["retry"][m]}
                     for m in ("forgetful", "sharing")},
           "random_baseline_lead": _mean([r["random_baseline_lead"] for r in results]),
           "n_runs": results[0]["n_runs"], "n_clean": results[0]["n_clean"], "n_harmful": results[0]["n_harmful"]}
    return agg, results


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out
