"""Data-driven figures (need results/report.json from run_all) -> docs/figures/

    python -m experiments.run_all && python -m experiments.make_figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.diagram_kit import W, save  # also sets IEEE-like fonts
from experiments.make_diagrams import fig15_hypotheses
from src.pipeline import score_all, timing_curve
from src.traces import STYLES, generate_dataset

SHORT = {"naive": "naive", "stealthy": "stealthy", "judge_injection": "judge-inj.",
         "delayed_memory": "delayed-mem.", "low_and_slow": "low-and-slow"}
RULE = {"best_single": "best single", "or_all": "OR (all)", "or_cusum": "OR (CUSUM)",
        "majority_2of4": "majority 2/4", "cascade": "cascade"}
NODE = {"N1_rule_matcher": "N1", "N2_anomaly_scorer": "N2", "N3_llm_judge_heuristic": "N3",
        "N4_provenance_monitor": "N4"}
GREYS = ["#222222", "#666666", "#999999", "#CCCCCC", "#EEEEEE"]


def heat(ax, m, xl, yl, fmt="{:.2f}", cmap="Reds", vmin=0, vmax=1, mark=None):
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xl)), xl, fontsize=6)
    ax.set_yticks(range(len(yl)), yl, fontsize=6)
    for i in range(len(yl)):
        for j in range(len(xl)):
            v = m[i][j]
            ax.text(j, i, "-" if v is None or np.isnan(v) else fmt.format(v), ha="center", va="center",
                    fontsize=6, color="white" if (v is not None and not np.isnan(v) and v > 0.6 * vmax) else "black")
    if mark:
        for (i, j) in mark:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.4))
    return im


def fig_payoff(a):
    fig, ax = plt.subplots(figsize=(W, 2.15))
    P = np.array(a["payoff_mean"])
    mix = np.array(a["defender_mix_mean"])
    heat(ax, P, [SHORT[s] for s in a["styles"]], [RULE[r] for r in a["rules"]],
         mark=[(int(np.argmax(P[i])), int(np.argmax(P[i]))) for i in range(0)] +
              [(i, int(np.argmax(P[i]))) for i in range(len(P))])
    ax.set_title("Attacker success (boxed: attacker's best reply per rule)", fontsize=6.6)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_payoff")


def fig_blind(a):
    nodes = list(NODE)
    m = np.array([[np.nan if x == y else (a["blind_spots"][f"{x}|{y}"] or np.nan) for y in nodes] for x in nodes])
    fig, ax = plt.subplots(figsize=(W, 2.0))
    heat(ax, m, [NODE[n] for n in nodes], [NODE[n] for n in nodes], cmap="Blues")
    ax.set_xlabel("node B also misses", fontsize=6.4)
    ax.set_ylabel("given node A missed", fontsize=6.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_blind")


def fig_nodestyle(a):
    nodes = ["N1_rule_matcher", "N2_anomaly_scorer", "N2c_cusum", "N3_llm_judge_heuristic", "N4_provenance_monitor"]
    lab = ["N1", "N2", "N2c", "N3", "N4"]
    m = np.array([[a["node_style_catch"][n][s] for s in a["styles"]] for n in nodes])
    fig, ax = plt.subplots(figsize=(W, 1.9))
    heat(ax, m, [SHORT[s] for s in a["styles"]], lab, cmap="Greens")
    ax.set_title("Caught-in-time rate per node (5% node-level cap)", fontsize=6.6)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_nodestyle")


def fig_funnel(a):
    fig, ax = plt.subplots(figsize=(W, 2.1))
    rules = a["rules"]
    x = np.arange(len(rules))
    for k, (stage, g) in enumerate(zip(["written", "retrieved", "acted_on"], ["#BBBBBB", "#777777", "#222222"])):
        ax.bar(x + (k - 1) * 0.26, [a["funnel"][r][stage] for r in rules], 0.26, label=stage.replace("_", " "),
               color=g, edgecolor="black", lw=0.4)
    ax.set_xticks(x, [RULE[r] for r in rules], fontsize=5.8, rotation=12)
    ax.set_ylabel("share of runs reaching stage", fontsize=6.4)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=5.8, ncol=3, frameon=False, loc="upper center")
    ax.tick_params(labelsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_funnel")


def fig_retry(a):
    fig, ax = plt.subplots(figsize=(W, 2.1))
    ks = [int(k) for k in a["retry"]["forgetful"]]
    for m_, ls, mk in [("forgetful", "-", "o"), ("sharing", "--", "s")]:
        mean = [a["retry"][m_][str(k)][0] for k in ks]
        lo = [a["retry"][m_][str(k)][1] for k in ks]
        hi = [a["retry"][m_][str(k)][2] for k in ks]
        ax.plot(ks, mean, ls, marker=mk, ms=3, color="black", lw=0.8, label=m_)
        ax.fill_between(ks, lo, hi, color="#999999", alpha=0.35, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("attempts k", fontsize=6.4)
    ax.set_ylabel("attacker success", fontsize=6.4)
    ax.set_ylim(0.8, 1.01)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_retry")


def fig_timing(curve):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.1), sharex=True)
    mk = {"best_single": "o", "or_all": "s", "or_cusum": "^", "majority_2of4": "D", "cascade": "x"}
    ls = {"best_single": ":", "or_all": "-", "or_cusum": "--", "majority_2of4": "-.", "cascade": (0, (1, 1))}
    for r, pts in curve.items():
        f = [p["fpr"] for p in pts]
        a1.plot(f, [p["caught"] for p in pts], ls=ls[r], marker=mk[r], ms=3, color="black", lw=0.8, label=RULE[r])
        d = [p["delay"] if p["delay"] is not None else np.nan for p in pts]
        a2.plot(f, d, ls=ls[r], marker=mk[r], ms=3, color="black", lw=0.8)
    a1.set_ylabel("caught in time", fontsize=6.4)
    a2.set_ylabel("mean delay (steps)", fontsize=6.4)
    a2.set_xlabel("false-alarm rate on clean runs", fontsize=6.4)
    a1.legend(fontsize=5.6, ncol=2, frameon=False, loc="upper left")
    for a_ in (a1, a2):
        a_.tick_params(labelsize=6)
        a_.grid(alpha=0.25, lw=0.4)
    a1.text(0.99, 0.05, "(a)", transform=a1.transAxes, ha="right", fontsize=6.6, fontweight="bold")
    a2.text(0.99, 0.88, "(b)", transform=a2.transAxes, ha="right", fontsize=6.6, fontweight="bold")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_timing")


def fig_signatures(seed=0):
    runs = generate_dataset(n_clean=100, n_per_attack=40, seed=seed)
    sc = score_all(runs)
    fig, ax = plt.subplots(figsize=(W, 2.15))
    clean = np.mean([sc[r.run_id]["N2_anomaly_scorer"][:10] for r in runs if r.label == "clean"], axis=0)
    ax.axhline(clean.mean(), color="black", lw=0.5, ls=":", label="clean mean")
    sty = {"naive": "-", "stealthy": "--", "judge_injection": "-.", "delayed_memory": ":", "low_and_slow": (0, (5, 1, 1, 1))}
    for s in STYLES:
        rs = [r for r in runs if r.attack_style == s]
        rel = {}
        for r in rs:
            for t, v in enumerate(sc[r.run_id]["N2_anomaly_scorer"]):
                rel.setdefault(t - r.harm_step, []).append(v)
        xs = sorted(k for k in rel if k >= -8)
        ax.plot(xs, [np.mean(rel[k]) for k in xs], ls=sty[s], color="black", lw=0.9, label=SHORT[s])
    ax.set_xlabel("steps relative to harm step", fontsize=6.4)
    ax.set_ylabel("mean N2 score", fontsize=6.4)
    ax.legend(fontsize=5.5, ncol=2, frameon=False, loc="upper left")
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_signatures")


if __name__ == "__main__":
    a = json.load(open("results/report.json"))
    fig_payoff(a); fig_blind(a); fig_nodestyle(a); fig_funnel(a); fig_retry(a)
    fig_signatures()
    fig_timing(timing_curve(seed=0))
    fig15_hypotheses([("Supported", "good"), ("Partly supported\n(mechanism only)", "warn"),
                      ("Supported", "good"), ("Partly supported\n(cuts ~38% of gain)", "warn")])
    print("figures written to docs/figures/")
