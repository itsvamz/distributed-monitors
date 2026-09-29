"""Run E0-E5 over many seeds and print a report with 95% bootstrap CIs.

    python -m experiments.run_all            # 30 seeds (default)
    python -m experiments.run_all --seeds 5  # quick check
Outputs results/report.json (aggregate) and results/seed_results.json.
"""
import argparse
import time

import numpy as np

from experiments.common import save_json
from src.pipeline import run_many


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n-clean", type=int, default=200)
    ap.add_argument("--n-per-attack", type=int, default=40)
    ap.add_argument("--cap", type=float, default=0.05)
    a = ap.parse_args()
    t0 = time.time()
    agg, per_seed = run_many(range(a.seeds), n_clean=a.n_clean, n_per_attack=a.n_per_attack, cap=a.cap)
    rules, styles = agg["rules"], agg["styles"]
    print(f"{a.seeds} seeds, {agg['n_runs']} runs each ({agg['n_clean']} clean / {agg['n_harmful']} harmful), cap={a.cap:.0%}")
    print("\nAttacker success rate (mean over seeds; lower is better for the defender)")
    print("rule".ljust(15) + "".join(s[:13].ljust(14) for s in styles) + "FPR")
    for i, r in enumerate(rules):
        print(r.ljust(15) + "".join(f"{agg['payoff_mean'][i][j]:.2f}".ljust(14) for j in range(len(styles)))
              + f"{agg['fpr_mean'][r]:.3f}")
    m, lo, hi = agg["security_value"]
    print(f"\nSecurity value (equilibrium attacker success): {m:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
    m2, lo2, hi2 = agg["best_pure_value"]
    print(f"Best fixed rule worst case:                    {m2:.3f}  [95% CI {lo2:.3f}-{hi2:.3f}]")
    g, glo, ghi = agg["mixing_gain"]
    print(f"Gain from randomizing rules:                   {g:.3f}  [95% CI {glo:.3f}-{ghi:.3f}]")
    print("Attacker best responses (count over seeds):", agg["best_response_counts"])
    print("Average defender mix:", {r: round(w, 2) for r, w in zip(rules, agg["defender_mix_mean"])})
    print("\nRetry curves (mean):")
    for model, curve in agg["retry"].items():
        print(" ", model, {k: round(v[0], 3) for k, v in curve.items()})
    print(f"\nRandom-alarm baseline lead time: {agg['random_baseline_lead']:.2f} steps")
    print(f"Elapsed {time.time()-t0:.1f}s")
    save_json(agg, "results/report.json")
    save_json(per_seed, "results/seed_results.json")


if __name__ == "__main__":
    main()
