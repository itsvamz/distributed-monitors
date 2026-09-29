"""E5: solve the defender/attacker game for one seed."""
from src.pipeline import evaluate


def main():
    r = evaluate(seed=3)
    print("payoff rows=rules, cols=", r["styles"])
    for n, row in zip(r["rules"], r["payoff"]):
        print(f"  {n:14s}", [round(v, 2) for v in row])
    print(f"\nsecurity value {r['security_value']:.3f}; best fixed rule {r['best_pure_rule']} -> {r['best_pure_value']:.3f}")
    print("defender mix:", {n: round(w, 2) for n, w in zip(r["rules"], r["defender_mix"])})
    print("attacker best response:", r["best_response"])


if __name__ == "__main__":
    main()
