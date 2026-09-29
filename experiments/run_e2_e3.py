"""E2 (shared blind spots) + E3 (fusion rules) for one seed."""
from src.pipeline import evaluate


def main():
    r = evaluate(seed=2)
    print("Per-node caught-in-time rate by style (node-level cap 5%):")
    for n, d in r["node_style_catch"].items():
        print(f"  {n:26s}", {s: round(v, 2) for s, v in d.items()})
    print("\nP(B misses | A misses):")
    for k, v in r["blind_spots"].items():
        print(f"  {k}: {v if v is None else round(v, 2)}")
    print("\nFusion rules (attacker success by style):")
    for n in r["rules"]:
        rep = r["report"][n]
        print(f"  {n:14s} FPR={rep['fpr']:.3f}", {s: round(v['attacker_success'], 2) for s, v in rep["styles"].items()})


if __name__ == "__main__":
    main()
