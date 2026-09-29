"""E0 + E1: dataset and per-node scores for one example run of each style."""
from src.pipeline import score_all
from src.traces import STYLES, generate_dataset


def main():
    runs = generate_dataset(n_clean=30, n_per_attack=5, seed=1)
    scores = score_all(runs)
    print(f"{len(runs)} runs; styles: {STYLES}")
    for s in STYLES:
        r = next(x for x in runs if x.attack_style == s)
        print(f"\n{s}: run={r.run_id} onset={r.onset_step} harm={r.harm_step}")
        for node, sc in scores[r.run_id].items():
            print(f"  {node:26s}", [round(v, 2) for v in sc])


if __name__ == "__main__":
    main()
