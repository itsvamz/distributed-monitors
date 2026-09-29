from src.calibration import calibrate_rule
from src.fusion import AnyNodeOR, N1, N2, N3, N4
from src.metrics import bootstrap_ci, memory_funnel_under_rule
from src.pipeline import evaluate, score_all
from src.traces import generate_dataset


def test_dataset_is_deterministic_per_seed():
    a = generate_dataset(10, 3, seed=4)
    b = generate_dataset(10, 3, seed=4)
    assert [r.run_id for r in a] == [r.run_id for r in b]


def test_calibrated_rule_respects_false_alarm_cap():
    runs = generate_dataset(100, 5, seed=1)
    sc = score_all(runs)
    clean = [r for r in runs if r.label == "clean"]
    for cap in (0.02, 0.05, 0.1):
        _, fpr = calibrate_rule(AnyNodeOR([N1, N2, N3, N4]), clean, sc, cap)
        assert fpr <= cap + 1e-12


def test_funnel_drops_to_zero_when_alarm_precedes_retrieval():
    runs = [r for r in generate_dataset(5, 3, seed=2) if r.attack_style == "delayed_memory"]
    f = memory_funnel_under_rule(runs, [4] * len(runs))        # alarm on the write step
    assert f["written"] == 1.0 and f["retrieved"] == 0.0 and f["acted_on"] == 0.0
    f2 = memory_funnel_under_rule(runs, [None] * len(runs))
    assert f2 == {"written": 1.0, "retrieved": 1.0, "acted_on": 1.0}


def test_bootstrap_ci_contains_mean_and_is_ordered():
    m, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert lo <= m <= hi


def test_evaluate_returns_consistent_game():
    r = evaluate(seed=0, n_clean=80, n_per_attack=15)
    assert r["security_value"] <= r["best_pure_value"] + 1e-9
    assert abs(sum(r["defender_mix"]) - 1) < 1e-6
    assert all(f["fpr"] <= 0.05 + 1e-9 for f in r["report"].values())
