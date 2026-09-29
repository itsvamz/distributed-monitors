import random

from src.monitors import AnomalyScorerNode, HeuristicJudge, ProvenanceMonitorNode, RuleMatcherNode
from src.traces import (
    make_clean_run,
    make_delayed_memory_run,
    make_judge_injection_run,
    make_naive_attack_run,
    make_stealthy_attack_run,
)


def test_n1_fires_on_naive_attack_but_not_clean():
    rng = random.Random(0)
    clean = make_clean_run("c0", rng)
    naive = make_naive_attack_run("n0", rng)
    n1 = RuleMatcherNode()
    assert max(n1.score_run(clean)) == 0.0
    assert max(n1.score_run(naive)) == 1.0


def test_n1_structurally_blind_to_stealthy():
    rng = random.Random(0)
    stealthy = make_stealthy_attack_run("s0", rng)
    n1 = RuleMatcherNode()
    assert max(n1.score_run(stealthy)) == 0.0  # by design: no dangerous tool name used


def test_n2_scores_harm_step_higher_than_clean_baseline():
    rng = random.Random(1)
    clean_runs = [make_clean_run(f"c{i}", rng) for i in range(20)]
    naive = make_naive_attack_run("n0", rng)
    n2 = AnomalyScorerNode()
    n2.fit(clean_runs)
    scores = n2.score_run(naive)
    assert scores[naive.harm_step] > 1.0  # clearly above baseline z^2 ~ 1


def test_n3_heuristic_is_suppressed_by_injection():
    rng = random.Random(2)
    naive = make_naive_attack_run("n0", rng)
    injected = make_judge_injection_run("j0", rng)
    n3 = HeuristicJudge()
    naive_score_at_harm = n3.score_run(naive)[naive.harm_step]
    injected_score_at_harm = n3.score_run(injected)[injected.harm_step]
    # Same underlying violation-ish action, but injection text should
    # suppress the judge's suspicion score.
    assert injected_score_at_harm < naive_score_at_harm


def test_n4_flags_write_then_read_of_untrusted_memory():
    rng = random.Random(3)
    run = make_delayed_memory_run("m0", rng)
    n4 = ProvenanceMonitorNode()
    scores = n4.score_run(run)
    assert scores[run.onset_step] == 1.0  # the poisoned write
    read_step = next(i for i, s in enumerate(run.steps) if s.memory_event and s.memory_event.op == "read")
    assert scores[read_step] == 1.0  # the retrieval of the flagged entry


def test_n4_blind_to_pure_behavioural_attack():
    rng = random.Random(4)
    naive = make_naive_attack_run("n0", rng)
    n4 = ProvenanceMonitorNode()
    assert max(n4.score_run(naive)) == 0.0


def test_cusum_accumulates_sustained_drift_and_forgets_a_spike():
    import random
    from src.monitors import AnomalyScorerNode, CusumNode
    from src.traces import Run, Step
    rng = random.Random(5)
    clean = [make_clean_run(f"c{i}", rng) for i in range(30)]
    n2, cu = AnomalyScorerNode(), CusumNode(k=1.15)
    n2.fit(clean); cu.fit(clean)

    def run_with(vals):
        steps = [Step(i, "read_file", "", "ok", [v] * 8) for i, v in enumerate(vals)]
        return Run("x", "t", steps, "harmful", "t", 0, len(vals) - 1)

    ramp = run_with([1.3] * 6)
    assert max(cu.score_run(ramp)) > 1.5 * max(n2.score_run(ramp))      # piles up
    spike = cu.score_run(run_with([0, 0, 0, 3, 0, 0, 0, 0]))
    assert spike[-1] < 0.5 * max(spike)                                  # decays


def test_low_and_slow_is_invisible_to_n1_n3_n4():
    from src.traces import make_low_and_slow_run
    rng = random.Random(6)
    r = make_low_and_slow_run("l0", rng)
    assert max(RuleMatcherNode().score_run(r)) == 0
    assert max(HeuristicJudge().score_run(r)) == 0
    assert max(ProvenanceMonitorNode().score_run(r)) == 0
