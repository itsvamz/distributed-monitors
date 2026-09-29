from src.fusion import N1, N2, N3, N4, AnyNodeOR, Cascade, MajorityVote, SingleNode


def test_or_fires_on_first_node_over_threshold():
    s = {"A": [0, .9, 0], "B": [0, 0, .9]}
    assert AnyNodeOR(["A", "B"]).apply(s, {"A": .5, "B": .5}, 3).alarm_step == 1


def test_majority_needs_two_nodes():
    s = {"A": [.9, .9], "B": [0, .9], "C": [0, 0]}
    t = {"A": .5, "B": .5, "C": .5}
    assert MajorityVote(["A", "B", "C"], 2).apply(s, t, 2).alarm_step == 1
    s2 = {"A": [.9, .9], "B": [0, 0], "C": [0, 0]}
    assert MajorityVote(["A", "B", "C"], 2).apply(s2, t, 2).alarm_step is None


def test_single_node():
    assert SingleNode("A").apply({"A": [0, 0, 1]}, {"A": .5}, 3).alarm_step == 2


def test_cascade_direct_nodes_alarm_without_calling_n3():
    z = [0.0, 0.0, 0.0]
    s = {N1: [0, 1, 0], N4: z, N2: z, N3: z}
    thr = {N1: .5, N4: .5, N2: 2.0, N3: .5}
    r = Cascade().apply(s, thr, 3)
    assert r.alarm_step == 1 and r.n3_calls == 0


def test_cascade_needs_n3_to_confirm_a_screened_step():
    s = {N1: [0, 0, 0], N4: [0, 0, 0], N2: [0, 1.2, 0], N3: [0, 0, 0]}
    thr = {N1: .5, N4: .5, N2: 2.0, N3: .5}
    r = Cascade(margin=0.5).apply(s, thr, 3)      # 1.2 >= 0.5*2.0 -> N3 called
    assert r.alarm_step is None and r.n3_calls == 1
    s[N3] = [0, 0.9, 0]
    assert Cascade(margin=0.5).apply(s, thr, 3).alarm_step == 1


def test_cascade_never_calls_n3_when_n2_is_quiet():
    s = {N1: [0, 0], N4: [0, 0], N2: [0.1, 0.1], N3: [9, 9]}
    r = Cascade().apply(s, {N1: .5, N4: .5, N2: 2.0, N3: .5}, 2)
    assert r.alarm_step is None and r.n3_calls == 0
