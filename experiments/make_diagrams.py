"""Draw every conceptual diagram used in the paper -> docs/figures/*.png

    python -m experiments.make_diagrams
"""
import matplotlib.pyplot as plt

from experiments.diagram_kit import PAL, arrow, box, canvas, diamond, save


def fig01_timeline():
    fig, ax = canvas(2.35)              # y 0..67
    rows = {"a": 50, "b": 18}
    ax.text(1, rows["a"] + 13.5, "(a) Grading only the end of the run", fontsize=7, fontweight="bold", ha="left")
    ax.text(1, rows["b"] + 13.5, "(b) Timing-aware grading", fontsize=7, fontweight="bold", ha="left")
    for y in rows.values():
        ax.plot([8, 96], [y, y], color="black", lw=0.7)
        for i in range(10):
            ax.plot([12 + i * 9, 12 + i * 9], [y - 1, y + 1], color="black", lw=0.5)
        ax.text(6.5, y, "step", fontsize=5.5, ha="right", va="center")
        ax.plot(28, y, marker="^", ms=5, color="#B58900", mec="black", mew=0.4)
        ax.text(28, y - 4, "onset", ha="center", va="top", fontsize=5.8)
        ax.plot(82, y, marker="X", ms=6, color="#C0392B", mec="black", mew=0.4)
        ax.text(82, y - 4, "harm executes", ha="center", va="top", fontsize=5.8)
    ya, yb = rows["a"], rows["b"]
    ax.plot(82, ya + 3.2, marker="*", ms=7, color="#2E7D32", mec="black", mew=0.4)
    ax.text(50, ya + 5.5, "alarm fires on the harm step: lead time = 0\n(end-of-run pass/fail still counts it as caught)",
            ha="center", va="bottom", fontsize=5.8)
    ax.plot(55, yb, marker="*", ms=7, color="#2E7D32", mec="black", mew=0.4)
    ax.text(55, yb - 4, "alarm", ha="center", va="top", fontsize=5.8)
    arrow(ax, (28, yb + 5), (55, yb + 5), style="<|-|>", ls="--")
    ax.text(41.5, yb + 7.6, "detection delay", ha="center", fontsize=5.6)
    arrow(ax, (55, yb + 5), (82, yb + 5), style="<|-|>")
    ax.text(68.5, yb + 7.6, "lead time > 0", ha="center", fontsize=5.6)
    save(fig, "fig01_timeline")


def fig02_problem_map():
    fig, ax = canvas(2.9)               # y 0..82.9
    cols = [16, 50, 84]
    ttl = ["Blind spot 1\nonly the end\nis graded", "Blind spot 2\nmemory is\ntrusted", "Blind spot 3\nthe monitor itself\nis trusted"]
    rq = ["RQ1 timing:\ndo traces reveal\ndrift early?", "RQ3 memory:\ndoes provenance\ncatch what\nbehaviour misses?",
          "RQ2 shared blind\nspots: do monitors\nfail together?"]
    ex = ["E1 per-node\nscores + lead time", "E4 memory testbed\n+ funnel metric", "E2 miss\ncorrelation"]
    for x, t, r, e in zip(cols, ttl, rq, ex):
        box(ax, x, 72, 29, 15, t, "attack", bold=True)
        arrow(ax, (x, 64.3), (x, 58.2))
        box(ax, x, 50, 29, 15, r, "node")
        arrow(ax, (x, 42.3), (x, 36.2))
        box(ax, x, 29, 29, 12, e, "data")
    box(ax, 50, 9, 92, 12, "RQ4 security under attack:\nequilibrium value of the fusion-rule game (E3, E5)", "good", bold=True)
    arrow(ax, (16, 22.8), (30, 15.2))
    arrow(ax, (50, 22.8), (50, 15.2))
    arrow(ax, (84, 22.8), (70, 15.2))
    save(fig, "fig02_problem_map")


def fig03_architecture():
    fig, ax = canvas(3.15)              # y 0..90
    box(ax, 50, 82, 64, 12, "Agent run (one or more sessions)\ntool calls | reasoning text | memory reads/writes", "white", bold=True)
    nodes = [("N1\nRule\nmatcher", "tool calls"), ("N2 / N2c\nAnomaly /\nCUSUM", "embeddings"),
             ("N3\nLLM\njudge", "reasoning"), ("N4\nProvenance\nmonitor", "memory ops")]
    xs = [13, 38, 63, 88]
    for x, (n, s) in zip(xs, nodes):
        arrow(ax, (50 + (x - 50) * 0.3, 75.9), (x, 63.3))
        box(ax, x, 56, 21, 14, n, "node")
        ax.text(x, 46.8, s, ha="center", fontsize=5.6, style="italic")
        arrow(ax, (x, 49), (x, 42))
    box(ax, 50, 37, 92, 9, "per-step suspicion scores (saved once, reused by every rule)", "data")
    arrow(ax, (42, 32.5), (42, 27))
    box(ax, 42, 22, 56, 9, "Fusion rule (best single | OR | majority | cascade)", "rule", bold=True, fs=6.1)
    box(ax, 86, 22, 24, 9, "thresholds\n(clean runs only)", "white", fs=5.8)
    arrow(ax, (74, 22), (70.2, 22))
    arrow(ax, (36, 17.5), (27, 11.5))
    arrow(ax, (48, 17.5), (72, 11.5))
    box(ax, 27, 6.5, 40, 9, "alarm at step t", "attack", bold=True)
    box(ax, 73, 6.5, 40, 9, "continue", "good", bold=True)
    save(fig, "fig03_architecture")


def fig04_game_flow():
    fig, ax = canvas(2.9)               # y 0..82.9
    box(ax, 24, 72, 42, 14, "Defender strategies\n(fusion rules, thresholds\nfixed at the FPR cap)", "node")
    box(ax, 76, 72, 42, 14, "Attacker strategies\n(attack styles;\nretries k = 1, 2, 5, 10)", "attack")
    arrow(ax, (24, 64.9), (42, 54.5))
    arrow(ax, (76, 64.9), (58, 54.5))
    box(ax, 50, 50, 46, 9, "simulate labelled runs per (rule, style)", "data")
    arrow(ax, (50, 45.4), (50, 40.5))
    box(ax, 50, 36, 52, 9, "payoff table P[i,j] = attacker success", "white", bold=True)
    arrow(ax, (50, 31.4), (50, 26.5))
    box(ax, 50, 22, 62, 9, "zero-sum LP:  min v  s.t.  x^T P <= v", "rule")
    arrow(ax, (50, 17.4), (50, 12.5))
    box(ax, 50, 8, 74, 8.5, "security value v + defender mix x + attacker best reply", "good", bold=True)
    save(fig, "fig04_game_flow")


def fig05_cascade():
    fig, ax = canvas(3.3)               # y 0..94.3
    box(ax, 30, 88, 36, 8, "next step t of the run", "white", bold=True)
    arrow(ax, (30, 84), (30, 79))
    diamond(ax, 30, 71, 44, 15, "N1 rule hit or\nN4 flagged memory?")
    arrow(ax, (52, 71), (70, 71), "yes", dy=2.2)
    box(ax, 82, 71, 24, 8, "ALARM", "attack", bold=True)
    arrow(ax, (30, 63.5), (30, 57), "no", dx=4, dy=0)
    diamond(ax, 30, 49, 44, 15, "N2 score above\nrelaxed call level?")
    arrow(ax, (52, 49), (70, 49), "yes", dy=2.2)
    box(ax, 82, 49, 26, 9, "call costly\nN3 judge", "node")
    arrow(ax, (82, 44.5), (82, 37))
    diamond(ax, 82, 29, 34, 15, "N3 score\nabove threshold?")
    arrow(ax, (99, 29), (99, 71), style="-", lw=0.0)
    ax.plot([99, 99], [29, 67], color="black", lw=0.6)
    arrow(ax, (99, 67), (94.2, 67), style="-|>")
    ax.text(96.5, 47, "yes", fontsize=6, style="italic", rotation=90)
    arrow(ax, (30, 41.5), (30, 22), "no", dx=4, dy=0)
    arrow(ax, (65, 29), (30, 22.5), "no", tpos=0.35, dy=2)
    box(ax, 30, 16, 36, 8, "no alarm: go to t+1", "good", bold=True)
    ax.text(50, 4, "cost: N3 is called only on screened steps", ha="center", fontsize=6, style="italic")
    save(fig, "fig05_cascade")


def fig06_data_pipeline():
    fig, ax = canvas(3.3)               # y 0..94.3
    box(ax, 50, 88, 80, 8, "Trace generator  (5 attack styles + clean runs, seeded)", "attack", bold=True)
    arrow(ax, (50, 84), (50, 79))
    box(ax, 50, 74, 80, 8, "Run / Step / MemoryEvent schema + onset & harm labels", "data")
    arrow(ax, (38, 70), (26, 63))
    arrow(ax, (62, 70), (74, 63))
    box(ax, 26, 58, 40, 9, "clean runs (200)\nfit N2, N2c; calibrate", "good")
    box(ax, 74, 58, 40, 9, "harmful runs (5 x 40)\nnever used for fitting", "attack")
    arrow(ax, (26, 53.4), (40, 46))
    arrow(ax, (74, 53.4), (60, 46))
    box(ax, 50, 41, 60, 9, "score every step with N1, N2, N2c, N3, N4", "node")
    arrow(ax, (50, 36.4), (50, 31.5))
    box(ax, 50, 27, 60, 9, "calibrate per rule to FPR <= 5% (clean only)", "rule")
    arrow(ax, (50, 22.4), (50, 17.5))
    box(ax, 50, 13, 66, 9, "evaluate 30 seeds -> payoff table -> LP", "white", bold=True)
    save(fig, "fig06_data_pipeline")


def fig07_memory_attack():
    fig, ax = canvas(2.2)               # y 0..62.9
    ax.add_patch(plt.Rectangle((3, 8), 42, 44, fc="#F4F4F4", ec="black", lw=0.6))
    ax.add_patch(plt.Rectangle((50, 8), 47, 44, fc="#F4F4F4", ec="black", lw=0.6))
    ax.text(24, 55, "Session 0", ha="center", fontsize=7, fontweight="bold")
    ax.text(73.5, 55, "Session 1 (later)", ha="center", fontsize=7, fontweight="bold")
    box(ax, 24, 42, 34, 8, "ordinary steps 0-3", "white", fs=6)
    box(ax, 24, 29, 36, 10, "step 4: write note\nsource = web (untrusted)", "attack", fs=6)
    arrow(ax, (24, 38), (24, 34.2))
    ax.text(24, 18, "N4 flags the write", ha="center", fontsize=6.2, fontweight="bold", color="#1B5E20")
    box(ax, 73.5, 42, 38, 8, "ordinary steps 5-7", "white", fs=6)
    box(ax, 73.5, 31, 40, 8, "step 8: read the note", "warn", fs=6)
    box(ax, 73.5, 20, 40, 8, "step 9: act on it (harm)", "attack", fs=6)
    arrow(ax, (73.5, 38), (73.5, 35))
    arrow(ax, (73.5, 27), (73.5, 24))
    arrow(ax, (42, 29), (53.5, 31), ls="--")
    ax.text(47.5, 4, "the note persists in memory across sessions", ha="center", fontsize=5.8, style="italic")
    ax.text(73.5, 11, "N1-N3 see only ordinary behaviour", ha="center", fontsize=5.8, style="italic")
    save(fig, "fig07_memory_attack")


def fig15_hypotheses(verdicts):
    fig, ax = canvas(2.7)               # y 0..77
    rows = [("H1", "equilibrium security is\nfar below fixed-attack\ndetection rate", verdicts[0]),
            ("H2", "diverse monitors beat\none strong (judge)\nmonitor", verdicts[1]),
            ("H3", "N4 recovers memory-\nattack security lost by\nN1-N3", verdicts[2]),
            ("H4", "sharing alarms across\nretries removes most\nretry advantage", verdicts[3])]
    for i, (h, t, v) in enumerate(rows):
        y = 68 - i * 18
        box(ax, 8, y, 10, 11, h, "node", bold=True)
        box(ax, 42, y, 46, 13, t, "white", fs=6.2)
        arrow(ax, (65.5, y), (71, y))
        box(ax, 85, y, 25, 13, v[0], v[1], fs=6, bold=True)
    save(fig, "fig15_hypotheses")


def fig16_roadmap():
    fig, ax = canvas(2.75)              # y 0..78.6
    steps = [("Now", "synthetic traces,\nheuristic judge", "data"),
             ("Next", "real transcripts\n(AgentDojo-style)", "node"),
             ("Then", "real LLM judge via\nRealLLMJudge", "node"),
             ("Then", "richer rules and\nattack styles", "rule"),
             ("Finally", "human onset/harm labels\n+ live pilot", "good")]
    for i, (a, b, c) in enumerate(steps):
        y = 68 - i * 15
        box(ax, 14, y, 20, 9.5, a, c, bold=True)
        box(ax, 60, y, 60, 11, b, "white")
        arrow(ax, (24, y), (30, y))
        if i < len(steps) - 1:
            arrow(ax, (14, y - 4.9), (14, y - 10.1))
    save(fig, "fig16_roadmap")


if __name__ == "__main__":
    fig01_timeline(); fig02_problem_map(); fig03_architecture(); fig04_game_flow()
    fig05_cascade(); fig06_data_pipeline(); fig07_memory_attack(); fig16_roadmap()
    print("diagrams written to docs/figures/")
