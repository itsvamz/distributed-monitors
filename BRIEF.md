# Project Brief: Distributed Monitors for Agent Safety

## Problem
AI agents act over many steps (think, call tools, write to memory). Two
things go unmeasured today: (1) **when** a safety problem is caught — a
detector that fires after the damage is done scores the same as one that
fires before it, on a normal pass/fail benchmark; and (2) whether a set of
monitors has **shared blind spots** — if the strongest monitor is itself an
LLM reading the same text the agent reasoned over, the same trick can fool
both at once.

## Approach
Four independent monitors watch different signals from the same agent run.
A fusion rule combines their per-step suspicion scores into one alarm.
Security is measured as a **game**: the defender picks a fusion rule (or a
mix of rules), the attacker picks the best counter-attack (including
retries), and the headline number is the attacker's win rate when both
play optimally — not the win rate against a fixed list of known attacks.

## The four monitors
| Node | Watches | Catches | Structurally blind to |
|---|---|---|---|
| N1 Rule matcher | tool calls/args | anything matching a known-dangerous pattern | anything not written as a rule |
| N2 Anomaly scorer | action embeddings | statistically unusual steps | "unusual but fine" vs "unusual and bad" |
| N3 LLM judge | reasoning text | intent/context a rule can't express | text engineered to convince a reader it's fine |
| N4 Provenance monitor | memory writes/reads | poisoned content at the moment it's planted or used | anything that isn't a memory operation |

## What gets built (features)
1. **Trace generator (E0)** — labeled clean + harmful runs, each with a
   step-level onset label and harm label. Four attack styles: naive,
   stealthy, judge-injection, delayed-memory.
2. **Monitor nodes (E1)** — N1–N4, each producing one suspicion score per
   step. N3 ships with an offline heuristic backend and a pluggable
   interface for a real LLM call.
3. **Threshold calibration** — every node's alarm threshold is picked so it
   alarms on at most a fixed % of clean runs, so no fusion rule wins by
   being trigger-happy.
4. **Shared blind-spot analysis (E2)** — for every pair of nodes, how often
   node B also misses an attack that fooled node A.
5. **Fusion rules (E3)** — best-single-node, any-node-OR (plain and with a CUSUM
   persistence node), majority-vote, and a cost-aware cascade. Each rule is tuned
   so its combined false-alarm rate stays under the same cap.
6. **Metrics** — detection delay, lead time, caught-in-time rate, false
   alarm rate, shared-blind-spot probability, memory-attack funnel
   (written → retrieved → acted-on), random-alarm baseline, retry curves
   (forgetful vs. suspicion-sharing across attempts).
7. **Game solver (E5)** — builds the payoff table (fusion rule × attack
   style) from measured data and solves the zero-sum matrix game exactly
   via linear programming, returning the defender's optimal mixed strategy
   and the equilibrium **security value**.
8. **Reporting** — one command runs E0→E5 over 30 seeds with bootstrap 95%
   confidence intervals, writes a JSON report, and regenerates every figure.

## Technical architecture
```
traces.py            synthetic agent-run generator (swap for real harness later)
monitors/
  n1_rule_matcher.py  n2_anomaly_scorer.py  n3_llm_judge.py  n4_provenance_monitor.py
calibration.py        per-node thresholds at a fixed false-alarm cap
fusion.py             4 fusion rules over per-node scores
metrics.py            every timing/security metric in the design doc
game.py               zero-sum game builder + LP equilibrium solver (scipy)
pipeline.py           orchestrates E0 -> E1 -> E2 -> E3 -> E5
experiments/          one script per stage + run_all.py (full pipeline)
tests/                unit tests per module
docs/labeling_guide.md   onset/harm labeling protocol with worked examples
```
Stack: Python, numpy, scipy (`linprog` for the LP, exact for this game
size). No network or paid API calls required to run the full pipeline —
N3's default backend is a documented heuristic stand-in; a real model call
is a one-file swap (`RealLLMJudge`, see `n3_llm_judge.py`).

## Explicitly out of scope (per design doc)
Alignment training (changing the model itself) and formal verification.
This is monitoring from the outside only.

## Honest limitations of this build
- Traces are synthetic, not from a real agent harness — the pipeline's
  logic and metrics are real and tested; the *numbers* it currently
  produces are illustrative, not empirical findings about real agents.
- N3 defaults to a keyword heuristic, not a real LLM call, so today's
  "judge injection" results demonstrate the *mechanism* the paper is
  worried about, not a measurement of whether a real model is fooled.
- Next real step: replace `traces.generate_dataset()` with real harness
  output (e.g. AgentDojo-style tasks) and swap in `RealLLMJudge`.
