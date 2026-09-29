# Distributed Monitors for Agent Safety — Reference Implementation (v2)

Four independent monitors watch an AI agent. A fusion rule decides when to
raise one alarm. Security is measured as a **game**: the defender picks a
rule (or a mix of rules), the attacker picks the best attack style, and the
headline number is the attacker's success rate at equilibrium (the
*security value*), not a detection rate against a fixed attack list.

## Quick start
```bash
pip install -r requirements.txt
./run_everything.sh                 # tests + 30-seed experiment + all figures
# or step by step
python -m tests.run_tests
python -m experiments.run_all --seeds 30
python -m experiments.make_diagrams && python -m experiments.make_figures
```
Everything runs offline; no API key is needed.

## What is new in v2
| Change | Why |
|---|---|
| **5th attack style `low_and_slow`** | An adaptive reply to a persistence (CUSUM) defender; makes the game genuinely adaptive. |
| **`CusumNode` (N2c)** | Page's CUSUM over N2's score: sustained mild drift accumulates, single spikes decay. |
| **Rule-level calibration** (`calibration.calibrate_rule`) | Every fusion rule is tuned so its *combined* clean-run false-alarm rate is <= the cap. OR automatically gets stricter per-node thresholds than majority vote. |
| **Real cascade** | Direct nodes (N1, N4) alarm alone; N2 only screens; N3 must confirm. Reports N3 calls, so cost is measured. |
| **Auto-selected "best single node"** | Chosen by measured caught-in-time rate, not by list position. |
| **Consistent "caught in time"** | Node-level blind-spot analysis uses the same strict rule (alarm before harm) as fusion rules. |
| **Memory funnel under a rule** | written -> retrieved -> acted-on now drops when a rule alarms early; v1 was always 100%. |
| **30 seeds + bootstrap 95% CIs** | Every reported number has an uncertainty interval. |
| **Timing (AMOC-style) curves** | `pipeline.timing_curve`: caught-in-time and delay versus false-alarm rate. |
| **Figure generators** | `make_diagrams.py` (flowcharts) and `make_figures.py` (data plots) reproduce every paper figure. |
| **23 tests + plain runner** | `python -m tests.run_tests` (no pytest needed). |

## Layout
```
src/traces.py            Run/Step/MemoryEvent + generators for 5 attack styles
src/monitors/            n1 rule | n2 anomaly (+ CUSUM) | n3 judge (+RealLLMJudge) | n4 provenance
src/calibration.py       node- and rule-level threshold calibration
src/fusion.py            best-single, OR, majority, cascade
src/metrics.py           lead time, delay, funnel, blind spot, retries, bootstrap CI
src/game.py              exact zero-sum LP (scipy HiGHS)
src/pipeline.py          evaluate(seed), run_many(seeds), timing_curve()
experiments/             run_all, per-stage scripts, make_diagrams, make_figures
tests/                   23 tests; docs/labeling_guide.md; docs/figures/
```

## Experiment map
| Stage | Code |
|---|---|
| E0 label runs | `src/traces.py`, `docs/labeling_guide.md` |
| E1 per-node timing | `src/monitors/*`, `pipeline.score_all` |
| E2 shared blind spots | `pipeline.evaluate` -> `blind_spots`, `node_style_catch` |
| E3 fusion rules | `src/fusion.py`, `calibration.calibrate_rule` |
| E4 adaptive attacks + memory | attack generators, `memory_funnel_under_rule`, retry models |
| E5 solve the game | `src/game.py` |

## Plugging in real components
1. **Real traces:** replace `traces.generate_dataset()` with a loader that returns the same `Run/Step/MemoryEvent` objects (for example from AgentDojo transcripts). Nothing downstream changes.
2. **Real judge:** `RealLLMJudge(call_fn)` in `n3_llm_judge.py` accepts any `prompt -> text` function.

## Honest limits
Traces are synthetic and N3 defaults to a keyword heuristic, so the numbers
demonstrate that the machinery works and illustrate the predicted effects.
They are not measurements of any real agent. The equilibrium is only as
informative as the 5 rules x 5 styles included.
