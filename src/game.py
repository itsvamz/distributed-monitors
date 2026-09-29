"""game.py — E5: build the payoff table from measured data and solve it.

The defender picks a fusion rule (row), the attacker picks an attack style,
possibly with k retries (column). Payoff[i][j] = attacker's empirical
success rate for that pairing, taken directly from simulated runs (E3/E4)
-- never guessed. Because attacker-success = 1 - defender-success on the
same runs, this is a zero-sum game, so "defender commits to a (possibly
mixed) strategy first, attacker best-responds" (a Stackelberg game) has the
same value as the maximin/minimax solution of the matrix game. We solve it
exactly with linear programming (scipy.optimize.linprog), which is exact
for zero-sum matrix games of this size (a handful of rows/columns) --
no need for iterative approximation methods here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve_zero_sum_game(payoff: np.ndarray) -> dict:
    """payoff[i, j] = attacker's success rate when defender plays row i and
    attacker plays column j (lower is better for the defender). Returns the
    defender's optimal mixed strategy, the game value (= attacker's success
    rate at equilibrium, i.e. the "security value"), and the attacker's
    best-response distribution.

    Solved as: minimize v subject to, for every attacker column j,
    sum_i x_i * payoff[i, j] <= v, sum_i x_i = 1, x_i >= 0.
    This is the standard LP formulation of a matrix-game maximin value.
    """
    n_rows, n_cols = payoff.shape
    # Variables: [x_1..x_n_rows, v]
    n_vars = n_rows + 1
    c = np.zeros(n_vars)
    c[-1] = 1.0  # minimize v

    # For each attacker column j: sum_i x_i * payoff[i,j] - v <= 0
    A_ub = np.zeros((n_cols, n_vars))
    for j in range(n_cols):
        A_ub[j, :n_rows] = payoff[:, j]
        A_ub[j, -1] = -1.0
    b_ub = np.zeros(n_cols)

    # sum_i x_i = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_rows] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, 1) for _ in range(n_rows)] + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")

    defender_mix = res.x[:n_rows]
    game_value = res.x[-1]

    # Attacker's best response(s): columns whose payoff under defender_mix
    # equals the game value (the "active" columns at equilibrium).
    col_payoffs = defender_mix @ payoff
    best_response_cols = [j for j in range(n_cols) if abs(col_payoffs[j] - game_value) < 1e-6]

    return {
        "defender_mix": defender_mix,
        "game_value": float(game_value),
        "col_payoffs": col_payoffs,
        "best_response_cols": best_response_cols,
    }


def best_pure_strategy_value(payoff: np.ndarray) -> tuple[int, float]:
    """The defender's best *fixed* (pure) rule and its worst-case attacker
    success rate: min over rows of (max over columns). Reported alongside
    the mixed-strategy equilibrium to show the gap the paper highlights --
    randomizing between rules beats committing to any single one."""
    worst_case_per_row = payoff.max(axis=1)
    best_row = int(np.argmin(worst_case_per_row))
    return best_row, float(worst_case_per_row[best_row])
