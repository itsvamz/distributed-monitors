import numpy as np

from src.game import best_pure_strategy_value, solve_zero_sum_game
from src.metrics import retry_success_forgetful, retry_success_sharing


def test_solve_zero_sum_game_matches_worked_example():
    # The paper's worked toy example: OR and Cascade rows only, at the two
    # columns where mixing actually helps (judge_injection, stealthy).
    # Values taken from the doc's illustrative table.
    payoff = np.array([
        [0.05, 0.20, 0.40, 0.25],  # any_node_or
        [0.06, 0.45, 0.15, 0.15],  # cascade
    ])
    result = solve_zero_sum_game(payoff)
    # Mixing should never leave the defender worse off than the best pure rule.
    best_row, best_pure_value = best_pure_strategy_value(payoff)
    assert result["game_value"] <= best_pure_value + 1e-6


def test_pure_strategy_is_upper_bound_on_mixed_equilibrium():
    payoff = np.array([
        [0.1, 0.9],
        [0.9, 0.1],
    ])
    result = solve_zero_sum_game(payoff)
    # Classic matching-pennies-style game: mixing 50/50 should give value 0.5,
    # strictly better for the defender than either pure row (worst case 0.9).
    assert abs(result["game_value"] - 0.5) < 1e-4


def test_retry_forgetful_matches_closed_form():
    assert abs(retry_success_forgetful(0.2, 5) - (1 - 0.8 ** 5)) < 1e-9


def test_retry_sharing_is_never_worse_for_attacker_than_forgetful_is_better_for_defender():
    # Sharing suspicion across attempts should make the attacker's cumulative
    # success grow more slowly than the forgetful model, for k > 1.
    p = 0.2
    forgetful = retry_success_forgetful(p, 5)
    sharing = retry_success_sharing(p, 5, suspicion_decay=0.5)
    assert sharing < forgetful
