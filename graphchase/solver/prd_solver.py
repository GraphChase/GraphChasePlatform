"""Projected Replicator Dynamics Algorithm (copied for graphchase PSRO use)."""

from __future__ import annotations

import numpy as np


def _partial_multi_dot(player_payoff_tensor, strategies, index_avoided):
    new_axis_order = [index_avoided] + [i for i in range(len(strategies)) if (i != index_avoided)]
    accumulator = np.transpose(player_payoff_tensor, new_axis_order)
    for i in range(len(strategies) - 1, -1, -1):
        if i != index_avoided:
            accumulator = np.dot(accumulator, strategies[i])
    return accumulator


def _simplex_projection(updated_strategy, gamma=0.0):
    n = len(updated_strategy)
    idx = np.arange(1, n + 1)
    u = np.sort(updated_strategy)[::-1]
    u_tmp = (1 - np.cumsum(u) - (n - idx) * gamma) / idx
    rho = np.searchsorted(u + u_tmp <= gamma, True)
    return np.maximum(updated_strategy + u_tmp[rho - 1], gamma)


def _projected_replicator_dynamics_step(payoff_tensors, strategies, dt, gamma):
    new_strategies = []
    for player in range(len(payoff_tensors)):
        current_payoff_tensor = payoff_tensors[player]
        current_strategy = strategies[player]

        values_per_strategy = _partial_multi_dot(current_payoff_tensor, strategies, player)
        average_return = np.dot(values_per_strategy, current_strategy)
        delta = current_strategy * (values_per_strategy - average_return)

        updated_strategy = current_strategy + dt * delta
        updated_strategy = _simplex_projection(updated_strategy, gamma)
        new_strategies.append(updated_strategy)
    return new_strategies


def projected_replicator_dynamics(
    payoff_tensors,
    prd_initial_strategies=None,
    prd_iterations=int(1e5),
    prd_dt=1e-3,
    prd_gamma=1e-6,
    average_over_last_n_strategies=None,
    **unused_kwargs,
):
    number_players = len(payoff_tensors)
    action_space_shapes = payoff_tensors[0].shape

    new_strategies = prd_initial_strategies or [
        np.ones(action_space_shapes[k]) / action_space_shapes[k] for k in range(number_players)
    ]

    average_over_last_n_strategies = average_over_last_n_strategies or prd_iterations
    meta_strategy_window = []
    for i in range(prd_iterations):
        new_strategies = _projected_replicator_dynamics_step(payoff_tensors, new_strategies, prd_dt, prd_gamma)
        if i >= prd_iterations - average_over_last_n_strategies:
            meta_strategy_window.append(new_strategies)
    average_new_strategies = np.mean(meta_strategy_window, axis=0)
    return average_new_strategies
