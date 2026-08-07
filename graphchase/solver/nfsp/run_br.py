from __future__ import annotations

import numpy as np
import torch


def evaluate(environment, Defender, Attacker, br_idx, nb_episodes=100, all_results=False):
    assert br_idx in (0, 1), "pls input proper best responsor idx, 0 for defender, 1 for attacker."
    with torch.no_grad():
        total_return = []
        for _ in range(nb_episodes):
            current_return = evaluate_episode(environment, Defender, Attacker, br_idx)
            total_return.append(current_return)
        total_return = np.array(total_return)
        avg_return = np.mean(total_return)
        std_return = np.std(total_return)
        if all_results:
            import scipy.stats
            import scipy as sp

            se = scipy.stats.sem(total_return)
            h = se * sp.stats.t._ppf((1 + 0.95) / 2.0, len(total_return) - 1)
            return (avg_return, h), total_return
        return avg_return, std_return


def evaluate_episode(environment, Defender, Attacker, br_idx):
    game_state = environment.reset()
    Defender.reset()
    Attacker.reset()
    current_return = 0.0
    while not game_state.is_end():
        defender_obs, attacker_obs = game_state.obs()
        def_current_legal_action, att_current_legal_action = game_state.legal_action()

        defender_a = Defender.select_action([defender_obs], [def_current_legal_action], is_evaluation=True)
        attacker_a = Attacker.select_action([attacker_obs], [att_current_legal_action], is_evaluation=True)
        game_state = environment.simu_step(defender_a, attacker_a)
        def_reward, att_reward = game_state.reward(is_evaluation=True)
        if br_idx == 0:
            current_return += def_reward
        else:
            current_return += att_reward
    return current_return
