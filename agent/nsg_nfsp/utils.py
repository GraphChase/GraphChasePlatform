
from itertools import product
import torch
import numpy as np
import scipy.stats
import scipy as sp
from collections import namedtuple

Transition = namedtuple('Transition', ('obs', 'action',
                                       'reward', 'next_obs', 'is_end')
                        )
Sample = namedtuple('Sample', ('obs', 'action'))

def query_legal_actions(player_idx, obs, Map, num_defender=None):
    attacker_history, defender_position = obs
    if player_idx == 0:
        if num_defender == 1:
            defender_legal_actions = Map[defender_position]
        else:
            assert num_defender > 1
            before_combination = []
            for i in range(len(defender_position)):
                before_combination.append(Map[defender_position[i]])
            defender_legal_actions = list(product(*before_combination))
        return defender_legal_actions
    else:
        assert num_defender == None
        attacker_legal_actions = Map[attacker_history[-1]]
        return attacker_legal_actions
    
def evaluate(environment, Defender, Attacker, br_idx, nb_episodes=100, all_results=False):
    """
    Evaluate the performance of agents

    Parameters:
    - br_idx: 0 means return defender reward, 1 for attacker
    - nb_episodes: total episodes of simulations

    """
    
    assert br_idx in (
        0, 1), 'pls input proper best responsor idx, 0 for defender, 1 for attacker.'
    with torch.no_grad():
        total_return = []
        for ep in range(nb_episodes):
            current_return = evaluate_episode(
                environment, Defender, Attacker, br_idx)
            total_return.append(max(current_return, 0.))
        total_return = np.array(total_return)
        avg_return = np.mean(total_return)
        std_return = np.std(total_return)
        if all_results:
            se = scipy.stats.sem(total_return)
            h = se * sp.stats.t._ppf((1 + 0.95) / 2., len(total_return) - 1)
            return (avg_return, h), total_return
        return avg_return, std_return

def evaluate_episode(env, Defender, Attacker, br_idx):
    observation, info = env.reset()
    terminated = False
    Defender.reset()
    Attacker.reset()
    
    current_return = 0.
    while not terminated:
        defender_obs = [info["evader_history"], tuple(info["defender_history"][-1])]
        attacker_obs = [info["evader_history"], tuple(info["defender_history"][0])]
        
        def_current_legal_action = list(product(*info["defender_legal_actions"]))
        att_current_legal_action = info["evader_legal_actions"]

        defender_a = Defender.select_action(
            [defender_obs], [def_current_legal_action], is_evaluation=True)
        attacker_a = Attacker.select_action(
            [attacker_obs], [att_current_legal_action], is_evaluation=True)
        
        env_action = np.array(list(attacker_a + defender_a[0])) # shape: (defender_num +1, )
        observation, reward, terminated, truncated, info = env.step(env_action) 

        def_reward = reward
        att_reward = -reward

        if br_idx == 0:
            current_return += def_reward
        else:
            current_return += att_reward
    return current_return

def decay_expl(expl_rate, episode, decay_length):
    start = expl_rate
    end = expl_rate*0.1
    expl = end+(start-end)*(1-episode/decay_length)
    return expl