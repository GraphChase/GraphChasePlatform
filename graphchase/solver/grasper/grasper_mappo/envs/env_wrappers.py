import copy
import numpy as np
from .env import RL_Env


class Env_list(object):
    def __init__(self, mappo_args, num_env, attacker_strategy=None, attacker_strategy_type="mix", action_type="all_path"):
        """
        environments: list of environments to run in subprocesses
        """
        self.mappo_args = mappo_args

        if attacker_strategy is not None:
            self.env_list = [
                RL_Env(
                    copy.deepcopy(mappo_args.game),
                    attacker_strategy[i],
                    attacker_strategy_type=attacker_strategy_type,
                    action_type=action_type,
                )
                for i in range(num_env)
            ]
        else:
            self.env_list = [
                RL_Env(
                    copy.deepcopy(mappo_args.game),
                    attacker_strategy_type=attacker_strategy_type,
                    action_type=action_type,
                )
                for _ in range(num_env)
            ]

        self.num_envs = mappo_args.n_rollout_threads
        self.num_agent = self.env_list[0].agent_num
        self.obs_dim = self.env_list[0].obs_dim
        self.action_dim = self.env_list[0].action_dim
        self.share_obs_dim = self.env_list[0].share_obs_dim
        self.time_horizon = self.env_list[0].time_horizon

    def initialize_attacker_strategy(self):
        for i, env in enumerate(self.env_list):
            if i % self.mappo_args.num_sample == 0:
                env.initialize_attacker_strategy()
            else:
                env.initialize_attacker_strategy(self.env_list[i-1].attacker_strategy)

    def step(self, actions):
        results = []
        for i, env in enumerate(self.env_list):
            results.append(env.step(actions[i]))
        share_obs, obs, rewards, dones, time_steps = zip(*results)
        return np.stack(share_obs), np.stack(obs), np.stack(rewards), np.stack(dones), time_steps

    def reset(self):
        results = [env.reset() for env in self.env_list]
        share_obs, obs, time_steps = zip(*results)
        return np.stack(share_obs), np.stack(obs), time_steps
