import torch
import os
import time
import numpy as np
from agent.pretrain_psro.ppo_defender_policy import PPOAgent
from agent.pretrain_psro.path_evader_runner import PathEvaderRunner
from agent.pretrain_psro.node_embedding import NodeEmbedding
from agent.base_runner import BaseRunner


class PretrainPsroDefenderRunner(BaseRunner):
    """
    Define a runner for defender
    _obs_wrapper: tranform env.observation_space to embeddings and add time information
    get_action: input env.observation_space and time_step, return defenders' action e.g. [Up Down ,...]
    """
    def __init__(self, env, 
                 policy: PPOAgent, 
                 args, 
                 node_model: NodeEmbedding=None):
        
        self.env = env
        self.policy = policy
        self.args = args
        self.node_model = node_model
        self.save_path = args.save_path       

        self.get_graph_model()

        if self.args.load_defender_model:
            actor_path = os.path.join(self.args.load_defender_model, f"{self.args.pretrain_model_checkpoint}_actor.pt")
            critic_path = os.path.join(self.args.load_defender_model, f"{self.args.pretrain_model_checkpoint}_critic.pt")
            print(f"load defender model from {self.args.load_defender_model}, checkpoint is {self.args.pretrain_model_checkpoint}")
            self.policy.actor.load_state_dict(torch.load(actor_path))
            self.policy.critic.load_state_dict(torch.load(critic_path))

    def _obs_wrapper(self, obs: np.ndarray, time_step=None) -> np.ndarray:
        obs_list = []
        for i in range(len(obs)):
            obs_list.extend(self.n2v[obs[i]])
        if time_step is not None:
            obs_list.append(float(time_step))
        return np.array(obs_list)
        
    def get_action(self, obs: np.ndarray, time_step: int) -> tuple[list, float]:
        obs = self._obs_wrapper(obs, time_step)
        action_idx, log_prob = self.policy.select_action(obs)
        action = self.policy.defender_action_map[action_idx.item()]
        return action, log_prob, obs, action_idx
    
    def get_env_actions(self, obs: np.ndarray, time_step: int) -> list:
        action, log_prob, obs, action_idx = self.get_action(obs, time_step)
        return action

    def train(self, evader_policy_list: list, meta_strategy: np.ndarray,):
        evader_strategy_num = len(evader_policy_list[0].path_selections)
        strategy = np.zeros((len(evader_policy_list), evader_strategy_num))

        for i, runner in enumerate(evader_policy_list):
            strategy[i] = runner._get_strategy()
        strategy = (strategy * meta_strategy[:, np.newaxis]).sum(axis=0)
        strategy /= np.sum(strategy)

        for i in range(self.args.train_pursuer_number):

            # sample one opponent path according to strategy
            if self.env.nextstate_as_action:
                _, evader_actions = evader_policy_list[0].get_action(strategy) 
                evader_actions = evader_actions[1:]
            else:
                evader_actions, _ = evader_policy_list[0].get_action(strategy)            

            # sample to collect data, rollout
            terminated = False
            observation, info = self.env.reset()
            t = 0

            while not terminated:
                evader_act = evader_actions[t]
                
                defender_action, log_prob, obs_emb, action_idx = self.get_action(observation, t)
                actions = np.array(defender_action) # (n,)

                if self.env.nextstate_as_action:
                    for i in range(1, len(observation)):
                        actions[i-1] = self.env.graph.change_state[observation[i] - 1][defender_action[i-1]]                
                
                actions = np.insert(actions, 0, evader_act) # shape: (n+1, )

                observation, reward, terminated, truncated, info = self.env.step(actions)
                t += 1

                self.policy.store_transition((obs_emb, action_idx, log_prob, reward[1]))      
            self.policy.train(self.args.ppo_epochs)

    def pretrain(self, evader_runner: PathEvaderRunner):
        time_list = []
        reward_list = []
        start_time = time.time()      

        for iteration in range(self.args.num_pretrain_iteration):
            iteration_reward = 0.0

            # Generate args.evader_oracle_size policies, each policy is the distribution of evader path
            strategy = evader_runner.random_generate_evader_oracle(self.args.evader_oracle_size)
            for evader_policy in strategy:
                eval_rewards = self.rollouts(self.env, evader_runner, self.args.rollout_evader_episodes, evader_policy)
                iteration_reward += eval_rewards[1]
                
            self.policy.train(self.args.ppo_epochs)

            print('\nIteration', iteration)
            adaptation_reward = iteration_reward / self.args.evader_oracle_size
            print('train_reward', adaptation_reward)

            end_time = time.time()
            time_list.append(end_time-start_time)
            reward_list.append(adaptation_reward)
            print(reward_list, time_list)

            if (iteration + 1) % self.args.save_interval == 0:
                defender_model_folder = os.path.join(self.save_path, "defender_pretrain_model")
                self.save(defender_model_folder, iteration+1)

    def get_graph_model(self,):
        """
        self.n2v: Get node2vector model
        """
        if self.args.use_node_embedding:
            assert self.node_model is not None, "Please use a node embedding model"
            self.n2v = self.node_model.train_embedding()
        else:
            self.n2v = None

    def rollouts(self, env, evader, eval_episodes, policy_dist):
        total_rewards = np.zeros((2,))

        for _ in range(eval_episodes):

            if env.nextstate_as_action:
                _, evader_actions = evader.get_action(policy_dist) 
                evader_actions = evader_actions[1:]
            else:
                evader_actions, _ = evader.get_action(policy_dist)

            terminated = False
            observation, info = env.reset()
            t = 0

            while not terminated:
                evader_act = evader_actions[t]
                
                defender_action, log_prob, obs_emb, action_idx = self.get_action(observation, t)
                if env.nextstate_as_action:
                    actions = np.array(defender_action) # (n,)
                    for i in range(1, len(observation)):
                        actions[i-1] = self.env.graph.change_state[observation[i] - 1][defender_action[i-1]]
                    actions = np.insert(actions, 0, evader_act) # shape: (n+1, )
                else:
                    actions = np.array(defender_action) # (n,)
                    actions = np.insert(actions, 0, evader_act) # shape: (n+1, )

                observation, reward, terminated, truncated, info = env.step(actions)
                t += 1

                self.policy.store_transition((obs_emb, action_idx, log_prob, reward[1]))

                if terminated or truncated:
                    total_rewards += np.array(reward)
        
        return total_rewards / eval_episodes
    
    def save(self, path, prefix):
        self.policy.save(path, prefix)