import torch
import torch.nn.functional as F
import copy
import numpy as np
import time
import os
from agent.nsgzero.mcts_defender_policy import NsgzeroDefenderPolicy
from agent.base_runner import BaseRunner
from common_utils import time_left, time_str, arg_max
from common_utils import directory_config, store_args


class NsgzeroDefenderRunner(BaseRunner):
    def __init__(self, env, policy:NsgzeroDefenderPolicy, args):
        super().__init__()
        self.args = args
        self.env = env

        self.save_path = directory_config(self.args.save_path)
        store_args(self.args, self.save_path)        

        self.time_horizon = env.time_horizon
        self.max_act = env.graph.degree
        self.num_defender = env.defender_num

        self.policy = policy
    
    def train_execute_episode(self, evader_runner):

        self.trajectory = {}
        self.trajectory["defender_his"] = np.zeros([self.time_horizon+1, self.num_defender])
        self.trajectory["attacker_his"] = np.zeros([self.time_horizon+1, 1])
        self.trajectory["defender_his_idx"] = np.zeros([self.time_horizon+1, self.num_defender])
        self.trajectory["attacker_his_idx"] = np.zeros([self.time_horizon+1, 1])
        self.trajectory["defender_legal_act"] = np.zeros([self.time_horizon+1, self.num_defender, self.max_act])
        self.trajectory["attacker_legal_act"] = np.zeros([self.time_horizon+1, 1, self.max_act])
        self.trajectory["return"] = np.zeros([self.time_horizon+1, 1])
        self.trajectory["mask"] = np.zeros([self.time_horizon+1, 1])

        track = {"selected_exit": None, "evader_reward": None, "is_br": None} if evader_runner.require_update else None

        t = 0
        observation, info = self.env.reset()
        terminated = False
        self.reset()
        evader_runner.reset()

        padded_defender_legal_act = [single_legal_actions + [0] * (self.max_act - len(single_legal_actions)) for single_legal_actions in info["defender_legal_actions"]]
        padded_attacker_legal_act = info["evader_legal_actions"]+[0] * (self.max_act - len(info["evader_legal_actions"]))
        self.trajectory["defender_legal_act"][t] = np.array(copy.deepcopy(padded_defender_legal_act))
        self.trajectory["attacker_legal_act"][t] = np.array(copy.deepcopy(padded_attacker_legal_act))
        self.trajectory["mask"][t] = 1

        while not terminated:
            t += 1
            
            defender_obs = (info["evader_history"], info["defender_history"][-1])
            attacker_obs = (info["evader_history"], info["defender_history"][0])

            # defender action
            defender_act, defender_act_idx = self.policy.train_select_act(defender_obs, info["defender_legal_actions"], prior=False)
            
            # evader action
            attacker_act, attacker_act_idx = evader_runner.train_select_act(attacker_obs)

            env_action = np.array((attacker_act,) + defender_act, dtype=int) # shape: (defender_num +1, )

            self.trajectory["defender_his_idx"][t] = defender_act_idx
            self.trajectory["attacker_his_idx"][t] = attacker_act_idx
                
            padded_defender_legal_act = [single_legal_actions + [0] * (self.max_act - len(single_legal_actions)) for single_legal_actions in info["defender_legal_actions"]]
            padded_attacker_legal_act = info["evader_legal_actions"]+[0] * (self.max_act - len(info["evader_legal_actions"]))
            self.trajectory["defender_legal_act"][t] = np.array(copy.deepcopy(padded_defender_legal_act))
            self.trajectory["attacker_legal_act"][t] = np.array(copy.deepcopy(padded_attacker_legal_act))
            self.trajectory["mask"][t] = 1

            # Env steps
            observation, reward, terminated, truncated, info = self.env.step(env_action)

            if terminated:
                self.trajectory["defender_his"][:info["episode"]["l"]] = np.array(copy.deepcopy(info["defender_history"]))
                self.trajectory["attacker_his"][:info["episode"]["l"]] = np.array(copy.deepcopy(info["evader_history"])).reshape(-1, 1)
                self.trajectory["return"][:info["episode"]["l"]] = copy.deepcopy(info["episode"]["defender_r"])
                        
                if track is not None:
                    track["selected_exit"] = evader_runner.selected_exit
                    track["evader_reward"] = -info["episode"]["defender_r"]
                    track["is_br"] = evader_runner.is_br

        return copy.deepcopy(self.trajectory), track

    def test_execute_episdoe(self, evader_runner, prior=False, temp=1):

        observation, info = self.env.reset()
        terminated = False
        self.reset()
        evader_runner.reset()

        while not terminated:
            
            defender_obs = (info["evader_history"], info["defender_history"][-1])
            attacker_obs = (info["evader_history"], info["defender_history"][0])

            # defender action
            defender_act = self.policy.select_act(defender_obs, info["defender_legal_actions"], prior=prior, temp=temp)
            # evader action
            attacker_act = evader_runner.select_act(attacker_obs)
            env_action = np.array((attacker_act,) + defender_act, dtype=int) # shape: (defender_num +1, )
                
            # Env steps
            observation, reward, terminated, truncated, infos = self.env.step(env_action)

            if terminated:
                return reward

    def train(self, evader_runner):
        start_time = time.time()
        last_time = start_time
        print(f"Beginning training for {self.args.max_episodes} episodes")
        
        last_train_e = 0
        last_test_e = -self.args.test_every - 1
        last_save_e = 0
        last_log_e = 0

        for train_episode in range(0, self.args.max_episodes):

            defender_trajectory, evader_data = self.train_execute_episode(evader_runner)
            self.policy.add_trajectory(defender_trajectory)

            if evader_runner.require_update:
                evader_runner.update(evader_data["selected_exit"], evader_data["evader_reward"])
                if evader_data["is_br"]:
                    exit_idx=evader_runner.exits.index(evader_data["selected_exit"])
                    evader_runner.cache[exit_idx]+=1                 
            
            if len(self.policy.buffer) >= self.args.train_from and (train_episode-last_train_e) / self.args.train_every >= 1.0:
                v_loss, def_pre_loss, att_pre_loss = self.policy.learn()
                print(f"training_episode: {train_episode+1}, v_loss: {v_loss.item()}, def_pre_loss: {def_pre_loss.item()}, att_pre_loss: {att_pre_loss.item()}")       
                last_train_e = train_episode

            if (train_episode-last_test_e) / self.args.test_every >= 1.0:
                stime = time.time()
                print(f"episodes: {train_episode} / {self.args.max_episodes}")
                last_test_e = train_episode

                test_performance = 0
                for _ in range(self.args.test_nepisodes):
                    test_reward = self.test_execute_episdoe(evader_runner)
                    test_reward = max(test_reward, 0.)
                    test_performance += test_reward

                print(f"Episode: {train_episode}, BR Defender return : {test_performance / self.args.test_nepisodes}")
                print(time.time()-stime)

            if self.args.save_model and (train_episode-last_save_e) / self.args.save_every >= 1.0:
                
                last_save_e = train_episode
                save_path = os.path.join(self.save_path, "models")
                self.save_models(save_path, train_episode)
                print(f"Saving models to {save_path}")

            if (train_episode-last_log_e) / self.args.log_every >= 1.0:
                print(f"Estimated time left: {time_left(last_time, last_log_e, train_episode, self.args.max_episodes)}. Time passed: {time_str(time.time() - start_time)}")
                last_time = time.time()
                last_log_e= train_episode
                if self.args.att_type=="nfsp":
                    prob = evader_runner.N_acts/evader_runner.N_acts.sum()
                    prob = np.around(prob, decimals=4)
                    print(f"Average Prob: {prob}")
                    print(f"Action Value Est: {evader_runner.act_est}")
                    value = []
                    for key in evader_runner.exits:
                        value.append(evader_runner.act_est[key])
                    a = - value[arg_max(value)]
                    print(f"Worst Case Est: {a}")   

    def save_models(self, save_folder, prefix=None):
        os.makedirs(save_folder, exist_ok=True)
        pr_net_name = 'pr_net.pt'
        dy_net_name = 'dy_net.pt'
        if prefix:
            pr_net_name = f"{str(prefix)} + '_{pr_net_name}'"
            dy_net_name = f"{str(prefix)} + '_{dy_net_name}'"
        torch.save(self.policy.pr_net.state_dict(), os.path.join(save_folder, pr_net_name))
        torch.save(self.policy.dy_net.state_dict(), os.path.join(save_folder, dy_net_name))

    def load_models(self, save_folder, prefix=None):
        pr_net_name = 'pr_net.pt'
        dy_net_name = 'dy_net.pt'
        if prefix:
            pr_net_name = f"{str(prefix)} + '_{pr_net_name}'"
            dy_net_name = f"{str(prefix)} + '_{dy_net_name}'"        
        self.policy.pr_net.load_state_dict(torch.load(f"{save_folder}/{pr_net_name}", map_location=torch.device(self.args.device)))
        self.policy.dy_net.load_state_dict(torch.load(f"{save_folder}/{dy_net_name}", map_location=torch.device(self.args.device)))

    def reset(self):
        pass