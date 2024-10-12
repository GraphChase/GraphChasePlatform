import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import itertools
import dgl
from agent.grasper.hyper_nets import (
    HyperNetwork, 
    Critic_With_Emb_Layer_Scratch, 
    Critic_With_Emb_Layer,
    Actor_With_Emb_Layer_Scratch,
    Actor_With_Emb_Layer
    )

from agent.grasper.utils import (
    huber_loss, 
    get_gard_norm, 
    ValueNorm
    )

class H_Actor(nn.Module):
    def __init__(self, args, obs_dim, hyper_input_dim, action_dim, device=torch.device("cpu")):
        super(H_Actor, self).__init__()
        self.hidden_size = args.hypernet_hidden_dim
        self.base = HyperNetwork(obs_dim, hyper_input_dim, action_dim, device, dynamic_hidden_dim=self.hidden_size,
                                 use_augmentation=args.use_augmentation, head_init_method=args.head_init_method)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pooled_node_emb, T, obs, batch=False):
        actor_features = self.base(pooled_node_emb, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, pooled_node_emb, T, obs, action, batch=False):
        actor_features = self.base(pooled_node_emb, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()
    
    def initialize_parameters(self, init_method):
        for param in self.parameters():
            if param.dim() > 1:  # 只对维度大于1的张量应用初始化方法
                init_method(param)      


class H_Critic(nn.Module):
    def __init__(self, args, cent_obs_dim, hyper_input_dim, device=torch.device("cpu")):
        super(H_Critic, self).__init__()
        self.hidden_size = args.hypernet_hidden_dim
        self.base = HyperNetwork(cent_obs_dim, hyper_input_dim, 1, device, dynamic_hidden_dim=self.hidden_size,
                                 use_augmentation=args.use_augmentation, head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pooled_node_emb, T, cent_obs, batch=False):
        values = self.base(pooled_node_emb, T, cent_obs, batch=batch)
        return values
    
    def initialize_parameters(self, init_method):
        for param in self.parameters():
            if param.dim() > 1:  # 只对维度大于1的张量应用初始化方法
                init_method(param)       


class H_Actor_With_Emb_Layer(nn.Module):
    def __init__(self, args, hyper_input_dim, action_dim, device=torch.device("cpu")):
        super(H_Actor_With_Emb_Layer, self).__init__()
        self.hidden_size = args.hypernet_hidden_dim
        self.base = Actor_With_Emb_Layer(args, hyper_input_dim, action_dim, args.node_num, args.defender_num, device,
                                         dynamic_hidden_dim=self.hidden_size, head_init_method=args.head_init_method)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pooled_node_emb, T, obs, batch=False):
        actor_features = self.base(pooled_node_emb, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, pooled_node_emb, T, obs, action, batch=False):
        actor_features = self.base(pooled_node_emb, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()

    def initialize_parameters(self, init_method):
        for param in self.parameters():
            if param.dim() > 1:  # 只对维度大于1的张量应用初始化方法
                init_method(param)         


class H_Critic_With_Emb_Layer(nn.Module):
    def __init__(self, args, hyper_input_dim, device=torch.device("cpu")):
        super(H_Critic_With_Emb_Layer, self).__init__()
        self.hidden_size = args.hypernet_hidden_dim
        self.base = Critic_With_Emb_Layer(args, hyper_input_dim, args.node_num, args.defender_num, device,
                                          dynamic_hidden_dim=self.hidden_size, head_init_method=args.head_init_method)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pooled_node_emb, T, cent_obs, batch=False):
        values = self.base(pooled_node_emb, T, cent_obs, batch=batch)
        return values
    
    def initialize_parameters(self, init_method):
        for param in self.parameters():
            if param.dim() > 1:  # 只对维度大于1的张量应用初始化方法
                init_method(param)        


class H_Actor_With_Emb_Layer_Scratch(nn.Module):
    def __init__(self, args, action_dim, device=torch.device("cpu")):
        super(H_Actor_With_Emb_Layer_Scratch, self).__init__()
        self.hidden_size = args.hypernet_hidden_dim
        self.base = Actor_With_Emb_Layer_Scratch(args, action_dim, args.node_num, args.defender_num, device,
                                                 dynamic_hidden_dim=self.hidden_size, head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, hgs, T, obs, batch=False):
        actor_features = self.base(hgs, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, hgs, T, obs, action, batch=False):
        actor_features = self.base(hgs, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()


class H_Critic_With_Emb_Layer_Scratch(nn.Module):
    def __init__(self, args, device=torch.device("cpu")):
        super(H_Critic_With_Emb_Layer_Scratch, self).__init__()
        self.hidden_size = args.hypernet_hidden_dim
        self.base = Critic_With_Emb_Layer_Scratch(args, args.node_num, args.defender_num, device,
                                                  dynamic_hidden_dim=self.hidden_size,
                                                  head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, hgs, T, cent_obs, batch=False):
        values = self.base(hgs, T, cent_obs, batch=batch)
        return values

class SharedReplayBuffer(object):
    def __init__(self, mappo_args, args, share_obs_shape, obs_shape, act_space, num_agent):
        self.args = args
        self.batch_size = args.batch_size
        self.gamma = mappo_args.gamma
        self.gae_lambda = mappo_args.gae_lambda
        self._use_gae = mappo_args.use_gae
        self._use_popart = mappo_args.use_popart
        self._use_valuenorm = mappo_args.use_valuenorm
        self.pooled_node_emb_shape = args.gnn_output_dim
        self.T_shape = args.max_time_horizon_for_state_emb
        self.share_obs_shape = share_obs_shape
        self.obs_shape = obs_shape
        self.act_space = act_space
        self.num_agent = num_agent

        self.share_obs = []
        self.obs = []
        self.pooled_node_embs = []
        self.Ts = []
        self.value_preds = []
        self.returns = []
        self.actions = []
        self.demo_act_probs = []
        self.action_log_probs = []
        self.rewards = []
        self.masks = []
        self.episode_length = []

        self.value_preds_one_episode = []
        self.rewards_one_episode = []
        self.returns_one_episode = []
        self.masks_one_episode = []


    def insert(self, pooled_node_embs, Ts, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, demo_act_probs=None):
        self.share_obs.append(share_obs.copy())
        self.obs.append(obs.copy())
        self.pooled_node_embs.append(pooled_node_embs.copy())
        self.Ts.append(Ts.copy())
        self.value_preds.append(value_preds.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.masks.append(masks.copy())
        self.value_preds_one_episode.append(value_preds.copy())
        self.rewards_one_episode.append(rewards.copy())
        self.returns_one_episode.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
        self.masks_one_episode.append(masks.copy())
        if demo_act_probs is not None:
            self.demo_act_probs.append(demo_act_probs.copy())

    def after_update(self):
        del self.pooled_node_embs[:]  # clear experience
        del self.Ts[:]
        del self.share_obs[:]
        del self.obs[:]
        del self.value_preds[:]
        del self.returns[:]
        del self.actions[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.masks[:]
        if len(self.demo_act_probs) > 0:
            del self.demo_act_probs[:]
        del self.episode_length[:]

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_gae:
            gae = 0
            for step in reversed(range(len(self.rewards_one_episode))):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards_one_episode[step] + self.gamma * value_normalizer.denormalize(self.value_preds_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                            * self.masks_one_episode[step] - value_normalizer.denormalize(self.value_preds_one_episode[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks_one_episode[step] * gae
                else:
                    delta = self.rewards_one_episode[step] + self.gamma * (self.value_preds_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                            * self.masks_one_episode[step] - self.value_preds_one_episode[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks_one_episode[step] * gae
                    self.returns_one_episode[step] = gae + self.value_preds_one_episode[step]
        else:
            for step in reversed(range(len(self.rewards_one_episode))):
                self.returns_one_episode[step] = (self.returns_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                                                 * self.gamma * self.masks_one_episode[step] + self.rewards_one_episode[step]
        self.returns.extend(self.returns_one_episode)
        del self.value_preds_one_episode[:]
        del self.rewards_one_episode[:]
        del self.returns_one_episode[:]
        del self.masks_one_episode[:]

    def get_batch(self, advantages, device):
        total_transition_num = len(self.pooled_node_embs) * self.num_agent
        batch_size = min(total_transition_num, self.batch_size)
        rand = torch.randperm(total_transition_num).numpy()
        indices = rand[:batch_size]
        pooled_node_embs_batch = torch.FloatTensor(np.array([self.pooled_node_embs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        Ts_batch = torch.FloatTensor(np.array([self.Ts[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if self.args.use_emb_layer:
            share_obs_batch = torch.LongTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.LongTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            share_obs_batch = torch.FloatTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.FloatTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        actions_batch = torch.FloatTensor(np.array([self.actions[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        value_preds_batch = torch.FloatTensor(np.array([self.value_preds[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        return_batch = torch.FloatTensor(np.array([self.returns[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        masks_batch = torch.FloatTensor(np.array([self.masks[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        old_action_log_probs_batch = torch.FloatTensor(np.array([self.action_log_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = torch.FloatTensor(np.array([advantages[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if len(self.demo_act_probs) > 0:
            demo_act_probs_batch = torch.FloatTensor(np.array([self.demo_act_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            demo_act_probs_batch = None
        return pooled_node_embs_batch, Ts_batch, share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch


class SharedReplayBufferEnd2End(object):
    def __init__(self, mappo_args, args, share_obs_shape, obs_shape, act_space, num_agent):
        self.args = args
        self.batch_size = args.batch_size
        self.gamma = mappo_args.gamma
        self.gae_lambda = mappo_args.gae_lambda
        self._use_gae = mappo_args.use_gae
        self._use_popart = mappo_args.use_popart
        self._use_valuenorm = mappo_args.use_valuenorm
        self.T_shape = args.max_time_horizon_for_state_emb
        self.share_obs_shape = share_obs_shape
        self.obs_shape = obs_shape
        self.act_space = act_space
        self.num_agent = num_agent

        self.share_obs = []
        self.obs = []
        self.hgs = []
        self.Ts = []
        self.value_preds = []
        self.returns = []
        self.actions = []
        self.demo_act_probs = []
        self.action_log_probs = []
        self.rewards = []
        self.masks = []
        self.episode_length = []
        self.value_preds_one_episode = []
        self.rewards_one_episode = []
        self.returns_one_episode = []
        self.masks_one_episode = []

    def insert(self, hgs, Ts, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, demo_act_probs=None):
        self.share_obs.append(share_obs.copy())
        self.obs.append(obs.copy())
        self.hgs.append(hgs)
        self.Ts.append(Ts.copy())
        self.value_preds.append(value_preds.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.masks.append(masks.copy())
        self.value_preds_one_episode.append(value_preds.copy())
        self.rewards_one_episode.append(rewards.copy())
        self.returns_one_episode.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
        self.masks_one_episode.append(masks.copy())
        if demo_act_probs is not None:
            self.demo_act_probs.append(demo_act_probs.copy())

    def after_update(self):
        del self.hgs[:]  # clear experience
        del self.Ts[:]
        del self.share_obs[:]
        del self.obs[:]
        del self.value_preds[:]
        del self.returns[:]
        del self.actions[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.masks[:]
        if len(self.demo_act_probs) > 0:
            del self.demo_act_probs[:]
        del self.episode_length[:]

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_gae:
            gae = 0
            for step in reversed(range(len(self.rewards_one_episode))):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards_one_episode[step] + self.gamma * value_normalizer.denormalize(self.value_preds_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                            * self.masks_one_episode[step] - value_normalizer.denormalize(self.value_preds_one_episode[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks_one_episode[step] * gae
                else:
                    delta = self.rewards_one_episode[step] + self.gamma * (self.value_preds_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                            * self.masks_one_episode[step] - self.value_preds_one_episode[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks_one_episode[step] * gae
                    self.returns_one_episode[step] = gae + self.value_preds_one_episode[step]
        else:
            for step in reversed(range(len(self.rewards_one_episode))):
                self.returns_one_episode[step] = (self.returns_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                                                 * self.gamma * self.masks_one_episode[step] + self.rewards_one_episode[step]
        self.returns.extend(self.returns_one_episode)
        del self.value_preds_one_episode[:]
        del self.rewards_one_episode[:]
        del self.returns_one_episode[:]
        del self.masks_one_episode[:]

    def get_batch(self, advantages, device):
        total_transition_num = len(self.share_obs) * self.num_agent
        batch_size = min(total_transition_num, self.batch_size)
        rand = torch.randperm(total_transition_num).numpy()
        indices = rand[:batch_size]
        graphs = [self.hgs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices]
        hgs_batch = dgl.batch(graphs)
        hgs_batch = hgs_batch.to(device)
        Ts_batch = torch.FloatTensor(np.array([self.Ts[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if self.args.use_emb_layer:
            share_obs_batch = torch.LongTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.LongTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            share_obs_batch = torch.FloatTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.FloatTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        actions_batch = torch.FloatTensor(np.array([self.actions[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        value_preds_batch = torch.FloatTensor(np.array([self.value_preds[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        return_batch = torch.FloatTensor(np.array([self.returns[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        masks_batch = torch.FloatTensor(np.array([self.masks[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        old_action_log_probs_batch = torch.FloatTensor(np.array([self.action_log_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = torch.FloatTensor(np.array([advantages[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if len(self.demo_act_probs) > 0:
            demo_act_probs_batch = torch.FloatTensor(np.array([self.demo_act_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            demo_act_probs_batch = None
        return hgs_batch, Ts_batch, share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch


class RHMAPPOPolicy:
    def __init__(self, mappo_args, args, obs_space, cent_obs_space, hyper_input_dim, act_space):
        self.device = args.device
        self.lr = mappo_args.lr
        self.critic_lr = mappo_args.critic_lr
        self.opti_eps = mappo_args.opti_eps
        self.weight_decay = mappo_args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.hyper_input_dim = hyper_input_dim
        self.act_space = act_space

        if args.use_emb_layer:
            self.actor = H_Actor_With_Emb_Layer(args, hyper_input_dim, self.act_space, self.device)
            self.critic = H_Critic_With_Emb_Layer(args, hyper_input_dim, self.device)
        else:
            self.actor = H_Actor(args, self.obs_space, self.hyper_input_dim, self.act_space, self.device)
            self.critic = H_Critic(args, self.share_obs_space, self.hyper_input_dim, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, pooled_node_emb, T, cent_obs, obs, batch=False):
        actions, action_log_probs = self.actor(pooled_node_emb, T, obs, batch=batch)
        values = self.critic(pooled_node_emb, T, cent_obs, batch=batch)
        return values, actions, action_log_probs

    def get_values(self, pooled_node_emb, T, cent_obs, batch=False):
        values = self.critic(pooled_node_emb, T, cent_obs, batch=batch)
        return values

    def evaluate_actions(self, pooled_node_emb, T, cent_obs, obs, action, batch=False):
        action_log_probs, dist_entropy, action_probs = self.actor.evaluate_actions(pooled_node_emb, T, obs, action, batch=batch)
        values = self.critic(pooled_node_emb, T, cent_obs, batch=batch)
        return values, action_log_probs, dist_entropy, action_probs

    def act(self, pooled_node_emb, T, obs, batch=False):
        actions, _ = self.actor(pooled_node_emb, T, obs, batch=batch)
        return actions


class RHMAPPOPolicyScratch:
    def __init__(self, mappo_args, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = mappo_args.lr
        self.critic_lr = mappo_args.critic_lr
        self.opti_eps = mappo_args.opti_eps
        self.weight_decay = mappo_args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        if args.use_emb_layer:
            self.actor = H_Actor_With_Emb_Layer_Scratch(args, self.act_space, self.device)
            self.critic = H_Critic_With_Emb_Layer_Scratch(args, self.device)
        else:
            raise ValueError("Must use observation embedding layer.")
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, hgs, T, cent_obs, obs, batch=False):
        actions, action_log_probs = self.actor(hgs, T, obs, batch=batch)
        values = self.critic(hgs, T, cent_obs, batch=batch)
        return values, actions, action_log_probs

    def get_values(self, hgs, T, cent_obs, batch=False):
        values = self.critic(hgs, T, cent_obs, batch=batch)
        return values

    def evaluate_actions(self, hgs, T, cent_obs, obs, action, batch=False):
        action_log_probs, dist_entropy, action_probs = self.actor.evaluate_actions(hgs, T, obs, action, batch=batch)
        values = self.critic(hgs, T, cent_obs, batch=batch)
        return values, action_log_probs, dist_entropy, action_probs

    def act(self, hgs, T, obs, batch=False):
        actions, _ = self.actor(hgs, T, obs, batch=batch)
        return actions

class RHMAPPO:
    def __init__(self, env, mappo_args, args):
        self.env = env
        self.mappo_args = mappo_args
        self.args = args
        self.device = args.device

        if args.graph_type == 'Grid_Graph':
            args.node_num = args.row_max_for_state_emb * args.column_max_for_state_emb  # enable varied grid size
        else:
            args.node_num = env.graph.get_graph_info().shape[0]
        args.defender_num = env.defender_num
        action_list = []
        for _ in range(env.defender_num):
            action_list.append([i for i in range(5)])
        defender_action_map = list(itertools.product(*action_list))
        args.action_dim = len(defender_action_map)         

        obs_dim = 1 + 1 + 1 + 1 # evader loc + own loc + time step + own id
        share_obs_size = self.env.defender_num + self.env.evader_num + 1 # position and timestep
        obs_size = share_obs_size + 1 # position, timestep and id
        if args.use_end_to_end:
            self.policy = RHMAPPOPolicyScratch(self.mappo_args, self.args, obs_size, share_obs_size, self.env.action_dim, device=self.device)
            self.buffer = SharedReplayBufferEnd2End(self.mappo_args, self.args, self.env.share_obs_dim, self.env.obs_dim, self.env.action_dim, self.env.defender_num)
        else:
            hyper_input_dim = args.gnn_output_dim + args.max_time_horizon_for_state_emb
            self.policy = RHMAPPOPolicy(self.mappo_args, self.args, obs_size, share_obs_size, hyper_input_dim, len(self.env._action_to_direction))
            self.buffer = SharedReplayBuffer(self.mappo_args, self.args, share_obs_size, obs_dim, len(self.env._action_to_direction), self.env.defender_num)        
        
        self.clip_param = mappo_args.clip_param
        self.ppo_epoch = mappo_args.ppo_epoch
        self.value_loss_coef = mappo_args.value_loss_coef
        self.entropy_coef = mappo_args.entropy_coef
        self.act_sup_coef_min = mappo_args.act_sup_coef_min
        self.act_sup_coef_max = mappo_args.act_sup_coef_max
        self.act_sup_coef_decay = mappo_args.act_sup_coef_decay
        self.act_sup_coef = self.act_sup_coef_max
        self.max_grad_norm = mappo_args.max_grad_norm
        self.huber_delta = mappo_args.huber_delta
        self._use_max_grad_norm = mappo_args.use_max_grad_norm
        self._use_clipped_value_loss = mappo_args.use_clipped_value_loss
        self._use_huber_loss = mappo_args.use_huber_loss
        self._use_popart = mappo_args.use_popart
        self._use_valuenorm = mappo_args.use_valuenorm
        self._use_advnorm = mappo_args.use_advnorm
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
        self._step = int(args.checkpoint / (args.num_games * args.num_task * args.num_sample))

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = F.mse_loss(return_batch, value_pred_clipped)
            value_loss_original = F.mse_loss(return_batch, values)
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        if self._use_huber_loss:
            value_loss = value_loss.mean()
        return value_loss

    def act_sup_coef_linear_decay(self):
        if self._step > self.act_sup_coef_decay:
            self.act_sup_coef = self.act_sup_coef_min
        else:
            self.act_sup_coef = self.act_sup_coef_max - self.act_sup_coef_max * (self._step / float(self.act_sup_coef_decay))

    def ppo_update(self, sample, update_actor=True):
        if self.args.use_end_to_end:
            hgs_batch, Ts_batch, share_obs_batch, obs_batch, actions_batch, value_preds_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch = sample
            values, action_log_probs, dist_entropy, log_probs = self.policy.evaluate_actions(hgs_batch,
                                                                                             Ts_batch, share_obs_batch,
                                                                                             obs_batch, actions_batch,
                                                                                             batch=True)
        else:
            pooled_node_embs_batch, Ts_batch, share_obs_batch, obs_batch, actions_batch, value_preds_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch = sample
            values, action_log_probs, dist_entropy, log_probs = self.policy.evaluate_actions(pooled_node_embs_batch,
                                                                                             Ts_batch, share_obs_batch,
                                                                                             obs_batch, actions_batch,
                                                                                             batch=True)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            if demo_act_probs_batch is not None:
                (policy_loss - dist_entropy * self.entropy_coef + self.act_sup_coef * F.kl_div(log_probs, demo_act_probs_batch, reduction="batchmean")).backward()
            else:
                (policy_loss - dist_entropy * self.entropy_coef).backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, demo_act_probs_batch

    def train(self, update_actor=True):
        buffer = self.buffer
        train_info = {'value_loss': 0, 'policy_loss': 0, 'dist_entropy': 0, 'actor_grad_norm': 0, 'critic_grad_norm': 0, 'ratio': 0}
        total_transition_num = len(buffer.share_obs) * buffer.num_agent
        if total_transition_num > buffer.batch_size:
            if self._use_popart or self._use_valuenorm:
                advantages = np.array(buffer.returns) - self.value_normalizer.denormalize(np.array(buffer.value_preds))
            else:
                advantages = np.array(buffer.returns) - np.array(buffer.value_preds)
            if self._use_advnorm:
                advantages_copy = advantages.copy()
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            for _ in range(self.ppo_epoch):
                sample = buffer.get_batch(advantages, self.device)
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, demo_act_probs_batch = self.ppo_update(sample, update_actor)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
            if self.ppo_epoch > 1:
                for k in train_info.keys():
                    train_info[k] /= self.ppo_epoch
            self._step += 1
            if demo_act_probs_batch is not None and self.act_sup_coef_min != self.act_sup_coef_max:
                self.act_sup_coef_linear_decay()
            self.buffer.after_update()
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    def initialize_policy_parameters(self, init_method):
        self.policy.actor.initialize_parameters(init_method)
        self.policy.critic.initialize_parameters(init_method)        

    def save(self, file_name):
        torch.save(self.policy.actor.state_dict(), file_name + "_actor.pt")
        torch.save(self.policy.critic.state_dict(), file_name + "_critic.pt")    
