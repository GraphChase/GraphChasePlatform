import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math

def kaiming_uniform(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

from agent.grasper.utils import (
    huber_loss, 
    get_gard_norm, 
    ValueNorm
    )

class SharedReplayBufferFT(object):
    def __init__(self, mappo_args, args, share_obs_shape, obs_shape, act_space, num_agent):
        self.args = args
        self.batch_size = 32
        self.gamma = mappo_args.gamma
        self.gae_lambda = mappo_args.gae_lambda
        self._use_gae = mappo_args.use_gae
        self._use_popart = mappo_args.use_popart
        self._use_valuenorm = mappo_args.use_valuenorm
        self.share_obs_shape = share_obs_shape
        self.obs_shape = obs_shape
        self.act_space = act_space
        self.num_agent = num_agent

        self.share_obs = []
        self.obs = []
        self.pooled_node_emb = []
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

    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, demo_act_probs=None, pooled_node_emb=None):
        self.share_obs.append(share_obs.copy())
        self.obs.append(obs.copy())
        self.value_preds.append(value_preds.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.masks.append(masks.copy())
        self.returns.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
        if demo_act_probs is not None:
            self.demo_act_probs.append(demo_act_probs.copy())
        if pooled_node_emb is not None:
            self.pooled_node_emb.append(pooled_node_emb.copy())

    def after_update(self):
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
        if len(self.pooled_node_emb) > 0:
            del self.pooled_node_emb[:]
        del self.episode_length[:]

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_gae:
            gae = 0
            for step in reversed(range(len(self.rewards))):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1] if step < len(self.rewards) - 1 else next_value) \
                            * self.masks[step] - value_normalizer.denormalize(self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step] * gae
                else:
                    delta = self.rewards[step] + self.gamma * (self.value_preds[step + 1] if step < len(self.rewards) - 1 else next_value) \
                            * self.masks[step] - self.value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step] * gae
                    self.returns[step] = gae + self.value_preds[step]
        else:
            for step in reversed(range(len(self.rewards))):
                self.returns[step] = (self.returns[step + 1] if step < len(self.rewards) - 1 else next_value) * self.gamma * self.masks[step] + self.rewards[step]

    def get_batch(self, advantages, device):
        total_transition_num = len(self.share_obs) * self.num_agent
        rand = torch.randperm(total_transition_num).numpy()
        indices = rand[:self.batch_size]
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
        if len(self.pooled_node_emb) > 0:
            pooled_node_embs_batch = torch.FloatTensor(np.array([self.pooled_node_emb[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            pooled_node_embs_batch = None
        return share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch, pooled_node_embs_batch        

class R_Actor(nn.Module):
    def __init__(self, args, state_dim, action_dim, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hypernet_hidden_dim
        self.linear1 = nn.Linear(state_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, action_dim)
        self.apply(kaiming_uniform)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, states, pooled_node_embs=None, batch=False):
        x = F.relu(self.linear1(states))
        x = F.relu(self.linear2(x))
        actor_features = self.linear3(x)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, states, action, pooled_node_embs=None, batch=False):
        x = F.relu(self.linear1(states))
        x = F.relu(self.linear2(x))
        actor_features = self.linear3(x)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()

    def init_paras(self, w, b):
        self.linear1.weight.data.copy_(w[0].data)
        self.linear1.bias.data.copy_(b[0].data)
        self.linear2.weight.data.copy_(w[1].data)
        self.linear2.bias.data.copy_(b[1].data)
        self.linear3.weight.data.copy_(w[2].data)
        self.linear3.bias.data.copy_(b[2].data)


class R_Critic(nn.Module):
    def __init__(self, args, state_dim, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hypernet_hidden_dim
        self.linear1 = nn.Linear(state_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)
        self.apply(kaiming_uniform)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, states, pooled_node_embs=None, batch=False):
        x = F.relu(self.linear1(states))
        x = F.relu(self.linear2(x))
        values = self.linear3(x)
        return values

    def init_paras(self, w, b):
        self.linear1.weight.data.copy_(w[0].data)
        self.linear1.bias.data.copy_(b[0].data)
        self.linear2.weight.data.copy_(w[1].data)
        self.linear2.bias.data.copy_(b[1].data)
        self.linear3.weight.data.copy_(w[2].data)
        self.linear3.bias.data.copy_(b[2].data)


class R_Actor_With_Emb_Layer(nn.Module):
    def __init__(self, args, action_dim, node_num, defender_num, device=torch.device("cpu")):
        super(R_Actor_With_Emb_Layer, self).__init__()
        self.args = args
        self.hidden_size = args.hypernet_hidden_dim
        if args.use_augmentation:
            policy_input_dim = args.state_emb_dim * (defender_num + 3) + args.gnn_output_dim
        else:
            policy_input_dim = args.state_emb_dim * (defender_num + 3)
        self.node_idx_emb_layer = nn.Embedding(node_num + 1, args.state_emb_dim)
        self.time_idx_emb_layer = nn.Embedding(args.max_time_horizon_for_state_emb, args.state_emb_dim)
        self.agent_id_emb_layer = nn.Embedding(defender_num, args.state_emb_dim)
        self.linear1 = nn.Linear(policy_input_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, action_dim)
        self.apply(kaiming_uniform)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, states, pooled_node_embs, batch=False):
        node_idx = states[:, :-2] if batch else states[:-2].unsqueeze(0)
        time_idx = states[:, -2] if batch else states[-2].unsqueeze(0)
        agent_id = states[:, -1] if batch else states[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        agent_id_emb = self.agent_id_emb_layer(agent_id)
        state_emb = torch.cat([node_idx_emb, time_idx_emb, agent_id_emb], dim=1)
        if self.args.use_augmentation:
            state_emb = torch.cat([state_emb, pooled_node_embs], dim=1)
        x = F.relu(self.linear1(state_emb))
        x = F.relu(self.linear2(x))
        actor_features = self.linear3(x)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, states, action, pooled_node_embs, batch=False):
        node_idx = states[:, :-2] if batch else states[:-2].unsqueeze(0)
        time_idx = states[:, -2] if batch else states[-2].unsqueeze(0)
        agent_id = states[:, -1] if batch else states[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        agent_id_emb = self.agent_id_emb_layer(agent_id)
        state_emb = torch.cat([node_idx_emb, time_idx_emb, agent_id_emb], dim=1)
        if self.args.use_augmentation:
            state_emb = torch.cat([state_emb, pooled_node_embs], dim=1)
        x = F.relu(self.linear1(state_emb))
        x = F.relu(self.linear2(x))
        actor_features = self.linear3(x)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()

    def init_paras(self, w, b):
        self.linear1.weight.data.copy_(w[0].data)
        self.linear1.bias.data.copy_(b[0].data)
        self.linear2.weight.data.copy_(w[1].data)
        self.linear2.bias.data.copy_(b[1].data)
        self.linear3.weight.data.copy_(w[2].data)
        self.linear3.bias.data.copy_(b[2].data)

    def init_emb_layer(self, node_idx_emb_state_dict, time_idx_emb_state_dict, agent_id_emb_state_dict):
        self.node_idx_emb_layer.load_state_dict(node_idx_emb_state_dict)
        self.time_idx_emb_layer.load_state_dict(time_idx_emb_state_dict)
        self.agent_id_emb_layer.load_state_dict(agent_id_emb_state_dict)


class R_Critic_With_Emb_Layer(nn.Module):
    def __init__(self, args, node_num, defender_num,  device=torch.device("cpu")):
        super(R_Critic_With_Emb_Layer, self).__init__()
        self.args = args
        self.hidden_size = args.hypernet_hidden_dim
        if args.use_augmentation:
            value_input_dim = args.state_emb_dim * (defender_num + 2) + args.gnn_output_dim
        else:
            value_input_dim = args.state_emb_dim * (defender_num + 2)
        self.node_idx_emb_layer = nn.Embedding(node_num + 1, args.state_emb_dim)
        self.time_idx_emb_layer = nn.Embedding(args.max_time_horizon_for_state_emb, args.state_emb_dim)
        self.linear1 = nn.Linear(value_input_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)
        self.apply(kaiming_uniform)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, states, pooled_node_embs, batch=False):
        node_idx = states[:, :-1] if batch else states[:-1].unsqueeze(0)
        time_idx = states[:, -1] if batch else states[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        state_emb = torch.cat([node_idx_emb, time_idx_emb], dim=1)
        if self.args.use_augmentation:
            state_emb = torch.cat([state_emb, pooled_node_embs], dim=1)
        x = F.relu(self.linear1(state_emb))
        x = F.relu(self.linear2(x))
        values = self.linear3(x)
        return values

    def init_paras(self, w, b):
        self.linear1.weight.data.copy_(w[0].data)
        self.linear1.bias.data.copy_(b[0].data)
        self.linear2.weight.data.copy_(w[1].data)
        self.linear2.bias.data.copy_(b[1].data)
        self.linear3.weight.data.copy_(w[2].data)
        self.linear3.bias.data.copy_(b[2].data)

    def init_emb_layer(self, node_idx_emb_state_dict, time_idx_emb_state_dict):
        self.node_idx_emb_layer.load_state_dict(node_idx_emb_state_dict)
        self.time_idx_emb_layer.load_state_dict(time_idx_emb_state_dict)


class RMAPPOPolicy:
    def __init__(self, mappo_args, args, obs_space, cent_obs_space, act_space):
        self.device = args.device
        self.lr = mappo_args.lr
        self.critic_lr = mappo_args.critic_lr
        self.opti_eps = mappo_args.opti_eps
        self.weight_decay = mappo_args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        if args.use_emb_layer:
            self.actor = R_Actor_With_Emb_Layer(args, self.act_space, args.node_num, args.defender_num, self.device)
            self.critic = R_Critic_With_Emb_Layer(args, args.node_num, args.defender_num, self.device)
        else:
            self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
            self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, cent_obs, obs, pooled_node_embs, batch=False):
        actions, action_log_probs = self.actor(obs, pooled_node_embs, batch=batch)
        values = self.critic(cent_obs, pooled_node_embs, batch=batch)
        return values, actions, action_log_probs

    def get_values(self, cent_obs, pooled_node_embs, batch=False):
        values = self.critic(cent_obs, pooled_node_embs, batch=batch)
        return values

    def evaluate_actions(self, cent_obs, obs, action, pooled_node_embs, batch=False):
        action_log_probs, dist_entropy, action_probs = self.actor.evaluate_actions(obs, action, pooled_node_embs, batch=batch)
        values = self.critic(cent_obs, pooled_node_embs, batch=batch)
        return values, action_log_probs, dist_entropy, action_probs

    def act(self, obs, pooled_node_embs, batch=False):
        actions, _ = self.actor(obs, pooled_node_embs, batch=batch)
        return actions
    
    def save(self, path):
        actor_path = f"{path}_actor.pt"
        critic_path = f"{path}_critic.pt"
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

class RMAPPO:
    def __init__(self, env, mappo_args, args):
        self.env = env
        self.mappo_args = mappo_args
        self.args = args
        self.device = args.device       

        obs_dim = 1 + 1 + 1 + 1 # evader loc + own loc + time step + own id
        share_obs_size = self.env.defender_num + self.env.evader_num + 1 # position and timestep
        obs_size = share_obs_size + 1 # position, timestep and id

        self.buffer = SharedReplayBufferFT(self.mappo_args, self.args, share_obs_size, obs_dim, len(self.env._action_to_direction), self.env.defender_num)

        self.policy = RMAPPOPolicy(self.mappo_args, self.args, obs_size, share_obs_size, len(self.env._action_to_direction))
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
        share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch, pooled_node_embs_batch = sample
        values, action_log_probs, dist_entropy, log_probs = self.policy.evaluate_actions(share_obs_batch, obs_batch, actions_batch, pooled_node_embs_batch, batch=True)
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
        trained = False
        train_info = {'value_loss': 0, 'policy_loss': 0, 'dist_entropy': 0, 'actor_grad_norm': 0, 'critic_grad_norm': 0, 'ratio': 0}
        total_transition_num = len(self.buffer.share_obs) * self.buffer.num_agent
        if total_transition_num > self.buffer.batch_size:
            trained = True
            if self._use_popart or self._use_valuenorm:
                advantages = np.array(self.buffer.returns) - self.value_normalizer.denormalize(np.array(self.buffer.value_preds))
            else:
                advantages = np.array(self.buffer.returns) - np.array(self.buffer.value_preds)
            if self._use_advnorm:
                advantages_copy = advantages.copy()
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            for _ in range(self.ppo_epoch):
                sample = self.buffer.get_batch(advantages, self.device)
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
        return train_info, trained

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    def initialize_policy_parameters(self, init_method):
        self.policy.actor.initialize_parameters(init_method)
        self.policy.critic.initialize_parameters(init_method)  
