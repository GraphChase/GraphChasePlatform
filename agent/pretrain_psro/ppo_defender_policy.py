import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from env.base_env import BaseGame
from itertools import product
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class DiscreteActor(nn.Module):
    """
    Simple neural network with softmax action selection
    """

    def __init__(self, num_inputs, action_dim, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_dim)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores, dim=-1)


class Critic(nn.Module):
    """
    Value network represents Q(s_t, a_t) for AC, or denotes V(S_t) for A2C.
    """

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x


class PPOAgent(object):      
    def __init__(self, env: BaseGame, args,):

        self.args = args
        self.gamma = self.args.ppo_gamma
        self.device = self.args.device
        self.batch_size = self.args.ppo_batch_size
        self.max_grad_norm = self.args.max_grad_norm
        self.clip_param = self.args.clip_param

        self.rollout_buffer = []

        node_emb_size = self.args.emb_size * 2 if self.args.line_order == "all" else self.args.emb_size
        if self.args.use_past_history:
            state_dim = env.time_horizon + env.evader_num + env.defender_num
        else:
            state_dim = 1 + env.defender_num

        if self.args.use_node_embedding:
            state_dim *= node_emb_size

        if not self.args.use_past_history:
            state_dim += 1

        action_list = []
        for _ in range(env.defender_num):
            action_list.append([i for i in range(5)])
        self.defender_action_map = list(product(*action_list))
        action_dim  = len(self.defender_action_map)        

        self.actor = DiscreteActor(state_dim, action_dim, self.args.ppo_hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.ppo_actor_lr)
        self.critic = Critic(state_dim, self.args.ppo_hidden_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.ppo_critic_lr)

    def predict_log_prob(self, state, old_action):
        probs = self.actor(state).to(self.device)
        m = Categorical(probs)
        log_prob = m.log_prob(old_action)
        return log_prob

    def select_action(self, state: np.ndarray):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        action = action.cpu().numpy()
        return action, log_prob

    def store_transition(self, transition):
        self.rollout_buffer.append(transition)

    def save(self, save_folder, prefix=None):
        os.makedirs(save_folder, exist_ok=True)
        actor_name = 'actor.pt'
        critic_name = 'critic.pt'
        if prefix:
            actor_name = f"{str(prefix)}_{actor_name}"
            critic_name = f"{str(prefix)}_{critic_name}"
        torch.save(self.actor.state_dict(), os.path.join(save_folder, actor_name))
        torch.save(self.critic.state_dict(), os.path.join(save_folder, critic_name))

    def train(self, train_num):
        if len(self.rollout_buffer) < self.batch_size:
            return

        buffer_states = torch.FloatTensor(np.array([item[0] for item in self.rollout_buffer])).to(self.device)
        buffer_actions = torch.LongTensor(np.array([item[1] for item in self.rollout_buffer])).to(self.device)
        buffer_old_log_probs = torch.stack([item[2] for item in self.rollout_buffer]).to(self.device)
        buffer_rewards = [item[3] for item in self.rollout_buffer]

        discounted_episode_return = 0
        buffer_returns = []
        for r in buffer_rewards[::-1]:
            if r != 0.0:
                discounted_episode_return = 0
            discounted_episode_return = r + self.gamma * discounted_episode_return
            buffer_returns.insert(0, discounted_episode_return)
        buffer_returns = torch.FloatTensor(buffer_returns).to(self.device)

        # epoch iteration, PPO core!!!
        for epoch in range(train_num):
            # mini batch
            for batch_indexes in BatchSampler(SubsetRandomSampler(range(len(self.rollout_buffer))), self.batch_size, False):
                states = buffer_states[batch_indexes]
                actions = buffer_actions[batch_indexes]
                log_probs_old = buffer_old_log_probs[batch_indexes]
                returns = buffer_returns[batch_indexes]

                value = self.critic(states).flatten()
                advantage = (returns - value).detach()  # state value, same as A2C

                # core of PPO2
                log_probs_new = self.predict_log_prob(states, actions).flatten()
                ratio = torch.exp(log_probs_new - log_probs_old.flatten())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                actor_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(returns, value)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        del self.rollout_buffer[:]  # clear experience