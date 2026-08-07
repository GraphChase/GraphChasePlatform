import torch
import numpy as np
import dgl

class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    """
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
        self.action_masks = []
        self.episode_length = []

        self.value_preds_one_episode = []
        self.rewards_one_episode = []
        self.returns_one_episode = []
        self.masks_one_episode = []

        self.share_obs_cache = []
        self.obs_cache = []
        self.pooled_node_embs_cache = []
        self.Ts_cache = []
        self.value_preds_cache = []
        self.returns_cache = []
        self.actions_cache = []
        self.demo_act_probs_cache = []
        self.action_log_probs_cache = []
        self.rewards_cache = []
        self.masks_cache = []
        self.action_masks_cache = []
        self.episode_length_cache = []

    def insert(self, pooled_node_embs, Ts, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, action_masks=None, demo_act_probs=None, cache=False):
        if action_masks is None:
            action_masks = np.ones((obs.shape[0], self.act_space), dtype=bool)
        if cache:
            self.share_obs_cache.append(share_obs.copy())
            self.obs_cache.append(obs.copy())
            self.pooled_node_embs_cache.append(pooled_node_embs.copy())
            self.Ts_cache.append(Ts.copy())
            self.value_preds_cache.append(value_preds.copy())
            self.actions_cache.append(actions.copy())
            self.action_log_probs_cache.append(action_log_probs.copy())
            self.rewards_cache.append(rewards.copy())
            self.masks_cache.append(masks.copy())
            self.action_masks_cache.append(action_masks.copy())
            self.value_preds_one_episode.append(value_preds.copy())
            self.rewards_one_episode.append(rewards.copy())
            self.returns_one_episode.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
            self.masks_one_episode.append(masks.copy())
            if demo_act_probs is not None:
                self.demo_act_probs_cache.append(demo_act_probs.copy())
        else:
            self.share_obs.append(share_obs.copy())
            self.obs.append(obs.copy())
            self.pooled_node_embs.append(pooled_node_embs.copy())
            self.Ts.append(Ts.copy())
            self.value_preds.append(value_preds.copy())
            self.actions.append(actions.copy())
            self.action_log_probs.append(action_log_probs.copy())
            self.rewards.append(rewards.copy())
            self.masks.append(masks.copy())
            self.action_masks.append(action_masks.copy())
            self.value_preds_one_episode.append(value_preds.copy())
            self.rewards_one_episode.append(rewards.copy())
            self.returns_one_episode.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
            self.masks_one_episode.append(masks.copy())
            if demo_act_probs is not None:
                self.demo_act_probs.append(demo_act_probs.copy())

    def store_cache(self):
        self.share_obs.extend(self.share_obs_cache)
        self.obs.extend(self.obs_cache)
        self.pooled_node_embs.extend(self.pooled_node_embs_cache)
        self.Ts.extend(self.Ts_cache)
        self.value_preds.extend(self.value_preds_cache)
        self.returns.extend(self.returns_cache)
        self.actions.extend(self.actions_cache)
        if len(self.demo_act_probs_cache) > 0:
            self.demo_act_probs.extend(self.demo_act_probs_cache)
        self.action_log_probs.extend(self.action_log_probs_cache)
        self.rewards.extend(self.rewards_cache)
        self.masks.extend(self.masks_cache)
        self.action_masks.extend(self.action_masks_cache)
        self.episode_length.extend(self.episode_length_cache)

        del self.pooled_node_embs_cache[:]  # clear experience
        del self.Ts_cache[:]
        del self.share_obs_cache[:]
        del self.obs_cache[:]
        del self.value_preds_cache[:]
        del self.returns_cache[:]
        del self.actions_cache[:]
        del self.action_log_probs_cache[:]
        del self.rewards_cache[:]
        del self.masks_cache[:]
        del self.action_masks_cache[:]
        if len(self.demo_act_probs_cache) > 0:
            del self.demo_act_probs_cache[:]
        del self.episode_length_cache[:]

    def after_update(self, update_every_n_episodes=1):
        if update_every_n_episodes > 0:
            ind = sum(self.episode_length[:update_every_n_episodes])
            del self.pooled_node_embs[:ind]  # clear experience
            del self.Ts[:ind]
            del self.share_obs[:ind]
            del self.obs[:ind]
            del self.value_preds[:ind]
            del self.returns[:ind]
            del self.actions[:ind]
            del self.action_log_probs[:ind]
            del self.rewards[:ind]
            del self.masks[:ind]
            del self.action_masks[:ind]
            if len(self.demo_act_probs) > 0:
                del self.demo_act_probs[:ind]
            del self.episode_length[:update_every_n_episodes]
        else:
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
            del self.action_masks[:]
            if len(self.demo_act_probs) > 0:
                del self.demo_act_probs[:]
            del self.episode_length[:]

    def compute_returns(self, next_value, value_normalizer=None, cache=False):
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
        if cache:
            self.returns_cache.extend(self.returns_one_episode)
        else:
            self.returns.extend(self.returns_one_episode)
        del self.value_preds_one_episode[:]
        del self.rewards_one_episode[:]
        del self.returns_one_episode[:]
        del self.masks_one_episode[:]

    def get_batch(self, advantages, device, indices=None):
        total_transition_num = len(self.pooled_node_embs) * self.num_agent
        if indices is None:
            batch_size = min(total_transition_num, self.batch_size)
            rand = torch.randperm(total_transition_num).numpy()
            indices = rand[:batch_size]
        else:
            indices = np.asarray(indices)
        pooled_node_embs_batch = torch.FloatTensor(np.array([self.pooled_node_embs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        Ts_batch = torch.FloatTensor(np.array([self.Ts[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        if self.args.use_emb_layer:
            share_obs_batch = torch.LongTensor(np.array([self.share_obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
            obs_batch = torch.LongTensor(np.array([self.obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        else:
            share_obs_batch = torch.FloatTensor(np.array([self.share_obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
            obs_batch = torch.FloatTensor(np.array([self.obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        action_masks_batch = torch.BoolTensor(
            np.array([self.action_masks[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])
        ).to(device)
        actions_batch = torch.FloatTensor(np.array([self.actions[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        value_preds_batch = torch.FloatTensor(np.array([self.value_preds[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        return_batch = torch.FloatTensor(np.array([self.returns[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        masks_batch = torch.FloatTensor(np.array([self.masks[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        old_action_log_probs_batch = torch.FloatTensor(np.array([self.action_log_probs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = torch.FloatTensor(np.array([advantages[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        if len(self.demo_act_probs) > 0:
            demo_act_probs_batch = torch.FloatTensor(np.array([self.demo_act_probs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        else:
            demo_act_probs_batch = None
        return (
            pooled_node_embs_batch,
            Ts_batch,
            share_obs_batch,
            obs_batch,
            action_masks_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            demo_act_probs_batch,
        )


class SharedReplayBufferEnd2End(object):
    """
    Buffer to store training data.
    """
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
        self.action_masks = []
        self.episode_length = []
        self.value_preds_one_episode = []
        self.rewards_one_episode = []
        self.returns_one_episode = []
        self.masks_one_episode = []

    def insert(self, hgs, Ts, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, action_masks=None, demo_act_probs=None):
        if action_masks is None:
            action_masks = np.ones((obs.shape[0], self.act_space), dtype=bool)
        self.share_obs.append(share_obs.copy())
        self.obs.append(obs.copy())
        self.hgs.append(hgs)
        self.Ts.append(Ts.copy())
        self.value_preds.append(value_preds.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.masks.append(masks.copy())
        self.action_masks.append(action_masks.copy())
        self.value_preds_one_episode.append(value_preds.copy())
        self.rewards_one_episode.append(rewards.copy())
        self.returns_one_episode.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
        self.masks_one_episode.append(masks.copy())
        if demo_act_probs is not None:
            self.demo_act_probs.append(demo_act_probs.copy())

    def after_update(self, update_every_n_episodes=1):
        ind = sum(self.episode_length[:update_every_n_episodes])
        del self.hgs[:ind]  # clear experience
        del self.Ts[:ind]
        del self.share_obs[:ind]
        del self.obs[:ind]
        del self.value_preds[:ind]
        del self.returns[:ind]
        del self.actions[:ind]
        del self.action_log_probs[:ind]
        del self.rewards[:ind]
        del self.masks[:ind]
        del self.action_masks[:ind]
        if len(self.demo_act_probs) > 0:
            del self.demo_act_probs[:ind]
        del self.episode_length[:update_every_n_episodes]

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

    def get_batch(self, advantages, device, indices=None):
        total_transition_num = len(self.share_obs) * self.num_agent
        if indices is None:
            batch_size = min(total_transition_num, self.batch_size)
            rand = torch.randperm(total_transition_num).numpy()
            indices = rand[:batch_size]
        else:
            indices = np.asarray(indices)
        graphs = [self.hgs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices]
        hgs_batch = dgl.batch(graphs)
        hgs_batch = hgs_batch.to(device)
        Ts_batch = torch.FloatTensor(np.array([self.Ts[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        if self.args.use_emb_layer:
            share_obs_batch = torch.LongTensor(np.array([self.share_obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
            obs_batch = torch.LongTensor(np.array([self.obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        else:
            share_obs_batch = torch.FloatTensor(np.array([self.share_obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
            obs_batch = torch.FloatTensor(np.array([self.obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        action_masks_batch = torch.BoolTensor(
            np.array([self.action_masks[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])
        ).to(device)
        actions_batch = torch.FloatTensor(np.array([self.actions[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        value_preds_batch = torch.FloatTensor(np.array([self.value_preds[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        return_batch = torch.FloatTensor(np.array([self.returns[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        masks_batch = torch.FloatTensor(np.array([self.masks[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        old_action_log_probs_batch = torch.FloatTensor(np.array([self.action_log_probs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = torch.FloatTensor(np.array([advantages[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        if len(self.demo_act_probs) > 0:
            demo_act_probs_batch = torch.FloatTensor(np.array([self.demo_act_probs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        else:
            demo_act_probs_batch = None
        return (
            hgs_batch,
            Ts_batch,
            share_obs_batch,
            obs_batch,
            action_masks_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            demo_act_probs_batch,
        )


class SharedReplayBufferFT(object):
    """
    Buffer to store training data.
    """
    def __init__(self, mappo_args, args, share_obs_shape, obs_shape, act_space, num_agent):
        self.args = args
        self.batch_size = int(getattr(args, "minibatch_size", 32))
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
        self.action_masks = []
        self.episode_length = []
        self.value_preds_one_episode = []
        self.rewards_one_episode = []
        self.returns_one_episode = []
        self.masks_one_episode = []

    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, demo_act_probs=None, pooled_node_emb=None, action_masks=None):
        if action_masks is None:
            action_masks = np.ones((obs.shape[0], self.act_space), dtype=bool)
        self.share_obs.append(share_obs.copy())
        self.obs.append(obs.copy())
        self.value_preds.append(value_preds.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.masks.append(masks.copy())
        self.action_masks.append(action_masks.copy())
        self.returns.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
        if demo_act_probs is not None:
            self.demo_act_probs.append(demo_act_probs.copy())
        if pooled_node_emb is not None:
            self.pooled_node_emb.append(pooled_node_emb.copy())

    def after_update(self, update_every_n_episodes=1):
        del self.share_obs[:]
        del self.obs[:]
        del self.value_preds[:]
        del self.returns[:]
        del self.actions[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.masks[:]
        del self.action_masks[:]
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

    def get_batch(self, advantages, device, indices=None):
        total_transition_num = len(self.share_obs) * self.num_agent
        if indices is None:
            rand = torch.randperm(total_transition_num).numpy()
            indices = rand[:self.batch_size]
        else:
            indices = np.asarray(indices)
        if self.args.use_emb_layer:
            share_obs_batch = torch.LongTensor(np.array([self.share_obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
            obs_batch = torch.LongTensor(np.array([self.obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        else:
            share_obs_batch = torch.FloatTensor(np.array([self.share_obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
            obs_batch = torch.FloatTensor(np.array([self.obs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        actions_batch = torch.FloatTensor(np.array([self.actions[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        value_preds_batch = torch.FloatTensor(np.array([self.value_preds[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        return_batch = torch.FloatTensor(np.array([self.returns[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        masks_batch = torch.FloatTensor(np.array([self.masks[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        action_masks_batch = torch.BoolTensor(
            np.array([self.action_masks[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])
        ).to(device)
        old_action_log_probs_batch = torch.FloatTensor(np.array([self.action_log_probs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = torch.FloatTensor(np.array([advantages[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        if len(self.demo_act_probs) > 0:
            demo_act_probs_batch = torch.FloatTensor(np.array([self.demo_act_probs[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        else:
            demo_act_probs_batch = None
        if len(self.pooled_node_emb) > 0:
            pooled_node_embs_batch = torch.FloatTensor(np.array([self.pooled_node_emb[int(ind // self.num_agent)][int(ind % self.num_agent)] for ind in indices])).to(device)
        else:
            pooled_node_embs_batch = None
        return (
            share_obs_batch,
            obs_batch,
            action_masks_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            demo_act_probs_batch,
            pooled_node_embs_batch,
        )
