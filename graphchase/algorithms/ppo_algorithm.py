from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from graphchase.interfaces.algorithm_base import AlgorithmBase


class PPOAlgorithm(AlgorithmBase):
    """
    PPO optimizer for training a single-agent policy with categorical actions.
    Batches are expected as a list of transition dicts collected by a runner.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        critic_learning_rate: float | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        update_epochs: int = 4,
        minibatch_size: int = 32,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_vloss: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_vloss = clip_vloss
        self.device = torch.device(device)
        self.optimizer: optim.Optimizer | None = None
        self.learning_rate = learning_rate
        self.critic_learning_rate = critic_learning_rate if critic_learning_rate is not None else learning_rate

    def _ensure_optimizer(self, agent: Any) -> None:
        if self.optimizer is not None:
            return
        param_groups: list[dict[str, Any]] = []
        seen: set[int] = set()

        actor_params = list(agent.actor.parameters()) if hasattr(agent, "actor") else []
        if actor_params:
            param_groups.append({"params": actor_params, "lr": self.learning_rate})
            seen.update(id(p) for p in actor_params)

        critic_params = list(agent.critic.parameters()) if hasattr(agent, "critic") else []
        critic_params = [p for p in critic_params if id(p) not in seen]
        if critic_params:
            param_groups.append({"params": critic_params, "lr": self.critic_learning_rate})
            seen.update(id(p) for p in critic_params)

        remaining_params = [p for p in agent.parameters() if id(p) not in seen]
        if remaining_params:
            param_groups.append({"params": remaining_params, "lr": self.learning_rate})

        if not param_groups:
            raise ValueError("No parameters available for optimizer initialization.")

        self.optimizer = optim.Adam(param_groups, lr=self.learning_rate, eps=1e-5)

    def update(self, batch: list[dict[str, Any]], agent: Any, **kwargs) -> dict[str, Any]:
        if not batch:
            return {}
        self._ensure_optimizer(agent)
        device = self.device
        actions_tensor = torch.tensor([int(item["action_index"]) for item in batch], device=device)
        old_logprobs = torch.tensor([float(item["logprob"]) for item in batch], device=device)
        rewards = torch.tensor([float(item["reward"]) for item in batch], device=device)
        dones = torch.tensor([float(item["done"]) for item in batch], device=device)
        values = torch.tensor([float(item["value"]) for item in batch], device=device)
        next_values = torch.tensor([float(item.get("next_value", 0.0)) for item in batch], device=device)
        legal_masks = [item.get("legal_mask") for item in batch]
        observations = [item["obs"] for item in batch]

        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = 0.0
        for t in reversed(range(len(batch))):
            next_non_terminal = 1.0 - dones[t]
            next_val = next_values[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        batch_size = len(batch)
        indices = np.arange(batch_size)
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.minibatch_size):
                end = min(start + self.minibatch_size, batch_size)
                mb_idx = indices[start:end]

                mb_actions = actions_tensor[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_values = values[mb_idx]
                mb_legal_masks = [legal_masks[i] for i in mb_idx]
                mb_obs = [observations[i] for i in mb_idx]

                can_batch = not any(mask is None for mask in mb_legal_masks)
                if can_batch:
                    obs_batch = torch.stack(mb_obs).to(device)
                    legal_mask_batch = torch.stack(mb_legal_masks).to(device)
                    _, new_logprobs, entropy, new_values, _ = agent.get_action_and_value_batch(
                        encoded_obs_batch=obs_batch,
                        legal_mask_batch=legal_mask_batch,
                        action_map=None,
                        action_index_batch=mb_actions,
                    )
                    new_logprobs = new_logprobs.to(device)
                    entropy = entropy.to(device)
                    new_values = new_values.to(device).view(-1)
                else:
                    new_logprobs_list = []
                    entropy_list = []
                    new_values_list = []
                    for obs_entry, action_idx, legal_mask in zip(mb_obs, mb_actions, mb_legal_masks):
                        obs_tensor = obs_entry.to(device)
                        action_list, new_logprob, entropy_val, new_value, _ = agent.get_action_and_value(
                            encoded_obs=obs_tensor,
                            legal_mask=legal_mask.to(device) if legal_mask is not None else None,
                            action_index=int(action_idx.item()),
                        )
                        new_logprobs_list.append(new_logprob)
                        entropy_list.append(entropy_val)
                        new_values_list.append(new_value)
                    new_logprobs = torch.stack(new_logprobs_list).to(device)
                    entropy = torch.stack(entropy_list).to(device)
                    new_values = torch.stack(new_values_list).to(device).view(-1)

                log_ratio = new_logprobs - mb_old_logprobs
                ratio = log_ratio.exp()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                if self.clip_vloss:
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(new_values - mb_values, -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss - self.entropy_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = ((ratio - 1) - log_ratio).mean()
                policy_losses.append(policy_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())

        metrics = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }
        return metrics
