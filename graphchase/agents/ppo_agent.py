from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
from graphchase.interfaces.agent_base import AgentBase


class PPOAgent(AgentBase, nn.Module):
    """
    Policy/value network that consumes pre-encoded observations (tensor) and
    outputs logits over a fixed action space size. Observation preprocessing and
    legal-action masking are handled outside the agent (e.g., in a runner).
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        device: str | torch.device = "cpu",
    ) -> None:
        AgentBase.__init__(self)
        nn.Module.__init__(self)
        self.device = torch.device(device)
        self.actor = nn.Sequential(
            self._layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.to(self.device)

    @staticmethod
    def _layer_init(layer: nn.Module, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=std)
            nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_action_and_value(
        self,
        encoded_obs: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        action_map: list[list[int] | None] | None = None,
        action_index: int | None = None,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, int]:
        encoded = encoded_obs.to(self.device)
        logits = self.actor(encoded)
        if legal_mask is not None:
            legal_mask = legal_mask.to(self.device)
            if legal_mask.shape != logits.shape:
                raise ValueError(f"legal_mask shape {legal_mask.shape} does not match logits shape {logits.shape}")
            if not legal_mask.any():
                raise ValueError("No valid actions available under provided legal_mask.")
            logits = logits.masked_fill(~legal_mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        if action_index is None:
            action_index = int(dist.sample().item())
        logprob = dist.log_prob(torch.tensor(action_index, device=self.device))
        entropy = dist.entropy()
        value = self.value(encoded)
        if action_map is not None:
            action = action_map[action_index]
        else:
            action = action_index
        return action, logprob, entropy, value, action_index

    def get_action_and_value_batch(
        self,
        encoded_obs_batch: torch.Tensor,
        legal_mask_batch: torch.Tensor | None = None,
        action_map: list[list[int] | None] | None = None,
        action_index_batch: torch.Tensor | None = None,
    ) -> tuple[list[Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = encoded_obs_batch.to(self.device)
        logits = self.actor(encoded)
        if legal_mask_batch is not None:
            legal_mask_batch = legal_mask_batch.to(self.device)
            if legal_mask_batch.shape != logits.shape:
                raise ValueError(
                    f"legal_mask_batch shape {legal_mask_batch.shape} does not match logits shape {logits.shape}"
                )
            if not legal_mask_batch.any(dim=1).all():
                raise ValueError("Some batch entries have no valid actions under provided legal_mask_batch.")
            logits = logits.masked_fill(~legal_mask_batch, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        if action_index_batch is None:
            action_index_batch = dist.sample()
        else:
            action_index_batch = action_index_batch.to(self.device)
        logprob_batch = dist.log_prob(action_index_batch)
        entropy_batch = dist.entropy()
        value_batch = self.value(encoded)
        action_indices_cpu = action_index_batch.detach().cpu().tolist()
        if action_map is not None:
            actions = [action_map[int(idx)] for idx in action_indices_cpu]
        else:
            actions = [int(idx) for idx in action_indices_cpu]
        return actions, logprob_batch, entropy_batch, value_batch, action_index_batch

    def act(self, observation: Any, state: Any = None, **kwargs) -> tuple[Any, Any]:
        legal_mask = kwargs.get("legal_mask")
        action_map = kwargs.get("action_map")
        if not isinstance(observation, torch.Tensor):
            raise ValueError("PPOAgent.act expects pre-encoded observation tensor.")
        action, _, _, _, _ = self.get_action_and_value(
            encoded_obs=observation,
            legal_mask=legal_mask,
            action_map=action_map,
        )
        return action, state

    def value(self, observation: Any, state: Any = None, **kwargs) -> torch.Tensor:
        if not isinstance(observation, torch.Tensor):
            raise ValueError("PPOAgent.value expects pre-encoded observation tensor.")
        encoded = observation.to(self.device)
        value = self.critic(encoded)
        return value.squeeze(-1)

    def learnable_params(self) -> dict[str, Any]:
        return {"params": list(self.parameters())}

    def reset_state(self) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)
