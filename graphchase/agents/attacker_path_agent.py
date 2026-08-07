from __future__ import annotations

from typing import Any
from graphchase.interfaces.agent_base import AgentBase


class PathAgent(AgentBase):
    """
    Deterministic attacker agent following a predefined node path.
    Assumes a single attacker; additional attackers will hold position (action 0).
    """

    def __init__(self, path: list[int], num_attackers: int = 1) -> None:
        super().__init__()
        self.path = list(path)
        self.num_attackers = num_attackers
        self._cursor = 0

    def act(self, observation: Any, state: Any = None, **kwargs) -> tuple[list[int], Any]:
        actions = [0 for _ in range(self.num_attackers)]
        for i in range(self.num_attackers):
            next_action = self._next_action()
            actions[i] = next_action
        return actions, state

    def _next_action(self) -> int:
        if self._cursor >= len(self.path) - 1:
            return 0
        action = int(self.path[self._cursor + 1])
        self._cursor += 1
        return action

    def reset_state(self) -> None:
        self._cursor = 0

    def value(self, observation: Any, state: Any = None, **kwargs) -> Any:
        return 0.0

    def learnable_params(self) -> dict[str, Any]:
        return {}
