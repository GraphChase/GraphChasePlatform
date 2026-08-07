from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .algorithm_base import AlgorithmBase


class BRSolverBase(ABC):
    """
    Interface for best-response solvers. Implementations may use planning,
    heuristic search, or short-horizon RL training to approximate BR policies.
    """

    @abstractmethod
    def solve(
        self,
        env_builder: Any,
        opponent_policy: Any,
        role: str,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Compute or approximate a best response against the opponent policy for
        the specified role. Returns solver outputs such as policy artifacts,
        value estimates, or diagnostics.
        """


class BRAlgorithmBase(AlgorithmBase):
    """
    Optional extension of AlgorithmBase for BR-specific optimization when BR is
    approximated via RL. Can be used to drive training loops inside a solver.
    """

    @abstractmethod
    def update_br(
        self,
        batch: dict[str, Any],
        br_agent: Any,
        opponent_policy: Any,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform a BR-focused update step."""


class BRMixin:
    """
    Mixin for runners to expose BR computation without constraining runner
    implementations. Runners using this mixin should set `self.br_solver` and
    provide access to `self.env_builder`.
    """

    def __init__(self, br_solver: Optional[BRSolverBase] = None) -> None:
        self.br_solver = br_solver

    def set_br_solver(self, br_solver: BRSolverBase) -> None:
        """Attach a BR solver implementation."""
        self.br_solver = br_solver

    def compute_best_response(
        self,
        opponent_policy: Any,
        role: str,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Delegate BR computation to the configured solver. Runners can override
        to customize orchestration or metrics.
        """
        if self.br_solver is None:
            raise NotImplementedError("BR solver is not configured.")
        if not hasattr(self, "env_builder"):
            raise AttributeError("Runner must define env_builder for BR usage.")
        return self.br_solver.solve(self.env_builder, opponent_policy, role, config)
