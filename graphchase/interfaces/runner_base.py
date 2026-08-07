from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import logging
from typing import Any, Callable, Optional

from .agent_base import AgentBase
from .algorithm_base import AlgorithmBase
from .br_mixin import BRMixin, BRSolverBase


class RunnerBase(BRMixin, ABC):
    """
    Interface for environment interaction loops. Runners orchestrate collection,
    evaluation, and delegation to algorithms for parameter updates.
    """

    def __init__(
        self,
        env_builder: Callable[..., Any],
        agent: AgentBase,
        algorithm: AlgorithmBase,
        br_solver: Optional[BRSolverBase] = None,
        config: Optional[dict[str, Any]] = None,
        metrics_logger: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    ) -> None:
        self.env_builder = env_builder
        self.agent = agent
        self.algorithm = algorithm
        self.config = config or {}
        self.metrics_logger = metrics_logger
        super().__init__(br_solver=br_solver)

    def __deepcopy__(self, memo: dict[int, Any]):
        """
        Share metrics_logger across clones to avoid spawning new logging
        backends (e.g., wandb runs) during PSRO iterations.
        """
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for key, value in self.__dict__.items():
            if key == "metrics_logger":
                setattr(clone, key, value)
            else:
                setattr(clone, key, copy.deepcopy(value, memo))
        return clone

    def set_agent(self, agent: AgentBase) -> None:
        """Replace the active agent with a compatible implementation."""
        self.agent = agent

    def set_algorithm(self, algorithm: AlgorithmBase) -> None:
        """Replace the active algorithm with a compatible implementation."""
        self.algorithm = algorithm

    @abstractmethod
    def collect(self, **kwargs) -> dict[str, Any]:
        """
        Interact with the environment to gather trajectories or transitions
        based on the current agent policy. Returns a batch for updates.
        """

    @abstractmethod
    def evaluate(self, **kwargs) -> dict[str, Any]:
        """Run evaluation episodes and return aggregated metrics."""

    @abstractmethod
    def run(self, **kwargs) -> None:
        """
        Main training or execution loop. Typically alternates between collect
        and algorithm.update steps, plus logging and checkpointing.
        """

    def close(self) -> None:
        """Optional cleanup hook for environments or background workers."""

    def log_metrics(self, metrics: dict[str, Any], **context: Any) -> None:
        """
        Forward metrics to the provided logger if available. Keeps runner agnostic
        to metric names so different algorithms can report custom fields.
        """
        if not metrics or self.metrics_logger is None:
            return
        payload_context = {k: v for k, v in context.items() if v is not None}
        try:
            self.metrics_logger(metrics, payload_context)
        except Exception as exc:  # best-effort logging; do not break training
            logging.getLogger(__name__).warning("Failed to log metrics: %s", exc)
