from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from .algorithm_base import AlgorithmBase


class AgentBase(ABC):
    """
    Interface for policies encapsulating action selection and value estimation.
    Agents stay independent from environment implementations and can be paired
    with different algorithms.
    """

    def __init__(self) -> None:
        self._algorithm: Optional["AlgorithmBase"] = None

    def bind_algorithm(self, algorithm: "AlgorithmBase") -> None:
        """Attach an algorithm responsible for updating this agent."""
        self._algorithm = algorithm

    def unbind_algorithm(self) -> None:
        """Detach the currently bound algorithm, if any."""
        self._algorithm = None

    def algorithm(self) -> Optional["AlgorithmBase"]:
        """Return the currently bound algorithm or None."""
        return self._algorithm

    @abstractmethod
    def act(self, observation: Any, state: Any = None, **kwargs) -> tuple[Any, Any]:
        """
        Produce an action (or action distribution) and updated recurrent state.
        The return contract should be stable across runners and algorithms.
        """

    @abstractmethod
    def value(self, observation: Any, state: Any = None, **kwargs) -> Any:
        """
        Estimate the state-value or action-value given observation and state.
        """

    @abstractmethod
    def learnable_params(self) -> dict[str, Any]:
        """
        Return parameters eligible for optimization (e.g., PyTorch parameters).
        """

    def reset_state(self) -> None:
        """Reset any internal recurrent or exploration state."""

    def save(self, path: str) -> None:
        """Persist agent parameters to disk."""

    def load(self, path: str) -> None:
        """Load agent parameters from disk."""
