from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AlgorithmBase(ABC):
    """
    Interface for optimization logic updating agent parameters based on batches
    of collected experience. Algorithms do not interact with the environment
    directly; they consume batches prepared by runners.
    """

    def prepare_step(self, global_step: int) -> None:
        """
        Optional hook for schedulers or bookkeeping before an update iteration.
        """

    def build_buffers(self, config: dict[str, Any]) -> Any:
        """
        Optional factory for replay or rollout buffers used during collection.
        """

    @abstractmethod
    def update(self, batch: dict[str, Any], agent: Any, **kwargs) -> dict[str, Any]:
        """
        Apply one or more optimization steps given a batch and the target agent.
        Return diagnostic metrics for logging.
        """
