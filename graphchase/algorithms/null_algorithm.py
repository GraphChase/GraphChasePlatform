from __future__ import annotations

from typing import Any
from graphchase.interfaces.algorithm_base import AlgorithmBase


class NullAlgorithm(AlgorithmBase):
    """
    Placeholder algorithm that performs no updates. Useful when a runner
    requires an AlgorithmBase but does not need learning.
    """

    def update(self, batch: dict[str, Any], agent: Any, **kwargs) -> dict[str, Any]:
        return {}
