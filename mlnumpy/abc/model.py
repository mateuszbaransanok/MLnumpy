from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlnumpy.abc.dataset import Dataset
from mlnumpy.abc.module import Module


class Model(Module, ABC):
    stage: str
    current_step: int
    current_global_step: int
    total_epoch_steps: int
    current_epoch: int
    batch_outputs: dict[str, Any]
    epoch_outputs: list[dict[str, Any]]
    batch_metrics: dict[str, Any]
    epoch_metrics: dict[str, Any]

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: Path,
    ) -> "Model":
        pass

    @abstractmethod
    def save(
        self,
        path: Path,
    ) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        dataset: Dataset,
    ) -> None:
        pass
