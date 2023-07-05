from abc import ABC, abstractmethod

from mlnumpy.abc.module import Module
from mlnumpy.abc.optimizer import Optimizer


class Layer(Module, ABC):
    @abstractmethod
    def setup(
        self,
        input_shape: tuple[int, ...],
        optimizer: Optimizer,
        return_errors: bool = True,
    ) -> tuple[int, ...]:
        pass
