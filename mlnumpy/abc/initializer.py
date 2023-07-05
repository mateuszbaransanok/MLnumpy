from abc import ABC, abstractmethod

import numpy as np


class Initializer(ABC):
    @abstractmethod
    def __call__(
        self,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        pass

    @staticmethod
    def _compute_fans(shape: tuple[int, ...]) -> tuple[int, int]:
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out
