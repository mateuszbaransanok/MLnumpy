import numpy as np

from mlnumpy.abc.initializer import Initializer


class ZeroInitializer(Initializer):
    def __call__(
        self,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        weight = np.zeros(shape=shape)
        return weight
