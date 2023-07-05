import numpy as np

from mlnumpy.abc.initializer import Initializer


class XavierInitializer(Initializer):
    def __call__(
        self,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        fan_in, fan_out = self._compute_fans(shape)
        r = np.sqrt(6 / (fan_in + fan_out))
        weight = np.random.uniform(-r, r, size=shape)
        return weight
