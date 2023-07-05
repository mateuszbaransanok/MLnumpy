import numpy as np

from mlnumpy.abc.initializer import Initializer


class NormalInitializer(Initializer):
    def __init__(
        self,
        loc: int,
        scale: int,
        factor: int,
    ) -> None:
        self.loc = loc
        self.scale = scale
        self.factor = factor

    def __call__(
        self,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        fan_in, fan_out = self._compute_fans(shape)
        weight = np.random.normal(self.loc, self.scale, size=shape)
        weight = weight * np.sqrt(self.factor / fan_out)
        return weight
