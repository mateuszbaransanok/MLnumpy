import numpy as np

from mlnumpy.abc.activation import Activation


class Identity(Activation):
    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return features

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        return errors
