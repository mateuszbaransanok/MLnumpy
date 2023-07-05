import numpy as np

from mlnumpy.abc.activation import Activation


class Tanh(Activation):
    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        self.output_layer = np.tanh(features)
        return self.output_layer

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        return (1 - np.tanh(self.output_layer) ** 2) * errors
