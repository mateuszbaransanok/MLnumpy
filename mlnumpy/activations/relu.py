import numpy as np

from mlnumpy.abc.activation import Activation


class ReLU(Activation):
    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        self.output_layer = np.where(features > 0, features, 0)
        return self.output_layer

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        return np.where(self.output_layer > 0, 1, 0) * errors
