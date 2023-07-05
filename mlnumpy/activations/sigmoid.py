import numpy as np

from mlnumpy.abc.activation import Activation


class Sigmoid(Activation):
    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        self.output_layer = 1 / (1 + np.exp(-features))
        return self.output_layer

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        return self.output_layer * (1 - self.output_layer) * errors
