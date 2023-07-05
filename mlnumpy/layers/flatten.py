import numpy as np

from mlnumpy.abc.layer import Layer
from mlnumpy.abc.optimizer import Optimizer


class Flatten(Layer):
    def setup(
        self,
        input_shape: tuple[int, ...],
        optimizer: Optimizer,
        return_errors: bool = True,
    ) -> tuple[int, ...]:
        self.input_shape = input_shape
        self.output_shape = (int(np.prod(input_shape)),)
        self.return_errors = return_errors
        return self.output_shape

    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return features.reshape(-1, *self.output_shape)

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        if self.return_errors:
            return errors.reshape(-1, *self.input_shape)
        return errors
