import numpy as np

from mlnumpy.abc.activation import Activation
from mlnumpy.abc.initializer import Initializer
from mlnumpy.abc.layer import Layer
from mlnumpy.abc.optimizer import Optimizer
from mlnumpy.activations.identity import Identity
from mlnumpy.initializers.xavier import XavierInitializer
from mlnumpy.initializers.zero import ZeroInitializer


class Dense(Layer):
    def __init__(
        self,
        size: int,
        activation: Activation | None = None,
        initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
        use_biases: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.activation = activation or Identity()
        self.initializer = initializer or XavierInitializer()
        self.bias_initializer = bias_initializer or ZeroInitializer()
        self.use_biases = use_biases

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"activation={self.activation.__class__.__name__}, "
            f"initializer={self.initializer.__class__.__name__}, "
            f"bias_initializer={self.bias_initializer.__class__.__name__}, "
            f"use_bias={self.use_biases}"
            ")"
        )

    def train(self) -> None:
        super().train()
        self.activation.train()

    def eval(self) -> None:
        super().eval()
        self.activation.eval()

    def setup(
        self,
        input_shape: tuple[int, ...],
        optimizer: Optimizer,
        return_errors: bool = True,
    ) -> tuple[int, ...]:
        self.optimizer = optimizer
        self.return_errors = return_errors

        self.weights = self.initializer((input_shape[0], self.size))
        if self.use_biases:
            self.biases = self.bias_initializer((1, self.size))

        return (self.size,)

    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        self.features = features

        outputs = features @ self.weights
        if self.use_biases:
            outputs += self.biases

        outputs = self.activation(outputs)
        return outputs

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        if not self.train_mode:
            raise ValueError("Backward pass in eval mode is forbidden")

        gradients = self.activation.backward(errors)

        if self.return_errors:
            errors = gradients @ self.weights.T

        self.optimizer(self.weights, self.features.T @ gradients)
        if self.use_biases:
            self.optimizer(self.biases, np.sum(gradients, axis=0, keepdims=True))

        return errors
