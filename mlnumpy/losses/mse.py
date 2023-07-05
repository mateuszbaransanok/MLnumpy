import numpy as np

from mlnumpy.abc.activation import Activation
from mlnumpy.abc.loss import Loss
from mlnumpy.activations.identity import Identity
from mlnumpy.activations.relu import ReLU
from mlnumpy.activations.sigmoid import Sigmoid
from mlnumpy.activations.softmax import Softmax
from mlnumpy.activations.tanh import Tanh


class MeanSquaredError(Loss):
    def __call__(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        error = targets - predictions
        loss = float(np.mean(np.square(error)))
        error = error / targets.shape[0]

        return error, loss

    def validate(
        self,
        activation: Activation,
    ) -> bool:
        return isinstance(activation, (ReLU, Identity, Tanh, Sigmoid, Softmax))
