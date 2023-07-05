import numpy as np

from mlnumpy.abc.activation import Activation
from mlnumpy.abc.loss import Loss
from mlnumpy.activations.identity import Identity


class SoftmaxCrossEntropy(Loss):
    def __call__(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        targets = np.argmax(targets, axis=1)
        m = targets.shape[0]

        exps = np.exp(predictions - np.max(predictions))
        predictions = exps / np.sum(exps, axis=1, keepdims=True)

        epsilon = 1e-10  # small constant to avoid division by zero
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)

        log_likelihood = -np.log(predictions[np.arange(m), targets])
        loss = float(np.mean(log_likelihood))

        error = np.array(predictions)
        error[range(m), targets] -= 1
        error = -error / m

        return error, loss

    def validate(
        self,
        activation: Activation,
    ) -> bool:
        return isinstance(activation, Identity)
