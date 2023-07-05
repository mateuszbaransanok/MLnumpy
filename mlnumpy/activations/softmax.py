import numpy as np

from mlnumpy.abc.activation import Activation


class Softmax(Activation):
    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        exps = np.exp(features - np.max(features))
        self.output_layer = np.clip(
            exps / np.sum(exps, axis=1, keepdims=True), 0.00000001, 0.99999999
        )
        return self.output_layer

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        expanded = np.expand_dims(self.output_layer, axis=1)
        diag = expanded * np.eye(self.output_layer.shape[1])
        jac = diag - np.einsum("mij,mkj->mik", diag, diag)
        return np.einsum("mj,mkj->mk", errors, jac)
