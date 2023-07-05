import numpy as np

from mlnumpy.abc.activation import Activation
from mlnumpy.abc.initializer import Initializer
from mlnumpy.abc.layer import Layer
from mlnumpy.abc.optimizer import Optimizer
from mlnumpy.activations.identity import Identity
from mlnumpy.initializers.xavier import XavierInitializer
from mlnumpy.initializers.zero import ZeroInitializer
from mlnumpy.utils.padding import add_padding, compute_output_shape, remove_padding


class Convolution2D(Layer):
    def __init__(
        self,
        filters: int,
        kernel: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        pad: tuple[int, int] = (0, 0),
        activation: Activation | None = None,
        kernel_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
        use_biases: bool = True,
    ) -> None:
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.use_biases = use_biases
        self.activation = activation or Identity()
        self.initializer = kernel_initializer or XavierInitializer()
        self.bias_initializer = bias_initializer or ZeroInitializer()

    def setup(
        self,
        input_shape: tuple[int, ...],
        optimizer: Optimizer,
        return_errors: bool = True,
    ) -> tuple[int, ...]:
        shape = (self.kernel[0], self.kernel[1], input_shape[2], self.filters)

        self.weights = self.initializer(shape)
        self.biases = self.bias_initializer((1, 1, 1, self.filters))
        self.optimizer = optimizer
        self.return_errors = return_errors

        nh = compute_output_shape(input_shape[0], self.kernel[0], self.pad[0], self.stride[0])
        nw = compute_output_shape(input_shape[1], self.kernel[1], self.pad[1], self.stride[1])

        self.input_shape = input_shape
        self.output_shape = (nh, nw, self.filters)
        return self.output_shape

    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        self.features = features
        features = add_padding(features, self.pad[0], self.pad[1])

        features_expanded = np.lib.stride_tricks.as_strided(
            features,
            shape=(
                features.shape[0],
                self.output_shape[0],
                self.output_shape[1],
                self.weights.shape[0],
                self.weights.shape[1],
                self.weights.shape[2],
            ),
            strides=(
                features.strides[0],
                features.strides[1] * self.stride[0],
                features.strides[2] * self.stride[1],
                features.strides[1],
                features.strides[2],
                features.strides[3],
            ),
            writeable=False,
        )

        outputs = np.einsum("MHWhwc,hwcf->MHWf", features_expanded, self.weights)
        if self.use_biases:
            outputs += self.biases

        outputs = self.activation(outputs)
        return outputs

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        gradients = self.activation.backward(errors)

        gradients = np.insert(
            arr=gradients,
            obj=np.repeat(np.arange(1, gradients.shape[1]), self.stride[0] - 1),
            values=0,
            axis=1,
        )
        gradients = np.insert(
            arr=gradients,
            obj=np.repeat(np.arange(1, gradients.shape[2]), self.stride[1] - 1),
            values=0,
            axis=2,
        )

        if self.return_errors:
            weight = np.rot90(self.weights, 2, axes=(0, 1))
            gradients_padded = add_padding(gradients, weight.shape[0] - 1, weight.shape[1] - 1)

            gradients_expanded = np.lib.stride_tricks.as_strided(
                gradients_padded,
                shape=(
                    gradients_padded.shape[0],
                    self.input_shape[0],
                    self.input_shape[1],
                    weight.shape[0],
                    weight.shape[1],
                    weight.shape[3],
                ),
                strides=(
                    gradients_padded.strides[0],
                    gradients_padded.strides[1],
                    gradients_padded.strides[2],
                    gradients_padded.strides[1],
                    gradients_padded.strides[2],
                    gradients_padded.strides[3],
                ),
                writeable=False,
            )

            errors = np.einsum("MHWhwf,hwcf->MHWc", gradients_expanded, weight)
            errors = remove_padding(errors, self.pad[0], self.pad[1])

        features_padded = add_padding(self.features, self.pad[0], self.pad[1])

        features_expanded = np.lib.stride_tricks.as_strided(
            features_padded,
            shape=(
                self.weights.shape[0],
                self.weights.shape[1],
                self.weights.shape[2],
                gradients.shape[0],
                gradients.shape[1],
                gradients.shape[2],
            ),
            strides=(
                features_padded.strides[1],
                features_padded.strides[2],
                features_padded.strides[3],
                features_padded.strides[0],
                features_padded.strides[1],
                features_padded.strides[2],
            ),
            writeable=False,
        )

        self.optimizer(self.weights, np.einsum("HWcMhw,Mhwf->HWcf", features_expanded, gradients))
        if self.use_biases:
            self.optimizer(self.biases, np.sum(gradients, axis=(0, 1, 2), keepdims=True))

        return errors
