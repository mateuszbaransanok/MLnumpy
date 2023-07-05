from abc import ABC, abstractmethod

import numpy as np

from mlnumpy.abc.layer import Layer
from mlnumpy.abc.optimizer import Optimizer
from mlnumpy.utils.padding import add_padding, compute_output_shape, remove_padding


class Pooling2D(Layer, ABC):
    def __init__(
        self,
        pool_size: tuple[int, int],
        stride: tuple[int, int],
        pad: tuple[int, int],
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad

    def setup(
        self,
        input_shape: tuple[int, ...],
        optimizer: Optimizer,
        return_errors: bool = True,
    ) -> tuple[int, ...]:
        nh = compute_output_shape(input_shape[0], self.pool_size[0], self.pad[0], self.stride[0])
        nw = compute_output_shape(input_shape[1], self.pool_size[1], self.pad[1], self.stride[1])

        self.error_shape = (-1, nh, nw, 1, 1, input_shape[2])
        self.output_shape = (nh, nw, input_shape[2])
        return self.output_shape

    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        self.features = features
        features_padded = add_padding(features, self.pad[0], self.pad[1])

        features_expanded = np.lib.stride_tricks.as_strided(
            features_padded,
            shape=(
                features_padded.shape[0],
                self.output_shape[0],
                self.output_shape[1],
                self.pool_size[0],
                self.pool_size[1],
                self.output_shape[2],
            ),
            strides=(
                features_padded.strides[0],
                features_padded.strides[1] * self.stride[0],
                features_padded.strides[2] * self.stride[1],
                features_padded.strides[1],
                features_padded.strides[2],
                features_padded.strides[3],
            ),
            writeable=False,
        )

        outputs = self._pooling(features_expanded)

        return outputs

    def backward(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        mask = self.mask * errors.reshape(self.error_shape)  # type: ignore[attr-defined]

        errors = np.zeros_like(self.features)
        errors = add_padding(errors, self.pad[0], self.pad[1])

        expanded_error = np.lib.stride_tricks.as_strided(
            errors,
            shape=(
                errors.shape[0],
                self.output_shape[0],
                self.output_shape[1],
                self.pool_size[0],
                self.pool_size[1],
                self.output_shape[2],
            ),
            strides=(
                errors.strides[0],
                errors.strides[1] * self.stride[0],
                errors.strides[2] * self.stride[1],
                errors.strides[1],
                errors.strides[2],
                errors.strides[3],
            ),
            writeable=True,
        )

        np.add.at(expanded_error, (), mask)

        errors = remove_padding(errors, self.pad[0], self.pad[1])

        return errors

    @abstractmethod
    def _pooling(
        self,
        features_expanded: np.ndarray,
    ) -> np.ndarray:
        pass


class MaxPooling2D(Pooling2D):
    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        stride: tuple[int, int] = (2, 2),
        pad: tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__(
            pool_size=pool_size,
            stride=stride,
            pad=pad,
        )

    def _pooling(
        self,
        features_expanded: np.ndarray,
    ) -> np.ndarray:
        features = np.max(features_expanded, axis=(-3, -2), keepdims=True)
        features_ = features_expanded + np.random.uniform(0, 1e-8, size=features_expanded.shape)
        self.mask = features_ == np.max(features_, axis=(-3, -2), keepdims=True)
        return np.squeeze(features, axis=(-3, -2))


class AveragePooling2D(Pooling2D):
    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        stride: tuple[int, int] = (2, 2),
        pad: tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__(
            pool_size=pool_size,
            stride=stride,
            pad=pad,
        )

    def _pooling(
        self,
        features_expanded: np.ndarray,
    ) -> np.ndarray:
        features = np.mean(features_expanded, axis=(-3, -2), keepdims=True)
        self.mask = np.ones_like(features_expanded) / (self.pool_size[0] * self.pool_size[1])
        return np.squeeze(features, axis=(-3, -2))
