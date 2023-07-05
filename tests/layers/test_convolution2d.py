from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from mlnumpy.abc.initializer import Initializer
from mlnumpy.layers.convolution2d import Convolution2D


class _WeightInitializer(Initializer):
    def __call__(
        self,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        return np.arange(0, np.prod(shape)).reshape(shape)


class TestConvolution2D(TestCase):
    def setUp(self) -> None:
        self.layer = Convolution2D(
            filters=1,
            kernel=(3, 3),
            kernel_initializer=_WeightInitializer(),
            bias_initializer=_WeightInitializer(),
        )

    def test_setup(self) -> None:
        input_shape = (4, 4, 3)
        expected = (2, 2, 1)

        actual = self.layer.setup(
            input_shape=input_shape,
            optimizer=Mock(),
            return_errors=True,
        )

        self.assertEqual(expected, actual)

    def test_forward(self) -> None:
        features = np.array(
            [
                [0.25, -0.1, 0.1, 0],
                [0.05, 0.2, -0.5, -0.15],
                [0.2, -0.3, -0.05, 0.2],
                [0.35, -0.1, 0, 0.25],
            ]
        ).reshape((1, 4, 4, 1))
        expected = np.array(
            [
                [-2.75, -2.6],
                [-0.25, 0.5],
            ]
        ).reshape((1, 2, 2, 1))
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=Mock(),
            return_errors=True,
        )

        actual = self.layer(features)

        np.testing.assert_almost_equal(actual, expected)

    def test_backward(self) -> None:
        features = np.array(
            [
                [0.25, -0.1, 0.1, 0],
                [0.05, 0.2, -0.5, -0.15],
                [0.2, -0.3, -0.05, 0.2],
                [0.35, -0.1, 0, 0.25],
            ]
        ).reshape((1, 4, 4, 1))
        errors = np.array(
            [
                [-1.25, 0.6],
                [0.25, 0.35],
            ]
        ).reshape((1, 2, 2, 1))
        expected = np.array(
            [
                [0.0, -1.25, -1.9, 1.2],
                [-3.75, -2.95, -3.0, 3.7],
                [-6.75, -3.1, -3.15, 6.55],
                [1.5, 3.85, 4.45, 2.8],
            ]
        ).reshape((1, 4, 4, 1))
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=Mock(),
            return_errors=True,
        )
        self.layer(features)

        actual = self.layer.backward(errors)

        np.testing.assert_almost_equal(actual, expected)
