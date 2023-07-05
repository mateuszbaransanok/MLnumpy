from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from mlnumpy.abc.initializer import Initializer
from mlnumpy.layers.dense import Dense


class _WeightInitializer(Initializer):
    def __call__(
        self,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        return np.arange(0, np.prod(shape)).reshape(shape)


class TestDense(TestCase):
    def setUp(self) -> None:
        self.layer = Dense(
            size=3,
            initializer=_WeightInitializer(),
            bias_initializer=_WeightInitializer(),
        )

    def test_setup(self) -> None:
        input_shape = (2,)
        expected = (3,)

        actual = self.layer.setup(
            input_shape=input_shape,
            optimizer=Mock(),
            return_errors=True,
        )

        self.assertEqual(expected, actual)

    def test_forward(self) -> None:
        features = np.array([[0.25, -0.2]])
        expected = np.array([[-0.6, 0.45, 1.5]])
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=Mock(),
            return_errors=True,
        )

        actual = self.layer(features)

        np.testing.assert_almost_equal(actual, expected)

    def test_backward(self) -> None:
        features = np.array([[0.25, -0.2]])
        errors = np.array([[0.1, -0.15, 0.05]])
        expected = np.array([[-0.05, -0.05]])
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=Mock(),
            return_errors=True,
        )
        self.layer(features)

        actual = self.layer.backward(errors)

        np.testing.assert_almost_equal(actual, expected)
