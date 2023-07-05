from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from mlnumpy.layers.flatten import Flatten


class TestFlatten(TestCase):
    def setUp(self) -> None:
        self.layer = Flatten()
        self.optimizer = Mock()

    def test_setup(self) -> None:
        input_shape = (4, 3, 2)
        expected = (24,)

        actual = self.layer.setup(
            input_shape=input_shape,
            optimizer=self.optimizer(),
            return_errors=True,
        )

        self.assertEqual(expected, actual)

    def test_forward(self) -> None:
        features = np.array([[[0.25, -0.2], [-0.3, 0.1]]])
        expected = np.array([[0.25, -0.2, -0.3, 0.1]])
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=self.optimizer(),
            return_errors=True,
        )

        actual = self.layer(features)

        np.testing.assert_almost_equal(actual, expected)

    def test_backward(self) -> None:
        features = np.array([[[0.25, -0.2], [-0.3, 0.1]]])
        errors = np.array([[0.1, -0.15, 0.4, -0.05]])
        expected = np.array([[[0.1, -0.15], [0.4, -0.05]]])
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=self.optimizer(),
            return_errors=True,
        )
        self.layer(features)

        actual = self.layer.backward(errors)

        np.testing.assert_almost_equal(actual, expected)
