from unittest import TestCase

import numpy as np

from mlnumpy.activations.identity import Identity


class TestIdentity(TestCase):
    def setUp(self) -> None:
        self.activation = Identity()

    def test_forward(self) -> None:
        features = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])
        expected = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])

        actual = self.activation(features)

        np.testing.assert_almost_equal(actual, expected)

    def test_backward(self) -> None:
        features = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])
        errors = np.array([[0.1, -0.15, 0.4, -0.05, 0.25]])
        expected = np.array([[0.1, -0.15, 0.4, -0.05, 0.25]])
        self.activation(features)

        actual = self.activation.backward(errors)

        np.testing.assert_almost_equal(actual, expected)
