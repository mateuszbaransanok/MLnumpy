from unittest import TestCase

import numpy as np

from mlnumpy.activations.tanh import Tanh


class TestTanh(TestCase):
    def setUp(self) -> None:
        self.activation = Tanh()

    def test_forward(self) -> None:
        features = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])
        expected = np.array([[0.2449187, -0.1973753, -0.2913126, 0.099668, 0.148885]])

        actual = self.activation(features)

        np.testing.assert_almost_equal(actual, expected)

    def test_backward(self) -> None:
        features = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])
        errors = np.array([[0.1, -0.15, 0.4, -0.05, 0.25]])
        expected = np.array([[0.0942335, -0.1443049, 0.3678868, -0.0495066, 0.2445392]])
        self.activation(features)

        actual = self.activation.backward(errors)

        np.testing.assert_almost_equal(actual, expected)
