from unittest import TestCase

import numpy as np

from mlnumpy.activations.sigmoid import Sigmoid


class TestSigmoid(TestCase):
    def setUp(self) -> None:
        self.activation = Sigmoid()

    def test_forward(self) -> None:
        features = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])
        expected = np.array([[0.5621765, 0.450166, 0.4255575, 0.5249792, 0.5374298]])

        actual = self.activation(features)

        np.testing.assert_almost_equal(expected, actual)

    def test_backward(self) -> None:
        features = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])
        errors = np.array([[0.1, -0.15, 0.4, -0.05, 0.25]])
        expected = np.array([[0.0246134, -0.0371275, 0.0977833, -0.0124688, 0.0621498]])
        self.activation(features)

        actual = self.activation.backward(errors)

        np.testing.assert_almost_equal(expected, actual)
