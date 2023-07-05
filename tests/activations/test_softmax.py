from unittest import TestCase

import numpy as np

from mlnumpy.activations.softmax import Softmax


class TestSoftmax(TestCase):
    def setUp(self) -> None:
        self.activation = Softmax()

    def test_forward(self) -> None:
        features = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])
        expected = np.array([[0.2512485, 0.1602031, 0.1449578, 0.2162516, 0.227339]])

        actual = self.activation(features)

        np.testing.assert_almost_equal(actual, expected)

    def test_backward(self) -> None:
        features = np.array([[0.25, -0.2, -0.3, 0.1, 0.15]])
        errors = np.array([[0.1, -0.15, 0.4, -0.05, 0.25]])
        expected = np.array([[0.0188123, -0.0201807, 0.049578, -0.0084743, 0.043914]])
        self.activation(features)

        actual = self.activation.backward(errors)

        np.testing.assert_almost_equal(actual, expected)
