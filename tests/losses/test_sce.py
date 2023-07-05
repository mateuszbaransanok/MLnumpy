from unittest import TestCase

import numpy as np

from mlnumpy.losses.sce import SoftmaxCrossEntropy


class TestSoftmaxCrossEntropy(TestCase):
    def test(self) -> None:
        criterion = SoftmaxCrossEntropy()
        targets = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        predictions = np.array([[0.2, 0.3, 0.6], [0.4, 0.5, 0.1], [0.3, 0.3, 0.4]])
        expected_errors = np.array(
            [[-0.093, -0.102, 0.195], [0.216, -0.129, -0.087], [-0.107, 0.226, -0.119]]
        )
        expected_loss = 1.019

        actual_errors, actual_loss = criterion(targets, predictions)

        np.testing.assert_almost_equal(actual_errors, expected_errors, decimal=3)
        np.testing.assert_almost_equal(actual_loss, expected_loss, decimal=3)
