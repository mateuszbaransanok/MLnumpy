from unittest import TestCase

import numpy as np

from mlnumpy.losses.ce import CrossEntropy


class TestCrossEntropy(TestCase):
    def test(self) -> None:
        criterion = CrossEntropy()
        targets = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        predictions = np.array([[0.2, 0.3, 0.6], [0.4, 0.5, 0.1], [0.3, 0.3, 0.4]])
        expected_errors = np.array(
            [[-0.416, -0.476, 0.555], [0.833, -0.666, -0.370], [-0.476, 1.111, -0.555]]
        )
        expected_loss = 0.541

        actual_errors, actual_loss = criterion(targets, predictions)

        np.testing.assert_almost_equal(actual_errors, expected_errors, decimal=3)
        np.testing.assert_almost_equal(actual_loss, expected_loss, decimal=3)
