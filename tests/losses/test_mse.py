from unittest import TestCase

import numpy as np

from mlnumpy.losses.mse import MeanSquaredError


class TestMeanSquaredError(TestCase):
    def test(self) -> None:
        criterion = MeanSquaredError()
        targets = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        predictions = np.array([[0.2, 0.3, 0.6], [0.4, 0.5, 0.1], [0.3, 0.3, 0.4]])
        expected_errors = np.array(
            [[-0.067, -0.1, 0.133], [0.2, -0.167, -0.033], [-0.1, 0.233, -0.133]]
        )
        expected_loss = 0.183

        actual_errors, actual_loss = criterion(targets, predictions)

        np.testing.assert_almost_equal(actual_errors, expected_errors, decimal=3)
        np.testing.assert_almost_equal(actual_loss, expected_loss, decimal=3)
