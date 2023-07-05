from unittest import TestCase

import numpy as np

from mlnumpy.optimizers.adam import Adam


class TestAdam(TestCase):
    def test(self) -> None:
        weights = np.array([0.2, -0.1, 1.0])
        gradients_list = [
            np.array([0, 0.1, -0.5]),
            np.array([-0.2, 0.1, 0]),
            np.array([-0.1, -0.2, -0.1]),
        ]
        expected = np.array([0.046, 0.092, 0.77])
        optimizer = Adam(
            learning_rate=0.1,
        )

        for gradients in gradients_list:
            optimizer(weights, gradients)

        np.testing.assert_almost_equal(weights, expected, decimal=3)
