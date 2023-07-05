from unittest import TestCase

import numpy as np

from mlnumpy.optimizers.adadelta import AdaDelta


class TestAdaDelta(TestCase):
    def test(self) -> None:
        weights = np.array([0.2, -0.1, 1.0])
        gradients_list = [
            np.array([0, 0.1, -0.5]),
            np.array([-0.2, 0.1, 0]),
            np.array([-0.1, -0.2, -0.1]),
        ]
        expected = np.array([0.1993, -0.0997, 0.9994])
        optimizer = AdaDelta()

        for gradients in gradients_list:
            optimizer(weights, gradients)

        np.testing.assert_almost_equal(weights, expected, decimal=4)
