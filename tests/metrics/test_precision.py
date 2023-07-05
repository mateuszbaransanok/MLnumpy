from unittest import TestCase

import numpy as np

from mlnumpy.metrics.precision import Precision


class TestPrecision(TestCase):
    def test__macro(self) -> None:
        metric = Precision(3, "macro")
        targets = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        predictions = np.array(
            [
                [0.2, 0.3, 0.6],
                [0.2, 0.5, 0.3],
                [0.2, 0.3, 0.6],
                [0.5, 0.4, 0.1],
                [0.4, 0.3, 0.3],
            ]
        )
        expected = 0.5

        actual = metric(targets, predictions)

        np.testing.assert_almost_equal(actual, expected, decimal=3)

    def test__micro(self) -> None:
        metric = Precision(3, "micro")
        targets = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        predictions = np.array(
            [
                [0.2, 0.3, 0.6],
                [0.2, 0.5, 0.3],
                [0.2, 0.3, 0.6],
                [0.5, 0.4, 0.1],
                [0.4, 0.3, 0.3],
            ]
        )
        expected = 0.6

        actual = metric(targets, predictions)

        np.testing.assert_almost_equal(actual, expected, decimal=3)
