from unittest import TestCase

import numpy as np

from mlnumpy.initializers.normal import NormalInitializer


class TestNormalInitializer(TestCase):
    def test(self) -> None:
        initializer = NormalInitializer(loc=3, scale=2, factor=1)

        actual = initializer((400, 100))

        np.testing.assert_almost_equal(actual.mean(), 0.3, decimal=2)
        np.testing.assert_almost_equal(actual.std(), 0.2, decimal=2)
