from unittest import TestCase

import numpy as np

from mlnumpy.initializers.xavier import XavierInitializer


class TestXavierInitializer(TestCase):
    def test(self) -> None:
        initializer = XavierInitializer()

        actual = initializer((400, 200))

        np.testing.assert_almost_equal(np.max(actual), 0.1, decimal=2)
        np.testing.assert_almost_equal(np.min(actual), -0.1, decimal=2)
