from unittest import TestCase

import numpy as np

from mlnumpy.initializers.zero import ZeroInitializer


class TestZeroInitializer(TestCase):
    def test(self) -> None:
        initializer = ZeroInitializer()
        expected = np.zeros((200, 100))

        actual = initializer((200, 100))

        np.testing.assert_equal(actual, expected)
