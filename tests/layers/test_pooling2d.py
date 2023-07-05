from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from mlnumpy.layers.pooling2d import AveragePooling2D, MaxPooling2D


class TestMaxPooling2D(TestCase):
    def setUp(self) -> None:
        self.layer = MaxPooling2D(
            pool_size=(2, 2),
        )

    def test_setup(self) -> None:
        input_shape = (4, 4, 3)
        expected = (2, 2, 3)

        actual = self.layer.setup(
            input_shape=input_shape,
            optimizer=Mock(),
            return_errors=True,
        )

        self.assertEqual(expected, actual)

    def test_forward(self) -> None:
        features = np.array(
            [
                [0.25, -0.1, 0.1, 0],
                [0.05, 0.2, -0.5, -0.15],
                [0.2, -0.3, -0.05, 0.2],
                [0.35, -0.1, 0, 0.25],
            ]
        ).reshape((1, 4, 4, 1))
        expected = np.array(
            [
                [0.25, 0.1],
                [0.35, 0.25],
            ]
        ).reshape((1, 2, 2, 1))
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=Mock(),
            return_errors=True,
        )

        actual = self.layer(features)

        np.testing.assert_almost_equal(actual, expected)

    def test_backward(self) -> None:
        features = np.array(
            [
                [0.25, -0.1, 0.1, 0],
                [0.05, 0.2, -0.5, -0.15],
                [0.2, -0.3, -0.05, 0.2],
                [0.35, -0.1, 0, 0.25],
            ]
        ).reshape((1, 4, 4, 1))
        errors = np.array(
            [
                [-1.25, 0.6],
                [0.25, 0.35],
            ]
        ).reshape((1, 2, 2, 1))
        expected = np.array(
            [
                [-1.25, 0.0, 0.6, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0],
                [0.25, 0.0, 0.0, 0.35],
            ]
        ).reshape((1, 4, 4, 1))
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=Mock(),
            return_errors=True,
        )
        self.layer(features)

        actual = self.layer.backward(errors)

        np.testing.assert_almost_equal(actual, expected)


class TestAveragePooling2D(TestCase):
    def setUp(self) -> None:
        self.layer = AveragePooling2D(
            pool_size=(2, 2),
        )

    def test_setup(self) -> None:
        input_shape = (4, 4, 3)
        expected = (2, 2, 3)

        actual = self.layer.setup(
            input_shape=input_shape,
            optimizer=Mock(),
            return_errors=True,
        )

        self.assertEqual(expected, actual)

    def test_forward(self) -> None:
        features = np.array(
            [
                [0.25, -0.1, 0.1, 0],
                [0.05, 0.2, -0.5, -0.15],
                [0.2, -0.3, -0.05, 0.2],
                [0.35, -0.1, 0, 0.25],
            ]
        ).reshape((1, 4, 4, 1))
        expected = np.array(
            [
                [0.1, -0.1375],
                [0.0375, 0.1],
            ]
        ).reshape((1, 2, 2, 1))
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=Mock(),
            return_errors=True,
        )

        actual = self.layer(features)

        np.testing.assert_almost_equal(actual, expected)

    def test_backward(self) -> None:
        features = np.array(
            [
                [0.25, -0.1, 0.1, 0],
                [0.05, 0.2, -0.5, -0.15],
                [0.2, -0.3, -0.05, 0.2],
                [0.35, -0.1, 0, 0.25],
            ]
        ).reshape((1, 4, 4, 1))
        errors = np.array(
            [
                [-1.25, 0.6],
                [0.25, 0.35],
            ]
        ).reshape((1, 2, 2, 1))
        expected = np.array(
            [
                [-0.3125, -0.3125, 0.15, 0.15],
                [-0.3125, -0.3125, 0.15, 0.15],
                [0.0625, 0.0625, 0.0875, 0.0875],
                [0.0625, 0.0625, 0.0875, 0.0875],
            ]
        ).reshape((1, 4, 4, 1))
        self.layer.setup(
            input_shape=features.shape[1:],
            optimizer=Mock(),
            return_errors=True,
        )
        self.layer(features)

        actual = self.layer.backward(errors)

        np.testing.assert_almost_equal(actual, expected)
