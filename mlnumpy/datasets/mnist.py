from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from mlnumpy import PROJECT_DIR
from mlnumpy.abc.dataset import Dataset


class MNIST(Dataset):
    URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

    def __init__(
        self,
        path: Path = PROJECT_DIR / "data/mnist",
    ) -> None:
        self._path = path
        self._load_data()

    def num_classes(self) -> int:
        return 10

    def input_shape(self) -> tuple[int, ...]:
        return 28, 28, 1

    def train_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._train_features, self._train_targets

    def val_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._val_features, self._val_targets

    def test_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._test_features, self._test_targets

    def _load_data(self) -> None:
        self._path.mkdir(exist_ok=True, parents=True)

        path = self._path.joinpath("mnist.npz")
        if not path.exists():
            urlretrieve(self.URL, path)
            print(f"Downloaded MNIST to {path}")

        with np.load(str(path)) as data:
            _train_features = self._normalize(data["x_train"])
            _train_targets = self._onehot(data["y_train"])

            self._train_features = _train_features[:50000]
            self._train_targets = _train_targets[:50000]
            self._val_features = _train_features[50000:]
            self._val_targets = _train_targets[50000:]
            self._test_features = self._normalize(data["x_test"])
            self._test_targets = self._onehot(data["y_test"])

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        return np.expand_dims(features, 3) / 255

    @staticmethod
    def _onehot(targets: np.ndarray) -> np.ndarray:
        return np.eye(np.max(targets) + 1)[targets]
