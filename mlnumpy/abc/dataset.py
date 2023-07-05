from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def train_data(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def val_data(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def test_data(self) -> tuple[np.ndarray, np.ndarray]:
        pass
