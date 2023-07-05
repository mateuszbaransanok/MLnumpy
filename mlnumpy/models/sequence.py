import math
import pickle as pkl
from pathlib import Path

import numpy as np

from mlnumpy.abc.callback import Callback
from mlnumpy.abc.dataset import Dataset
from mlnumpy.abc.layer import Layer
from mlnumpy.abc.loss import Loss
from mlnumpy.abc.metric import Metric
from mlnumpy.abc.model import Model
from mlnumpy.abc.optimizer import Optimizer
from mlnumpy.layers.dense import Dense


class SequenceModel(Model):
    def __init__(
        self,
        layers: list[Layer],
        optimizer: Optimizer,
        loss: Loss,
        metrics: dict[str, Metric] | None = None,
        callbacks: list[Callback] | None = None,
        max_epochs: int = 1000,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or {}
        self.callbacks = callbacks or []
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.current_epoch = 0
        self.current_global_step = 0
        self.stage = "train"
        self.is_training = True
        self.is_compiled = False

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layers={self.layers}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"loss={self.loss.__class__.__name__}"
            ")"
        )

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "Model":
        with path.open("rb") as file:
            return pkl.load(file)  # type: ignore[no-any-return]

    def save(
        self,
        path: Path,
    ) -> None:
        with path.open("wb") as file:
            pkl.dump(self, file)

    def train(self) -> None:
        super().train()
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        super().eval()
        for layer in self.layers:
            layer.eval()

    def __call__(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        for layer in self.layers:
            features = layer(features)
        return features

    def backward(
        self,
        errors: np.ndarray,
        return_errors: bool = True,
    ) -> np.ndarray:
        for layer in reversed(self.layers):
            errors = layer.backward(errors)
        return errors

    def fit(
        self,
        dataset: Dataset,
    ) -> None:
        self._compile(dataset)

        features_train, targets_train = dataset.train_data()
        features_val, targets_val = dataset.val_data()
        features_test, targets_test = dataset.test_data()

        while self.is_training:
            features_train, targets_train = self._shuffle(features_train, targets_train)
            self.stage = "train"
            self.train()
            self._run_epoch(features_train, targets_train)

            self.stage = "val"
            self.eval()
            self._run_epoch(features_val, targets_val)

            if self.current_epoch + 1 >= self.max_epochs:
                self.is_training = False
            else:
                self.current_epoch += 1

        self.stage = "test"
        self.eval()
        self._run_epoch(features_test, targets_test)

    def _compile(self, dataset: Dataset) -> None:
        if self.is_compiled:
            raise ValueError("Model is already compiled")

        if not self.layers:
            raise ValueError("Model has not any layer")

        input_shape = dataset.input_shape()
        for i, layer in enumerate(self.layers):
            input_shape = layer.setup(
                input_shape=input_shape,
                optimizer=self.optimizer,
                return_errors=i != 0,
            )

        last_layer = self.layers[-1]
        if not isinstance(last_layer, Dense):
            raise ValueError("Last layer must be a Dense")

        if dataset.num_classes() != last_layer.size:
            raise ValueError(
                f"Incompatible shape targets [{dataset.num_classes()}] and {last_layer.size}"
            )

        if not self.loss.validate(last_layer.activation):
            raise ValueError(f"Loss {self.loss} not supports {last_layer.activation} activation")

        for metric in self.metrics.values():
            metric.setup(
                num_classes=dataset.num_classes(),
            )

        self.is_compiled = True

    def _run_epoch(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

        self.epoch_outputs = []
        self.current_step = 0
        self.dataset_size = features.shape[0]
        self.total_epoch_steps = math.ceil(self.dataset_size / self.batch_size)

        for batch_num in range(0, self.dataset_size, self.batch_size):
            batch_features = features[batch_num : batch_num + self.batch_size]
            batch_targets = targets[batch_num : batch_num + self.batch_size]

            self._run_batch(
                features=batch_features,
                targets=batch_targets,
            )

            self.current_step += 1
            if self.stage == "train":
                self.current_global_step += 1

        self.epoch_metrics = {
            name: metric(
                targets=np.concatenate([output["targets"] for output in self.epoch_outputs]),
                predictions=np.concatenate(
                    [output["predictions"] for output in self.epoch_outputs]
                ),
            )
            for name, metric in self.metrics.items()
        }
        self.epoch_metrics["loss"] = np.mean([output["loss"] for output in self.epoch_outputs])

        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def _run_batch(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(self)

        predictions = self(features)
        errors, loss = self.loss(targets, predictions)

        self.batch_outputs = {
            "targets": targets,
            "predictions": predictions,
            "loss": loss,
        }

        self.batch_metrics = {
            name: metric(
                targets=targets,
                predictions=predictions,
            )
            for name, metric in self.metrics.items()
        }
        self.batch_metrics["loss"] = loss

        if self.train_mode:
            self.backward(errors, return_errors=False)

        for callback in self.callbacks:
            callback.on_batch_end(self)

        self.epoch_outputs.append(self.batch_outputs)

    @staticmethod
    def _shuffle(
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        size = features.shape[0]
        order = np.random.permutation(size)
        return features[order], targets[order]
