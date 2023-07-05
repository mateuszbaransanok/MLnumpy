from pathlib import Path

from mlnumpy.abc.callback import Callback
from mlnumpy.abc.model import Model


class BestModelSaverCallback(Callback):
    def __init__(
        self,
        path: Path,
        metric: str,
        stage: str = "val",
        mode: str = "max",
    ) -> None:
        self.path = path
        self.metric = metric
        self.stage = stage
        self.best = -float("inf") if mode == "max" else float("inf")
        self.mode = -1 if mode == "max" else 1

    def on_epoch_end(
        self,
        model: Model,
    ) -> None:
        if (
            model.stage == self.stage
            and model.epoch_metrics[self.metric] * self.mode < self.best * self.mode
        ):
            self.best = model.epoch_metrics[self.metric]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving best model {self.stage}/{self.metric}={self.best} to {self.path}")
            model.save(self.path)
