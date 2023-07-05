from mlnumpy.abc.callback import Callback
from mlnumpy.abc.model import Model


class MetricLogCallback(Callback):
    def on_batch_end(
        self,
        model: Model,
    ) -> None:
        log = f"EPOCH={model.current_epoch + model.current_step / model.total_epoch_steps:.3f} | "
        for metric, value in model.batch_metrics.items():
            log += f"{model.stage}/{metric}: {value:.5f} | "

        print(f"\r{log}", end="")

    def on_epoch_end(
        self,
        model: Model,
    ) -> None:
        log = f"EPOCH={model.current_epoch + 1} | "
        for metric, value in model.epoch_metrics.items():
            log += f"{model.stage}/{metric}: {value:.5f} | "

        print(f"\r{log}")
