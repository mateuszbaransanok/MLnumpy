import typing
from abc import ABC

if typing.TYPE_CHECKING:
    from mlnumpy.abc.model import Model


class Callback(ABC):
    def on_batch_begin(
        self,
        model: "Model",
    ) -> None:
        pass

    def on_batch_end(
        self,
        model: "Model",
    ) -> None:
        pass

    def on_epoch_begin(
        self,
        model: "Model",
    ) -> None:
        pass

    def on_epoch_end(
        self,
        model: "Model",
    ) -> None:
        pass
