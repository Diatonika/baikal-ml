from attrs import define
from pyarrow import Table as ArrowTable
from torch import Tensor


@define
class TSSample:
    absolute_features: ArrowTable
    relative_features: ArrowTable

    target_features: ArrowTable


@define
class TSSampleBatch:
    absolute_features: Tensor
    relative_features: Tensor

    target_features: Tensor

    def pin_memory(self) -> "TSSampleBatch":
        self.absolute_features = self.absolute_features.pin_memory()
        self.relative_features = self.relative_features.pin_memory()
        self.target_features = self.target_features.pin_memory()

        return self
