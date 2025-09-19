from collections.abc import Sequence

from attrs import define
from torch import Tensor, stack
from torch.utils.dlpack import from_dlpack

from baikal.ml.torch.dataset.arrow_time_series_dataset import ArrowTimeSeriesSample


@define
class ArrowTimeSeriesBatch:
    features: Tensor
    targets: Tensor

    @staticmethod
    def from_samples(
        samples: Sequence[ArrowTimeSeriesSample],
    ) -> "ArrowTimeSeriesBatch":
        feature_tensors: list[Tensor] = []
        for sample in samples:
            feature_tensor = stack(
                [
                    from_dlpack(feature.combine_chunks())
                    for feature in sample.features.itercolumns()
                ],
                dim=1,
            )

            feature_tensors.append(feature_tensor)

        target_tensors: list[Tensor] = []
        for sample in samples:
            target_tensor = stack(
                [
                    from_dlpack(target.combine_chunks())
                    for target in sample.targets.itercolumns()
                ],
                dim=1,
            )

            target_tensors.append(target_tensor)

        return ArrowTimeSeriesBatch(stack(feature_tensors), stack(target_tensors))

    def pin_memory(self) -> "ArrowTimeSeriesBatch":
        self.features = self.features.pin_memory()
        self.targets = self.targets.pin_memory()

        return self
