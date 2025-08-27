from collections.abc import Iterable

from attrs import define
from pyarrow import Table as ArrowTable
from torch.utils.data import Dataset as TorchDataset

from baikal.ml.torch.dataset.window_strategy import WindowStrategy


@define
class ArrowTimeSeriesSample:
    features: ArrowTable
    targets: ArrowTable


class ArrowTimeSeriesDataset(TorchDataset[ArrowTimeSeriesSample]):
    def __init__(
        self,
        dataset: ArrowTable,
        windows: WindowStrategy,
        *,
        feature_window: int,
        target_window: int,
        features: Iterable[str],
        targets: Iterable[str],
    ) -> None:
        self._dataset = dataset
        self._windows = windows

        self._feature_window = feature_window
        self._target_window = target_window

        self._features = tuple(features)
        self._targets = tuple(targets)

    def __len__(self) -> int:
        return self._windows.length()

    def __getitem__(self, index: int) -> ArrowTimeSeriesSample:
        merged_window = self._dataset.slice(
            self._windows.window(index), self._feature_window + self._target_window
        )

        feature_window = merged_window.slice(0, self._feature_window)
        target_window = merged_window.slice(self._feature_window, self._target_window)

        return ArrowTimeSeriesSample(
            features=feature_window.select(self._features),
            targets=target_window.select(self._targets),
        )
