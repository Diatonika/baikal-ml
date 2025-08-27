from collections.abc import Iterable

from pyarrow import Table as ArrowTable
from torch.utils.data import Dataset as TorchDataset

from baikal.ml.torch.dataset.structures import TSSample
from baikal.ml.torch.dataset.window_strategy import WindowStrategy


class ArrowTorchDataset(TorchDataset[TSSample]):
    def __init__(
        self,
        dataset: ArrowTable,
        windows: WindowStrategy,
        *,
        feature_window: int,
        target_window: int,
        absolute_features: Iterable[str],
        relative_features: Iterable[str],
        target_features: Iterable[str],
    ) -> None:
        self._dataset = dataset
        self._windows = windows

        self._feature_window = feature_window
        self._target_window = target_window

        self._absolute_features = tuple(absolute_features)
        self._relative_features = tuple(relative_features)
        self._target_features = tuple(target_features)

    def __len__(self) -> int:
        return self._windows.length()

    def __getitem__(self, index: int) -> TSSample:
        merged_window = self._dataset.slice(
            self._windows.window(index), self._feature_window + self._target_window
        )

        feature_window = merged_window.slice(0, self._feature_window)
        target_window = merged_window.slice(self._feature_window, self._target_window)

        return TSSample(
            absolute_features=feature_window.select(self._absolute_features),
            relative_features=feature_window.select(self._relative_features),
            target_features=target_window.select(self._target_features),
        )
