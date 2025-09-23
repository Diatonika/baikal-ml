from torch.utils.data import Dataset as TorchDataset

from baikal.ml.torch.dataset.arrow_time_series_dataset import ArrowTimeSeriesSample


class MockArrowTimeSeriesDataset(TorchDataset[ArrowTimeSeriesSample]):
    def __init__(self, sample: ArrowTimeSeriesSample, length: int) -> None:
        self._sample = sample
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, _: int) -> ArrowTimeSeriesSample:
        return self._sample
