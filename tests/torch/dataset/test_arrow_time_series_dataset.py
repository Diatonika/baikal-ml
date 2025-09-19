from pyarrow import Table as ArrowTable

from baikal.ml.torch.dataset import (
    ArrowTimeSeriesDataset,
    ArrowTimeSeriesSample,
    WindowStrategy,
)


class PassThroughWindowStrategy(WindowStrategy):
    def __init__(self, length: int) -> None:
        self._length = length

    def length(self) -> int:
        return self._length

    def window(self, index: int) -> int:
        return index


def test_arrow_torch_dataset() -> None:
    dataset_length = 10

    features = [float(index) for index in range(dataset_length)]
    targets = list(range(dataset_length))

    feature_window = 2
    target_window = 1

    dataset = ArrowTimeSeriesDataset(
        ArrowTable.from_pydict({"feature": features, "target": targets}),
        PassThroughWindowStrategy(dataset_length),
        feature_window=feature_window,
        target_window=target_window,
        features=["feature"],
        targets=["target"],
    )

    for index in range(dataset_length - feature_window - target_window + 1):
        sample = dataset[index]

        assert isinstance(sample, ArrowTimeSeriesSample)

        assert len(sample.features) == feature_window
        assert len(sample.targets) == target_window

        assert sample.features.column_names == ["feature"]
        assert sample.targets.column_names == ["target"]

        assert sample.features.column("feature").to_pylist() == [
            float(i) for i in range(index, index + feature_window)
        ]

        assert sample.targets.column("target").to_pylist() == list(
            range(index + feature_window, index + feature_window + target_window)
        )
