from datetime import date

from polars import DataFrame, date_range

from baikal.ml.torch.dataset import StrideWindowStrategy


def test_stride_window_strategy_one_gap() -> None:
    date_time_series = date_range(
        date(2020, 1, 1),
        date(2020, 1, 31),
        "1d",
        closed="both",
        eager=True,
    ).alias("date_time")

    dataset = DataFrame([date_time_series]).remove(date_time=date(2020, 1, 15))

    strategy = StrideWindowStrategy(
        dataset.rechunk().to_arrow(),
        "date_time",
        frequency="1d",
        feature_window=5,
        target_window=5,
        stride=1,
    )

    mapping = [strategy.window(index) for index in range(strategy.length())]
    assert mapping == [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20]


def test_stride_window_strategy_two_gaps() -> None:
    date_time_series = date_range(
        date(2020, 1, 1),
        date(2020, 1, 31),
        "1d",
        closed="both",
        eager=True,
    ).alias("date_time")

    dataset = (
        DataFrame([date_time_series])
        .remove(date_time=date(2020, 1, 10))
        .remove(date_time=date(2020, 1, 20))
    )

    strategy = StrideWindowStrategy(
        dataset.rechunk().to_arrow(),
        "date_time",
        frequency="1d",
        feature_window=4,
        target_window=4,
        stride=1,
    )

    mapping = [strategy.window(index) for index in range(strategy.length())]
    assert mapping == [0, 1, 9, 10, 18, 19, 20, 21]


def test_stride_window_strategy_empty() -> None:
    date_time_series = date_range(
        date(2020, 1, 1),
        date(2020, 1, 31),
        "1d",
        closed="both",
        eager=True,
    ).alias("date_time")

    dataset = (
        DataFrame([date_time_series])
        .remove(date_time=date(2020, 1, 10))
        .remove(date_time=date(2020, 1, 20))
        .remove(date_time=date(2020, 1, 30))
    )

    strategy = StrideWindowStrategy(
        dataset.rechunk().to_arrow(),
        "date_time",
        frequency="1d",
        feature_window=5,
        target_window=5,
        stride=1,
    )

    mapping = [strategy.window(index) for index in range(strategy.length())]
    assert not len(mapping)


def test_stride_window_strategy_stride() -> None:
    date_time_series = date_range(
        date(2020, 1, 1),
        date(2020, 1, 31),
        "1d",
        closed="both",
        eager=True,
    ).alias("date_time")

    dataset = DataFrame([date_time_series])

    strategy = StrideWindowStrategy(
        dataset.rechunk().to_arrow(),
        "date_time",
        frequency="1d",
        feature_window=5,
        target_window=5,
        stride=5,
    )

    mapping = [strategy.window(index) for index in range(strategy.length())]
    assert mapping == [0, 5, 10, 15, 20]


def test_stride_window_strategy_stride_with_gap() -> None:
    date_time_series = date_range(
        date(2020, 1, 1),
        date(2020, 1, 31),
        "1d",
        closed="both",
        eager=True,
    ).alias("date_time")

    dataset = (
        DataFrame([date_time_series])
        .remove(date_time=date(2020, 1, 14))
        .remove(date_time=date(2020, 1, 15))
        .remove(date_time=date(2020, 1, 16))
    )

    strategy = StrideWindowStrategy(
        dataset.rechunk().to_arrow(),
        "date_time",
        frequency="1d",
        feature_window=5,
        target_window=5,
        stride=5,
    )

    mapping = [strategy.window(index) for index in range(strategy.length())]
    assert mapping == [0, 13, 18]


def test_stride_window_strategy_boundaries() -> None:
    date_time_series = date_range(
        date(2020, 1, 1),
        date(2020, 1, 31),
        "1d",
        closed="both",
        eager=True,
    ).alias("date_time")

    dataset = DataFrame([date_time_series])

    strategy = StrideWindowStrategy(
        dataset.rechunk().to_arrow(),
        "date_time",
        frequency="1d",
        feature_window=5,
        target_window=5,
        stride=1,
        start=date(2020, 1, 3),
        end=date(2020, 1, 25),
    )

    mapping = [strategy.window(index) for index in range(strategy.length())]
    assert mapping == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
