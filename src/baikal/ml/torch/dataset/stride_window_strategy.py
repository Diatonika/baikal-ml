from datetime import timedelta
from typing import cast

from polars import (
    DataFrame,
    Duration,
    Expr,
    String,
    all as polar_all,
    col,
    from_arrow,
    int_range,
    len as polar_len,
    lit,
)
from pyarrow import Table as ArrowTable

from baikal.ml.torch.dataset.window_strategy import WindowStrategy
from baikal.ml.util.polars import as_duration


class StrideWindowStrategy(WindowStrategy):
    def __init__(
        self,
        dataset: ArrowTable,
        date_time_column: str,
        *,
        frequency: timedelta | str,
        feature_window: int,
        target_window: int,
        stride: int,
    ) -> None:
        self._window_mapping = StrideWindowStrategy._build_window_mapping(
            dataset,
            date_time_column,
            frequency=frequency,
            feature_window=feature_window,
            target_window=target_window,
            stride=stride,
        )

    def length(self) -> int:
        return self._window_mapping.height

    def window(self, index: int) -> int:
        return cast(int, self._window_mapping.item(index, "index"))

    # region Private

    @staticmethod
    def _build_window_mapping(
        dataset: ArrowTable,
        date_time_column: str,
        *,
        frequency: timedelta | str,
        feature_window: int,
        target_window: int,
        stride: int,
    ) -> DataFrame:
        time_series_frame = cast(
            DataFrame, from_arrow(dataset.select([date_time_column]), rechunk=False)
        )

        window_length = feature_window + target_window - 1

        window_duration: Expr
        if isinstance(frequency, str):
            window_duration = as_duration(lit(frequency, dtype=String)) * window_length
        else:
            window_duration = lit(frequency, dtype=Duration) * window_length

        return (
            time_series_frame.set_sorted(date_time_column)
            .lazy()
            .select(
                int_range(polar_len()).alias("index"),
                col(date_time_column).alias("start"),
                col(date_time_column).add(window_duration).alias("end"),
                col(date_time_column).shift(-window_length).alias("__shift__"),
            )
            .with_columns(col("end").eq_missing(col("__shift__")).alias("__valid__"))
            .with_columns(col("__valid__").rle_id().alias("__sequence__"))
            .filter(col("__valid__"))
            .select(
                polar_all()
                .gather_every(stride)
                .over("__sequence__", mapping_strategy="explode")
            )
            .drop("^__.*__$")
            .collect()
        )

    # endregion
