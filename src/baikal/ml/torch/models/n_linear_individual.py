from torch import Tensor, empty, nn


class NLinearIndividual(nn.Module):
    def __init__(
        self, lookback_window: int, forecast_window: int, channel_count: int
    ) -> None:
        super().__init__()

        self._forecast_window = forecast_window
        self._channel_count = channel_count

        self._channels = nn.ModuleList()
        for _ in range(channel_count):
            self._channels.append(nn.Linear(lookback_window, forecast_window))

    def forward(self, x: Tensor) -> Tensor:
        normalization_value = x[:, -1:, :]
        normalized_x = x - normalization_value

        output = empty(
            (x.size(0), self._forecast_window, self._channel_count),
            dtype=x.dtype,
            device=x.device,
        )

        for i, linear in enumerate(self._channels):
            output[:, :, i] = linear(normalized_x[:, :, i])

        return output + normalization_value
