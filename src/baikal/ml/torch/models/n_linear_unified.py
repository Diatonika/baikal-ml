from torch import Tensor, nn


class NLinearUnified(nn.Module):
    def __init__(self, lookback_window: int, forecast_window: int) -> None:
        super().__init__()

        self._linear = nn.Linear(lookback_window, forecast_window)

    def forward(self, x: Tensor) -> Tensor:
        normalization_value = x[:, -1:, :]
        normalized_x = x - normalization_value
        output: Tensor = self._linear(normalized_x.permute(0, 2, 1)).permute(0, 2, 1)

        return output + normalization_value
