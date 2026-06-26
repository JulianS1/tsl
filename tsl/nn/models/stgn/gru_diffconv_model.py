import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from tsl.nn import utils
from tsl.nn.models.base_model import BaseModel


class DilatedConvBlock(nn.Module):
    """Single dilated conv layer: Conv1d → BatchNorm → ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DiffConvStream(nn.Module):
    """Stack of dilated convolutions with exponentially growing dilation rates."""

    def __init__(self, in_channels: int, hidden_size: int, kernel_size: int = 3,
                 num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_size
            layers.append(
                DilatedConvBlock(in_ch, hidden_size, kernel_size, dilation=2**i))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, in_channels)
        x = x.permute(0, 2, 1)   # (batch, channels, seq_len)
        x = self.layers(x)
        return x.permute(0, 2, 1)  # (batch, seq_len, hidden_size)


class GRUStream(nn.Module):
    """Stacked GRU returning all timestep hidden states."""

    def __init__(self, in_channels: int, hidden_size: int, num_layers: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional)
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        return out


class GRUDiffConvModel(BaseModel):
    """GRU + Dilated Convolution temporal backbone for spatiotemporal forecasting.

    Both streams consume the same node features independently, their outputs
    are concatenated, normalised, and projected to the forecast horizon.
    Each node is processed in parallel by merging the batch and node dimensions.

    Args:
        input_size (int): Number of input features per node.
        hidden_size (int): Internal width of both GRU and DiffConv streams.
        output_size (int): Number of output features per node.
        horizon (int): Forecasting horizon (number of future steps).
        exog_size (int): Number of exogenous input features.
            (default: :obj:`0`)
        n_layers (int): Number of stacked GRU layers.
            (default: :obj:`2`)
        conv_layers (int): Number of dilated conv layers.
            (default: :obj:`3`)
        kernel_size (int): Kernel size for all dilated conv layers.
            (default: :obj:`3`)
        bidirectional (bool): Whether the GRU reads both directions.
            (default: :obj:`False`)
    """

    return_type = Tensor

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 32,
                 output_size: int = 1,
                 horizon: int = 12,
                 exog_size: int = 0,
                 n_layers: int = 2,
                 conv_layers: int = 3,
                 kernel_size: int = 3,
                 bidirectional: bool = False):
        super(GRUDiffConvModel, self).__init__()

        in_ch = input_size + exog_size
        self.gru_stream = GRUStream(in_ch, hidden_size, n_layers, bidirectional)
        self.conv_stream = DiffConvStream(in_ch, hidden_size, kernel_size,
                                          conv_layers)

        fused_dim = self.gru_stream.out_dim + hidden_size
        self.norm = nn.LayerNorm(fused_dim)
        self.proj = nn.Linear(fused_dim, horizon * output_size)

        self._horizon = horizon
        self._output_size = output_size

    def forward(self, x: Tensor, edge_index=None, edge_weight=None,
                u=None) -> Tensor:
        """"""
        # x: [batch, time, nodes, features]
        x = utils.maybe_cat_exog(x, u)
        b, t, n, f = x.shape

        # process all nodes in parallel
        x = rearrange(x, 'b t n f -> (b n) t f')

        gru_out = self.gru_stream(x)    # [b*n, t, gru_dim]
        conv_out = self.conv_stream(x)  # [b*n, t, hidden_size]

        fused = torch.cat([gru_out, conv_out], dim=-1)  # [b*n, t, fused_dim]
        fused = self.norm(fused)

        # project from last timestep to forecast horizon
        out = self.proj(fused[:, -1])   # [b*n, horizon * output_size]
        return rearrange(out, '(b n) (h f) -> b h n f',
                         b=b, n=n, h=self._horizon, f=self._output_size)
