import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from tsl.nn import utils
from tsl.nn.layers.graph_convs.diff_conv import DiffConv
from tsl.nn.models.base_model import BaseModel

from edgeScaler import EdgeScaler


class GRUDiffConvModel(BaseModel):
    return_type = Tensor

    def __init__(self, input_size, hidden_size=32, output_size=1, horizon=12,
                 exog_size=0, window_size=12, edge_mode='linear'):
        super().__init__()
        in_ch = input_size + exog_size
        self.gru = nn.GRU(in_ch, hidden_size, batch_first=True)
        self.edge_scaler = EdgeScaler(window_size, in_ch, hidden_size,
                                      scaling_mode=edge_mode)
        self.diffconv = DiffConv(hidden_size, hidden_size, k=1)
        self.readout = nn.Linear(hidden_size, horizon * output_size)
        self._horizon = horizon
        self._output_size = output_size

    def forward(self, x: Tensor, edge_index=None, edge_weight=None,
                u=None) -> Tensor:
        # x: (B, T, N, F)
        x = utils.maybe_cat_exog(x, u)
        B, T, N, F = x.shape

        #Compute modulated edge weights from the input window
        scaled_w = self.edge_scaler.scale(x, edge_index, edge_weight)

        #GRU
        x_in = rearrange(x, 'b t n f -> (b n) t f')
        h, _ = self.gru(x_in)                              # (B*N, T, hidden)
        h = h[:, -1, :]
        h = rearrange(h, '(b n) d -> b n d', b=B, n=N)

        #DiffConv with scaled edge weights
        out = self.diffconv(h, edge_index, scaled_w)

        #Readout to forecast horizon
        out = self.readout(out)                            # (B, N, horizon * output_size)
        return rearrange(out, 'b n (h f) -> b h n f',
                         h=self._horizon, f=self._output_size)
