import torch
import torch.nn as nn


class EdgeScaler(nn.Module):
    def __init__(self, window_size, num_features, hidden_features, scaling_mode):
        super().__init__()

        self.scaling_mode = scaling_mode
        flat_dim = 3 * window_size * num_features  # z_ij dim for uniform/linear/mlp
        seq_channels = 3 * num_features             # z_ij channel dim for tcn (time kept separate)

        if scaling_mode == 'uniform':
            self.b = nn.Parameter(torch.zeros(1))

        if scaling_mode == 'linear':
            self.w = nn.Parameter(torch.zeros(flat_dim, 1))
            self.b = nn.Parameter(torch.zeros(1))

        if scaling_mode == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(flat_dim, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, 1)
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        if scaling_mode == 'tcn':
            # Convolves over the temporal axis (length T) of each edge's
            # [x_i, x_j, x_i-x_j] sequence (channels = 3*F), not over the
            # (arbitrarily ordered) edge axis.
            self.tcn = nn.Sequential(
                nn.Conv1d(seq_channels, hidden_features, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_features, 1, kernel_size=3, padding=1)
            )
            nn.init.zeros_(self.tcn[-1].weight)
            nn.init.zeros_(self.tcn[-1].bias)


    def scale(self, x, edge_index, edge_weight):

        """
        x:           (B, T, N, F)  — input window
        edge_index:  (2, E)        — source/target node indices
        edge_weight: (E,)          — static adjacency weights a_ij

        returns: (E,) modulated edge weights, averaged over batch
        """

        if self.scaling_mode == 'fixed':
            return edge_weight

        B, T, N, F = x.shape
        src, dst = edge_index  # src=j (source), dst=i (target)

        if self.scaling_mode == 'tcn':
            x_t = x.permute(0, 2, 3, 1)            # (B, N, F, T)
            x_i = x_t[:, dst, :, :]                # (B, E, F, T)
            x_j = x_t[:, src, :, :]
            z = torch.cat([x_i, x_j, x_i - x_j], dim=2)  # (B, E, 3*F, T)
            E = z.shape[1]
            z = z.reshape(B * E, 3 * F, T)         # (B*E, 3*F, T)
            logit = self.tcn(z)                    # (B*E, 1, T)
            logit = logit.mean(dim=-1).reshape(B, E)  # (B, E)

        else:
            # Flatten time and features for each node: (B, N, T*F)
            x_flat = x.permute(0, 2, 1, 3).reshape(B, N, T * F)
            x_i = x_flat[:, dst, :]
            x_j = x_flat[:, src, :]
            z   = torch.cat([x_i, x_j, x_i - x_j], dim=-1)  # (B, E, 3*T*F)

            if self.scaling_mode == 'uniform':
                logit = self.b.expand(z.shape[0], z.shape[1])

            elif self.scaling_mode == 'linear':
                logit = (z @ self.w).squeeze(-1) + self.b

            elif self.scaling_mode == 'mlp':
                logit = self.mlp(z).squeeze(-1)  # (B, E)

        s = 2 * torch.sigmoid(logit)
        s = s.mean(dim=0)
        return edge_weight * s


