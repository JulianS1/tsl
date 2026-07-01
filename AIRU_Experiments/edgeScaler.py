import torch
import torch.nn as nn


class EdgeScaler(nn.Module):
    def __init__(self, in_features, hidden_features,scaling_mode):
        super().__init__()

        self.scaling_mode = scaling_mode
        z = 3 * in_features

        if scaling_mode == 'uniform':
            self.b = nn.Parameter(torch.zeros(1))

        if scaling_mode == 'linear':
            self.w = nn.Parameter(torch.zeros(z, 1))
            self.b = nn.Parameter(torch.zeros(1))

        if scaling_mode == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(z, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, 1)
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        if scaling_mode == 'tcn':
            self.tcn = nn.Sequential(
                nn.Conv1d(z, hidden_features, kernel_size=3, padding=1),
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
        # Flatten time and features for each node: (B, N, T*F)
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, T * F)  # (B, N, T*F)

        src, dst = edge_index  # src=j (source), dst=i (target)
        x_i = x_flat[:, dst, :]  # (B, E, T*F)
        x_j = x_flat[:, src, :]  # (B, E, T*F)
        z   = torch.cat([x_i, x_j, x_i - x_j], dim=-1)  # (B, E, 3*T*F)

        if self.scaling_mode == 'uniform':
            logit = self.b.expand(z.shape[0], z.shape[1], 1)

        elif self.scaling_mode == 'linear':
            logit = z @ self.w + self.b

        elif self.scaling_mode == 'mlp':
            logit = self.mlp(z)  # (B, E, 1)
        
        elif self.scaling_mode == 'tcn':
            logit = self.tcn(z.permute(0, 2, 1)).permute(0, 2, 1)  # (B, E, 1)

        s = 2 * torch.sigmoid(logit).squeeze(-1)  # (B, E), values in (0, 2)
        s = s.mean(dim=0)                          # (E,)  average over batch

        return edge_weight * s


