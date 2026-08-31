import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATv2Conv

class VarunaGATBackbone(nn.Module):
    def __init__(self, in_channels=4, hidden=64, heads=4, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(in_channels, hidden, heads=heads, edge_dim=2))
        for _ in range(n_layers - 1):
            self.layers.append(GATv2Conv(hidden * heads, hidden, heads=heads, edge_dim=2))
        self.act = nn.ELU()

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = self.act(layer(x, edge_index, edge_attr))
        return x


class RegionConditionedDecoder(nn.Module):
    def __init__(self, latent_dim, n_regions=3, n_depths=15, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + n_regions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.out = nn.Linear(hidden, n_depths * 4)
        self.n_depths = n_depths

    def forward(self, z, region_onehot):
        z_cond = torch.cat([z, region_onehot], dim=1)
        h = self.mlp(z_cond)
        raw = self.out(h).view(-1, self.n_depths, 4)
        gamma = raw[..., 0]
        nu    = torch.nn.functional.softplus(raw[..., 1])
        alpha = torch.nn.functional.softplus(raw[..., 2]) + 1
        beta  = torch.nn.functional.softplus(raw[..., 3])
        return gamma, nu, alpha, beta


def compute_edge_attr(edge_index, static_directions, u_flat, v_flat):
    src = edge_index[0].numpy()
    current = np.stack([u_flat[src], v_flat[src]], axis=1)
    alignment = np.sum(static_directions.numpy() * current, axis=1)
    edge_attr = np.stack([alignment, np.ones_like(alignment)], axis=1)
    return torch.tensor(edge_attr, dtype=torch.float32)


def compute_uncertainty(nu, alpha, beta):
    aleatoric = beta / (nu * (alpha - 1))
    epistemic = beta / (alpha - 1)
    return aleatoric + epistemic