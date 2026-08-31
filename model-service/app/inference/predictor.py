import torch
import numpy as np
from ..models import compute_edge_attr, compute_uncertainty


def predict(bundle, so_grid, uo_grid, vo_grid, zos_grid):
    x_grid = np.stack([so_grid, uo_grid, vo_grid, zos_grid], axis=0)
    x_nodes = torch.tensor(x_grid.reshape(4, -1).T, dtype=torch.float32).to(bundle.device)

    edge_attr = compute_edge_attr(
        bundle.edge_index.cpu(), bundle.static_directions,
        uo_grid.flatten(), vo_grid.flatten()
    ).to(bundle.device)

    with torch.no_grad():
        z = bundle.backbone(x_nodes, bundle.edge_index, edge_attr)
        gamma, nu, alpha, beta = bundle.decoder(z, bundle.region_onehot_tensor)
        uncertainty = compute_uncertainty(nu, alpha, beta)

    H, W = len(bundle.lat_grid), len(bundle.lon_grid)
    temperature = gamma.cpu().numpy().reshape(H, W, -1)
    uncertainty = uncertainty.cpu().numpy().reshape(H, W, -1)

    return {
        "depths": bundle.target_depths.tolist(),
        "lat": bundle.lat_grid.tolist(),
        "lon": bundle.lon_grid.tolist(),
        "temperature": temperature.tolist(),
        "uncertainty": uncertainty.tolist(),
    }