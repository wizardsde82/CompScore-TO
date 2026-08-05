from __future__ import annotations

import torch

from compscore_to.models.surrogate import GraphBatch


def density_to_graph(
    density: torch.Tensor, load: torch.Tensor, boundary: torch.Tensor, threshold: float = 0.01
) -> GraphBatch:
    batches = []
    node_features = []
    coordinates = []
    edge_indices = []
    edge_features = []
    offset = 0
    directions = torch.tensor(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)), device=density.device
    )
    for batch_index in range(density.shape[0]):
        active = torch.nonzero(density[batch_index, 0] > threshold, as_tuple=False)
        shape = density.shape[-3:]
        linear = active[:, 0] * shape[1] * shape[2] + active[:, 1] * shape[2] + active[:, 2]
        mapping = {int(value.item()): index for index, value in enumerate(linear)}
        local_edges = []
        local_edge_features = []
        for node_index, coordinate in enumerate(active):
            for direction in directions:
                neighbor = coordinate + direction
                if torch.any(neighbor < 0) or torch.any(
                    neighbor >= torch.tensor(shape, device=density.device)
                ):
                    continue
                target_linear = int(
                    (
                        neighbor[0] * shape[1] * shape[2] + neighbor[1] * shape[2] + neighbor[2]
                    ).item()
                )
                target = mapping.get(target_linear)
                if target is not None:
                    local_edges.append((node_index + offset, target + offset))
                    local_edge_features.append(
                        torch.cat((direction.float(), torch.ones(1, device=density.device)))
                    )
        values = density[batch_index, 0, active[:, 0], active[:, 1], active[:, 2]].unsqueeze(-1)
        node_load = load[batch_index, :, active[:, 0], active[:, 1], active[:, 2]].transpose(0, 1)
        node_boundary = boundary[
            batch_index, :, active[:, 0], active[:, 1], active[:, 2]
        ].transpose(0, 1)
        node_features.append(torch.cat((values, node_load, node_boundary), dim=-1))
        coordinates.append(active.float())
        batches.append(
            torch.full((active.shape[0],), batch_index, device=density.device, dtype=torch.long)
        )
        edge_indices.extend(local_edges)
        edge_features.extend(local_edge_features)
        offset += active.shape[0]
    return GraphBatch(
        node_features=torch.cat(node_features),
        edge_index=torch.tensor(edge_indices, device=density.device, dtype=torch.long).transpose(
            0, 1
        ),
        edge_features=torch.stack(edge_features),
        batch=torch.cat(batches),
        coordinates=torch.cat(coordinates),
        spatial_shape=tuple(int(value) for value in density.shape[-3:]),
    )
