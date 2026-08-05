from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GraphBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    batch: torch.Tensor
    coordinates: torch.Tensor
    spatial_shape: tuple[int, int, int]


@dataclass(frozen=True)
class SurrogateOutput:
    compliance: torch.Tensor
    stress: torch.Tensor
    manufacturing: torch.Tensor


def scatter_sum(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    output = values.new_zeros((count, values.shape[-1]))
    output.index_add_(0, index, values)
    return output


def scatter_mean(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    total = scatter_sum(values, index, count)
    normalizer = values.new_zeros((count, 1))
    normalizer.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return total / normalizer.clamp_min(1.0)


class EquivariantMessageLayer(nn.Module):
    def __init__(self, channels: int, edge_channels: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(channels * 2 + edge_channels + 1, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        self.update = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        self.coordinate_weight = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        displacement = coordinates[source] - coordinates[target]
        squared_distance = displacement.square().sum(dim=-1, keepdim=True)
        inputs = torch.cat(
            (features[source], features[target], edge_features, squared_distance), dim=-1
        )
        messages = self.message(inputs)
        aggregated = scatter_mean(messages, target, features.shape[0])
        updated = self.norm(features + self.update(torch.cat((features, aggregated), dim=-1)))
        coordinate_messages = displacement * self.coordinate_weight(messages)
        coordinate_update = scatter_mean(coordinate_messages, target, coordinates.shape[0])
        return updated, coordinates + coordinate_update


class MultiPhysicsSurrogate(nn.Module):
    def __init__(
        self,
        node_channels: int = 8,
        edge_channels: int = 4,
        hidden_channels: int = 128,
        layers: int = 5,
    ) -> None:
        super().__init__()
        self.node_input = nn.Sequential(
            nn.Linear(node_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.layers = nn.ModuleList(
            [EquivariantMessageLayer(hidden_channels, edge_channels) for _ in range(layers)]
        )
        self.compliance_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, 1)
        )
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
            nn.Softplus(),
        )
        self.manufacturing_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, 1)
        )

    def forward(self, graph: GraphBatch) -> SurrogateOutput:
        features = self.node_input(graph.node_features)
        coordinates = graph.coordinates
        for layer in self.layers:
            features, coordinates = layer(
                features, coordinates, graph.edge_index, graph.edge_features
            )
        graph_count = int(graph.batch.max().item()) + 1
        pooled = scatter_mean(features, graph.batch, graph_count)
        return SurrogateOutput(
            compliance=F.softplus(self.compliance_head(pooled)).squeeze(-1),
            stress=self.stress_head(features).squeeze(-1),
            manufacturing=self.manufacturing_head(pooled).squeeze(-1),
        )
