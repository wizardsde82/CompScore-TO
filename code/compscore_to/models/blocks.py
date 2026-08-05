from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def group_count(channels: int) -> int:
    for count in (32, 16, 8, 4, 2, 1):
        if channels % count == 0:
            return count
    return 1


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=value.device) / max(half - 1, 1)
        )
        phase = value.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return F.pad(embedding, (0, self.dimension - embedding.shape[-1]))


class ResidualBlock3d(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, embedding_channels: int = 0
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(input_channels), input_channels)
        self.conv1 = nn.Conv3d(input_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(output_channels), output_channels)
        self.conv2 = nn.Conv3d(output_channels, output_channels, 3, padding=1)
        self.embedding = (
            nn.Linear(embedding_channels, output_channels * 2) if embedding_channels else None
        )
        self.skip = (
            nn.Conv3d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, tensor: torch.Tensor, embedding: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(tensor)))
        normalized = self.norm2(hidden)
        if self.embedding is not None and embedding is not None:
            scale, shift = self.embedding(F.silu(embedding)).chunk(2, dim=1)
            normalized = normalized * (1.0 + scale[:, :, None, None, None])
            normalized = normalized + shift[:, :, None, None, None]
        hidden = self.conv2(F.silu(normalized))
        return self.skip(tensor) + hidden


class SelfAttention3d(nn.Module):
    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.projection = nn.Linear(channels, channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = tensor.shape
        sequence = self.norm(tensor).flatten(2).transpose(1, 2)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        attended = (
            self.projection(attended).transpose(1, 2).reshape(batch, channels, depth, height, width)
        )
        return tensor + attended


class CrossAttention3d(nn.Module):
    def __init__(self, channels: int, context_channels: int, heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(context_channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(context_channels, channels)
        self.value = nn.Linear(context_channels, channels)
        self.output = nn.Linear(channels, channels)
        self.heads = heads
        self.scale = (channels // heads) ** -0.5

    def forward(self, tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = tensor.shape
        sequence = tensor.flatten(2).transpose(1, 2)
        query = self.query(self.norm(sequence))
        normalized_context = self.context_norm(context)
        key = self.key(normalized_context)
        value = self.value(normalized_context)
        query = query.reshape(batch, -1, self.heads, channels // self.heads).transpose(1, 2)
        key = key.reshape(batch, -1, self.heads, channels // self.heads).transpose(1, 2)
        value = value.reshape(batch, -1, self.heads, channels // self.heads).transpose(1, 2)
        weights = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, -1, channels)
        result = sequence + self.output(attended)
        return result.transpose(1, 2).reshape(batch, channels, depth, height, width)


class Downsample3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layer(tensor)


class Upsample3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        target = tuple(size * 2 for size in tensor.shape[-3:])
        return self.layer(F.interpolate(tensor, size=target, mode="trilinear", align_corners=False))
