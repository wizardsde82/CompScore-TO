from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from compscore_to.models.blocks import ResidualBlock3d, group_count


@dataclass(frozen=True)
class LatentDistribution:
    mean: torch.Tensor
    log_variance: torch.Tensor

    def sample(self) -> torch.Tensor:
        deviation = torch.exp(0.5 * self.log_variance)
        return self.mean + deviation * torch.randn_like(deviation)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        value = self.mean.square() + self.log_variance.exp() - 1.0 - self.log_variance
        return 0.5 * value.flatten(1).sum(dim=1).mean()


class Encoder3d(nn.Module):
    def __init__(self, latent_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 4)
        self.input = nn.Conv3d(1, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        previous = widths[0]
        for index, width in enumerate(widths):
            stage = nn.ModuleList([ResidualBlock3d(previous, width), ResidualBlock3d(width, width)])
            downsample = (
                nn.Conv3d(width, width, 4, stride=2, padding=1) if index < 3 else nn.Identity()
            )
            self.blocks.append(nn.ModuleDict({"stage": stage, "downsample": downsample}))
            previous = width
        self.norm = nn.GroupNorm(group_count(previous), previous)
        self.output = nn.Conv3d(previous, latent_channels * 2, 3, padding=1)

    def forward(self, tensor: torch.Tensor) -> LatentDistribution:
        hidden = self.input(tensor)
        for block in self.blocks:
            for layer in block["stage"]:
                hidden = layer(hidden)
            hidden = block["downsample"](hidden)
        parameters = self.output(F.silu(self.norm(hidden)))
        mean, log_variance = parameters.chunk(2, dim=1)
        return LatentDistribution(mean, log_variance.clamp(-30.0, 20.0))


class Decoder3d(nn.Module):
    def __init__(self, latent_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        widths = (base_channels * 4, base_channels * 4, base_channels * 2, base_channels)
        self.input = nn.Conv3d(latent_channels, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        previous = widths[0]
        for index, width in enumerate(widths):
            stage = nn.ModuleList([ResidualBlock3d(previous, width), ResidualBlock3d(width, width)])
            upsample = (
                nn.ConvTranspose3d(width, width, 4, stride=2, padding=1)
                if index < 3
                else nn.Identity()
            )
            self.blocks.append(nn.ModuleDict({"stage": stage, "upsample": upsample}))
            previous = width
        self.norm = nn.GroupNorm(group_count(previous), previous)
        self.output = nn.Conv3d(previous, 1, 3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        hidden = self.input(tensor)
        for block in self.blocks:
            for layer in block["stage"]:
                hidden = layer(hidden)
            hidden = block["upsample"](hidden)
        return torch.sigmoid(self.output(F.silu(self.norm(hidden))))


class TopologyVAE(nn.Module):
    def __init__(self, latent_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        self.encoder = Encoder3d(latent_channels, base_channels)
        self.decoder = Decoder3d(latent_channels, base_channels)

    def encode(self, topology: torch.Tensor) -> LatentDistribution:
        return self.encoder(topology)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, topology: torch.Tensor) -> tuple[torch.Tensor, LatentDistribution]:
        distribution = self.encode(topology)
        reconstruction = self.decode(distribution.sample())
        return reconstruction, distribution
