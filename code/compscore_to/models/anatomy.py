from __future__ import annotations

import torch
from torch import nn

from compscore_to.models.blocks import ResidualBlock3d


class AnatomyEncoder(nn.Module):
    def __init__(
        self, token_channels: int = 256, token_count: int = 512, base_channels: int = 32
    ) -> None:
        super().__init__()
        widths = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            token_channels,
            token_channels,
        )
        self.input = nn.Conv3d(1, widths[0], 3, padding=1)
        self.stages = nn.ModuleList()
        previous = widths[0]
        for index, width in enumerate(widths):
            self.stages.append(
                nn.Sequential(
                    ResidualBlock3d(previous, width),
                    ResidualBlock3d(width, width),
                    nn.Conv3d(width, width, 3, stride=2, padding=1) if index < 3 else nn.Identity(),
                )
            )
            previous = width
        self.token_pool = nn.AdaptiveAvgPool3d((8, 8, 8))
        self.token_projection = nn.Linear(previous, token_channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(previous, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 1, 3, padding=1),
        )
        self.token_count = token_count

    def features(self, anatomy: torch.Tensor) -> torch.Tensor:
        hidden = self.input(anatomy)
        for stage in self.stages:
            hidden = stage(hidden)
        return hidden

    def forward(self, anatomy: torch.Tensor) -> torch.Tensor:
        hidden = self.features(anatomy)
        tokens = self.token_pool(hidden).flatten(2).transpose(1, 2)
        tokens = self.token_projection(tokens)
        if tokens.shape[1] != self.token_count:
            raise RuntimeError("anatomy token count does not match configuration")
        return tokens

    def reconstruct(self, anatomy: torch.Tensor) -> torch.Tensor:
        hidden = self.features(anatomy)
        return torch.sigmoid(self.decoder(hidden))
