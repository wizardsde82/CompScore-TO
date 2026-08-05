from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from compscore_to.models.blocks import (
    CrossAttention3d,
    ResidualBlock3d,
    SelfAttention3d,
    SinusoidalEmbedding,
)


class ConditionalStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        embedding_channels: int,
        context_channels: int,
        attention: bool,
    ) -> None:
        super().__init__()
        self.first = ResidualBlock3d(input_channels, output_channels, embedding_channels)
        self.second = ResidualBlock3d(output_channels, output_channels, embedding_channels)
        self.cross = CrossAttention3d(output_channels, context_channels)
        self.self_attention = SelfAttention3d(output_channels) if attention else nn.Identity()

    def forward(
        self, tensor: torch.Tensor, embedding: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.first(tensor, embedding)
        hidden = self.second(hidden, embedding)
        hidden = self.cross(hidden, context)
        return self.self_attention(hidden)


class PhysicsConditionedUNet(nn.Module):
    def __init__(
        self, latent_channels: int = 4, base_channels: int = 96, context_channels: int = 256
    ) -> None:
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 4)
        embedding_channels = base_channels * 4
        self.noise_embedding = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.input = nn.Conv3d(latent_channels, widths[0], 3, padding=1)
        self.down_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        previous = widths[0]
        for index, width in enumerate(widths):
            self.down_stages.append(
                ConditionalStage(previous, width, embedding_channels, context_channels, index == 3)
            )
            if index < len(widths) - 1:
                self.downsamplers.append(nn.Conv3d(width, width, 3, stride=2, padding=1))
            previous = width
        self.middle = ConditionalStage(
            widths[-1], widths[-1], embedding_channels, context_channels, True
        )
        self.up_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        reversed_widths = tuple(reversed(widths))
        previous = reversed_widths[0]
        for index, width in enumerate(reversed_widths):
            self.up_stages.append(
                ConditionalStage(
                    previous + width, width, embedding_channels, context_channels, index == 0
                )
            )
            if index < len(reversed_widths) - 1:
                self.upsamplers.append(nn.Conv3d(width, width, 3, padding=1))
            previous = width
        self.output_norm = nn.GroupNorm(32, widths[0])
        self.output = nn.Conv3d(widths[0], latent_channels, 3, padding=1)

    def forward(
        self, tensor: torch.Tensor, noise: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.noise_embedding(noise)
        hidden = self.input(tensor)
        skips = []
        for index, stage in enumerate(self.down_stages):
            hidden = stage(hidden, embedding, context)
            skips.append(hidden)
            if index < len(self.downsamplers):
                hidden = self.downsamplers[index](hidden)
        hidden = self.middle(hidden, embedding, context)
        for index, stage in enumerate(self.up_stages):
            skip = skips.pop()
            if hidden.shape[-3:] != skip.shape[-3:]:
                hidden = F.interpolate(
                    hidden, size=skip.shape[-3:], mode="trilinear", align_corners=False
                )
            hidden = stage(torch.cat((hidden, skip), dim=1), embedding, context)
            if index < len(self.upsamplers):
                target = tuple(size * 2 for size in hidden.shape[-3:])
                hidden = self.upsamplers[index](
                    F.interpolate(hidden, target, mode="trilinear", align_corners=False)
                )
        return self.output(F.silu(self.output_norm(hidden)))


class EDMPreconditioner(nn.Module):
    def __init__(self, network: PhysicsConditionedUNet, sigma_data: float = 0.5) -> None:
        super().__init__()
        self.network = network
        self.sigma_data = sigma_data

    def coefficients(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_data = torch.as_tensor(self.sigma_data, device=sigma.device)
        denominator = sigma.square() + sigma_data.square()
        skip = sigma_data.square() / denominator
        output = sigma * sigma_data / denominator.sqrt()
        input_coefficient = denominator.rsqrt()
        noise = sigma.log() / 4.0
        return skip, output, input_coefficient, noise

    def forward(
        self, noisy: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        skip, output, input_coefficient, noise = self.coefficients(sigma)
        network_output = self.network(
            input_coefficient.reshape(-1, 1, 1, 1, 1) * noisy,
            noise,
            context,
        )
        return (
            skip.reshape(-1, 1, 1, 1, 1) * noisy + output.reshape(-1, 1, 1, 1, 1) * network_output
        )
