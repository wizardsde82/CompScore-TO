from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from compscore_to.models.surrogate import SurrogateOutput
from compscore_to.models.vae import LatentDistribution


@dataclass(frozen=True)
class VAELossOutput:
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


@dataclass(frozen=True)
class SurrogateLossOutput:
    total: torch.Tensor
    compliance: torch.Tensor
    stress: torch.Tensor
    manufacturing: torch.Tensor


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    distribution: LatentDistribution,
    kl_weight: float,
) -> VAELossOutput:
    reconstruction_loss = F.mse_loss(reconstruction, target)
    kl = distribution.kl()
    return VAELossOutput(reconstruction_loss + kl_weight * kl, reconstruction_loss, kl)


def edm_loss(
    prediction: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor, sigma_data: float
) -> torch.Tensor:
    weight = (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()
    error = (prediction - target).square().flatten(1).mean(dim=1)
    return (weight * error).mean()


def anatomy_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def surrogate_loss(
    prediction: SurrogateOutput,
    compliance_target: torch.Tensor,
    stress_target: torch.Tensor,
    manufacturing_target: torch.Tensor,
    stress_weight: float = 0.5,
    manufacturing_weight: float = 0.3,
) -> SurrogateLossOutput:
    compliance_value = F.mse_loss(prediction.compliance, compliance_target)
    stress_value = F.mse_loss(prediction.stress, stress_target)
    manufacturing_value = F.binary_cross_entropy_with_logits(
        prediction.manufacturing, manufacturing_target
    )
    total = (
        compliance_value + stress_weight * stress_value + manufacturing_weight * manufacturing_value
    )
    return SurrogateLossOutput(total, compliance_value, stress_value, manufacturing_value)
