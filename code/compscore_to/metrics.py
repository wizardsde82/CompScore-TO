from __future__ import annotations

import torch


def compliance_error(generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (generated - reference).abs() / reference.abs().clamp_min(1e-8) * 100.0


def dice_coefficient(
    prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    predicted = prediction >= threshold
    expected = target >= threshold
    intersection = (predicted & expected).flatten(1).sum(dim=1)
    denominator = predicted.flatten(1).sum(dim=1) + expected.flatten(1).sum(dim=1)
    return 2.0 * intersection / denominator.clamp_min(1)


def fatigue_life_proxy(
    alternating_stress: torch.Tensor,
    mean_stress: torch.Tensor,
    endurance_limit: float,
    ultimate_strength: float,
) -> torch.Tensor:
    goodman = alternating_stress / endurance_limit + mean_stress / ultimate_strength
    return goodman.clamp_min(1e-8).reciprocal()


def mean_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).abs().mean()


def root_mean_square_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean().sqrt()


def coefficient_of_determination(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum().clamp_min(1e-8)
    return 1.0 - residual / total


def structural_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(2, left.ndim))
    left_mean = left.mean(dim=axes, keepdim=True)
    right_mean = right.mean(dim=axes, keepdim=True)
    left_variance = (left - left_mean).square().mean(dim=axes)
    right_variance = (right - right_mean).square().mean(dim=axes)
    covariance = ((left - left_mean) * (right - right_mean)).mean(dim=axes)
    first = 2.0 * left_mean.squeeze() * right_mean.squeeze() + 0.01**2
    second = 2.0 * covariance + 0.03**2
    denominator = left_mean.squeeze().square() + right_mean.squeeze().square() + 0.01**2
    denominator = denominator * (left_variance + right_variance + 0.03**2)
    return first * second / denominator.clamp_min(1e-8)
