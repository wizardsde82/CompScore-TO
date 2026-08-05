from __future__ import annotations

import math

import torch


class LinearWarmupCosine(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, self.factor)

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def sample_noise_levels(
    batch_size: int, p_mean: float, p_std: float, device: torch.device
) -> torch.Tensor:
    return (torch.randn(batch_size, device=device) * p_std + p_mean).exp()


def guidance_weights(
    step: int, total_steps: int, compliance: float, stress: float, manufacturing: float
) -> tuple[float, float, float]:
    progress = step / total_steps
    return compliance * progress, stress * (1.0 - progress), manufacturing * (1.0 - progress)
