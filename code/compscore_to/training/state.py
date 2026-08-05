from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from compscore_to.runtime import atomic_save


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")

    def mapping(self) -> dict[str, int | float]:
        return {"epoch": self.epoch, "step": self.step, "best_loss": self.best_loss}


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, values: dict[str, Any]) -> None:
        self.decay = float(values["decay"])
        self.shadow = values["shadow"]


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    seed: int,
    ema: ExponentialMovingAverage | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state.mapping(),
        "seed": seed,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    atomic_save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None = None,
) -> tuple[TrainState, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    values = payload["state"]
    state = TrainState(
        epoch=int(values["epoch"]), step=int(values["step"]), best_loss=float(values["best_loss"])
    )
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    return state, int(payload["seed"])
