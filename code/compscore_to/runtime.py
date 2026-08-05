from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(payload: object, destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if distributed_ready() else 0


def world_size() -> int:
    return torch.distributed.get_world_size() if distributed_ready() else 1


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not distributed_ready():
        return value
    result = value.detach().clone()
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result / world_size()
