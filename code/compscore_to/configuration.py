from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class DataConfig:
    root: str
    grid_size: int
    spacing_mm: float
    train_fraction: float
    validation_fraction: float
    load_cases: int
    volume_fractions: tuple[float, ...]


@dataclass(frozen=True)
class ModelConfig:
    latent_channels: int
    anatomy_channels: int
    anatomy_tokens: int
    base_channels: int
    channel_multipliers: tuple[int, ...]
    attention_levels: tuple[int, ...]
    message_passing_layers: int


@dataclass(frozen=True)
class VAEConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    kl_weight: float
    gradient_clip: float
    restart_epochs: tuple[int, ...]


@dataclass(frozen=True)
class DiffusionConfig:
    iterations: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    ema_decay: float
    p_mean: float
    p_std: float
    sigma_min: float
    sigma_max: float
    sigma_data: float


@dataclass(frozen=True)
class SurrogateConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    stress_weight: float
    manufacturing_weight: float


@dataclass(frozen=True)
class GuidanceConfig:
    compliance: float
    stress: float
    manufacturing: float
    steps: int


@dataclass(frozen=True)
class PhysicsConfig:
    simp_penalty: float
    titanium_modulus_gpa: float
    volume_fraction: float
    filter_radius_voxels: float
    minimum_feature_mm: float
    maximum_overhang_degrees: float
    peri_implant_shell_mm: float


@dataclass(frozen=True)
class StatisticsConfig:
    seeds: int
    bootstrap_resamples: int
    false_discovery_rate: float


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    output_dir: str
    data: DataConfig
    model: ModelConfig
    vae: VAEConfig
    diffusion: DiffusionConfig
    surrogate: SurrogateConfig
    guidance: GuidanceConfig
    physics: PhysicsConfig
    statistics: StatisticsConfig


def _tuple(values: list[Any]) -> tuple[Any, ...]:
    return tuple(values)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ProjectConfig:
    base = OmegaConf.load(path)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    values = OmegaConf.to_container(base, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("configuration root must be a mapping")
    data = values["data"]
    model = values["model"]
    vae = values["vae"]
    diffusion = values["diffusion"]
    surrogate = values["surrogate"]
    guidance = values["guidance"]
    physics = values["physics"]
    statistics = values["statistics"]
    if not all(
        isinstance(item, dict)
        for item in (data, model, vae, diffusion, surrogate, guidance, physics, statistics)
    ):
        raise TypeError("configuration sections must be mappings")
    return ProjectConfig(
        seed=int(values["seed"]),
        output_dir=str(values["output_dir"]),
        data=DataConfig(
            root=str(data["root"]),
            grid_size=int(data["grid_size"]),
            spacing_mm=float(data["spacing_mm"]),
            train_fraction=float(data["train_fraction"]),
            validation_fraction=float(data["validation_fraction"]),
            load_cases=int(data["load_cases"]),
            volume_fractions=_tuple(data["volume_fractions"]),
        ),
        model=ModelConfig(
            latent_channels=int(model["latent_channels"]),
            anatomy_channels=int(model["anatomy_channels"]),
            anatomy_tokens=int(model["anatomy_tokens"]),
            base_channels=int(model["base_channels"]),
            channel_multipliers=_tuple(model["channel_multipliers"]),
            attention_levels=_tuple(model["attention_levels"]),
            message_passing_layers=int(model["message_passing_layers"]),
        ),
        vae=VAEConfig(**vae),
        diffusion=DiffusionConfig(**diffusion),
        surrogate=SurrogateConfig(**surrogate),
        guidance=GuidanceConfig(**guidance),
        physics=PhysicsConfig(**physics),
        statistics=StatisticsConfig(**statistics),
    )
