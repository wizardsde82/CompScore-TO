from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compscore_to.configuration import load_config
from compscore_to.data.dataset import AnatomyTopologyDataset
from compscore_to.models.anatomy import AnatomyEncoder
from compscore_to.models.diffusion import EDMPreconditioner, PhysicsConditionedUNet
from compscore_to.models.surrogate import MultiPhysicsSurrogate
from compscore_to.models.vae import TopologyVAE
from compscore_to.runtime import configure_logging, set_seed
from compscore_to.training.orchestrator import TrainingOrchestrator


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="compscore-train")
    result.add_argument("--config", type=Path, default=Path("settings/main.yaml"))
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument(
        "--stage", choices=("vae", "anatomy", "surrogate", "diffusion"), required=True
    )
    result.add_argument("overrides", nargs="*")
    return result


def main() -> None:
    arguments = parser().parse_args()
    configure_logging()
    config = load_config(arguments.config, arguments.overrides)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AnatomyTopologyDataset(arguments.manifest, "train")
    batch_size = {
        "vae": config.vae.batch_size,
        "anatomy": config.vae.batch_size,
        "surrogate": config.surrogate.batch_size,
        "diffusion": config.diffusion.batch_size,
    }[arguments.stage]
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    orchestrator = TrainingOrchestrator(config, device)
    vae = TopologyVAE(config.model.latent_channels)
    anatomy = AnatomyEncoder(config.model.anatomy_channels, config.model.anatomy_tokens)
    if arguments.stage == "vae":
        orchestrator.train_vae(vae, loader)
    elif arguments.stage == "anatomy":
        orchestrator.train_anatomy(anatomy, loader)
    elif arguments.stage == "surrogate":
        surrogate = MultiPhysicsSurrogate(layers=config.model.message_passing_layers)
        orchestrator.train_surrogate(surrogate, loader)
    else:
        network = PhysicsConditionedUNet(
            config.model.latent_channels, config.model.base_channels, config.model.anatomy_channels
        )
        diffusion = EDMPreconditioner(network, config.diffusion.sigma_data)
        orchestrator.train_diffusion(diffusion, vae, anatomy, loader)


if __name__ == "__main__":
    main()
