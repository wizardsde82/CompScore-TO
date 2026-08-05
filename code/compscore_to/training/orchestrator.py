from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from compscore_to.configuration import ProjectConfig
from compscore_to.losses import anatomy_reconstruction_loss, edm_loss, surrogate_loss, vae_loss
from compscore_to.models.anatomy import AnatomyEncoder
from compscore_to.models.diffusion import EDMPreconditioner
from compscore_to.models.surrogate import MultiPhysicsSurrogate
from compscore_to.models.vae import TopologyVAE
from compscore_to.physics.graph import density_to_graph
from compscore_to.training.schedules import LinearWarmupCosine, sample_noise_levels
from compscore_to.training.state import ExponentialMovingAverage, TrainState, save_checkpoint


class TrainingOrchestrator:
    def __init__(self, config: ProjectConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.logger = logging.getLogger(type(self).__name__)
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

    def train_vae(
        self, model: TopologyVAE, loader: DataLoader[dict[str, torch.Tensor]]
    ) -> TrainState:
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.vae.learning_rate,
            weight_decay=self.config.vae.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.config.vae.restart_epochs[0]
        )
        state = TrainState()
        for epoch in range(self.config.vae.epochs):
            model.train()
            for batch in loader:
                topology = batch["topology"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, distribution = model(topology)
                output = vae_loss(reconstruction, topology, distribution, self.config.vae.kl_weight)
                output.total.backward()
                clip_grad_norm_(model.parameters(), self.config.vae.gradient_clip)
                optimizer.step()
                state.step += 1
                state.best_loss = min(state.best_loss, float(output.total.detach()))
            state.epoch = epoch + 1
            scheduler.step()
            save_checkpoint(str(self.output / "vae.pt"), model, optimizer, state, self.config.seed)
            self.logger.info("vae epoch=%d loss=%.6f", state.epoch, state.best_loss)
        return state

    def train_anatomy(
        self, model: AnatomyEncoder, loader: DataLoader[dict[str, torch.Tensor]]
    ) -> TrainState:
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.diffusion.learning_rate,
            weight_decay=self.config.diffusion.weight_decay,
        )
        state = TrainState()
        for epoch in range(40):
            model.train()
            for batch in loader:
                anatomy = batch["anatomy"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model.reconstruct(anatomy)
                loss = anatomy_reconstruction_loss(reconstruction, anatomy)
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                state.step += 1
                state.best_loss = min(state.best_loss, float(loss.detach()))
            state.epoch = epoch + 1
            save_checkpoint(
                str(self.output / "anatomy.pt"), model, optimizer, state, self.config.seed
            )
            self.logger.info("anatomy epoch=%d loss=%.6f", state.epoch, state.best_loss)
        return state

    def train_surrogate(
        self, model: MultiPhysicsSurrogate, loader: DataLoader[dict[str, torch.Tensor]]
    ) -> TrainState:
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.surrogate.learning_rate,
            weight_decay=self.config.surrogate.weight_decay,
        )
        state = TrainState()
        for epoch in range(self.config.surrogate.epochs):
            model.train()
            for batch in loader:
                density = batch["topology"].to(self.device)
                load = batch["load"].to(self.device)
                boundary = batch["boundary"].to(self.device)
                graph = density_to_graph(density, load, boundary)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(graph)
                loss = surrogate_loss(
                    prediction,
                    batch["compliance"].to(self.device),
                    batch["stress"].to(self.device).flatten(),
                    batch["manufacturing"].to(self.device),
                    self.config.surrogate.stress_weight,
                    self.config.surrogate.manufacturing_weight,
                )
                loss.total.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                state.step += 1
                state.best_loss = min(state.best_loss, float(loss.total.detach()))
            state.epoch = epoch + 1
            save_checkpoint(
                str(self.output / "surrogate.pt"), model, optimizer, state, self.config.seed
            )
            self.logger.info("surrogate epoch=%d loss=%.6f", state.epoch, state.best_loss)
        return state

    def train_diffusion(
        self,
        model: EDMPreconditioner,
        vae: TopologyVAE,
        anatomy_encoder: AnatomyEncoder,
        loader: DataLoader[dict[str, torch.Tensor]],
    ) -> TrainState:
        model.to(self.device)
        vae.to(self.device).eval()
        anatomy_encoder.to(self.device).eval()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.diffusion.learning_rate,
            weight_decay=self.config.diffusion.weight_decay,
        )
        scheduler = LinearWarmupCosine(
            optimizer, self.config.diffusion.warmup_steps, self.config.diffusion.iterations
        )
        ema = ExponentialMovingAverage(model, self.config.diffusion.ema_decay)
        state = TrainState()
        iterator = iter(loader)
        while state.step < self.config.diffusion.iterations:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
                state.epoch += 1
            topology = batch["topology"].to(self.device)
            anatomy = batch["anatomy"].to(self.device)
            with torch.no_grad():
                target = vae.encode(topology).sample()
                context = anatomy_encoder(anatomy)
            sigma = sample_noise_levels(
                target.shape[0],
                self.config.diffusion.p_mean,
                self.config.diffusion.p_std,
                self.device,
            )
            noisy = target + torch.randn_like(target) * sigma[:, None, None, None, None]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy, sigma, context)
            loss = edm_loss(prediction, target, sigma, self.config.diffusion.sigma_data)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            state.step += 1
            state.best_loss = min(state.best_loss, float(loss.detach()))
            if state.step % 1000 == 0:
                save_checkpoint(
                    str(self.output / "diffusion.pt"),
                    model,
                    optimizer,
                    state,
                    self.config.seed,
                    ema,
                )
                self.logger.info("diffusion step=%d loss=%.6f", state.step, state.best_loss)
        return state
