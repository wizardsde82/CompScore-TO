from __future__ import annotations

import torch
from torch.nn import functional as F


def simp_modulus(
    density: torch.Tensor, modulus: float = 110.0, penalty: float = 3.0
) -> torch.Tensor:
    return density.clamp_min(1e-6).pow(penalty) * modulus


def compliance(displacement: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    return (displacement * force).flatten(1).sum(dim=1)


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    xx, yy, zz, xy, yz, xz = stress.unbind(dim=1)
    normal = 0.5 * ((xx - yy).square() + (yy - zz).square() + (zz - xx).square())
    shear = 3.0 * (xy.square() + yz.square() + xz.square())
    return (normal + shear).clamp_min(0.0).sqrt()


def stress_shielding_ratio(
    implant_stress: torch.Tensor, intact_stress: torch.Tensor, shell: torch.Tensor
) -> torch.Tensor:
    axes = tuple(range(1, implant_stress.ndim))
    numerator = (implant_stress * shell).sum(dim=axes)
    denominator = (intact_stress * shell).sum(dim=axes).clamp_min(1e-8)
    return numerator / denominator


def overhang_score(density: torch.Tensor, maximum_angle: float = 45.0) -> torch.Tensor:
    supported = F.max_pool3d(density[:, :, :-1], kernel_size=3, stride=1, padding=1)
    unsupported = (density[:, :, 1:] - supported).clamp_min(0.0)
    tolerance = torch.tan(torch.deg2rad(torch.tensor(maximum_angle, device=density.device)))
    return 1.0 - (unsupported / tolerance.clamp_min(1e-6)).clamp(0.0, 1.0).mean(dim=(1, 2, 3, 4))


def feature_size_score(density: torch.Tensor, minimum_voxels: int = 1) -> torch.Tensor:
    kernel = minimum_voxels * 2 + 1
    eroded = -F.max_pool3d(-density, kernel_size=kernel, stride=1, padding=minimum_voxels)
    opened = F.max_pool3d(eroded, kernel_size=kernel, stride=1, padding=minimum_voxels)
    difference = (density - opened).abs().mean(dim=(1, 2, 3, 4))
    return 1.0 - difference.clamp(0.0, 1.0)


def connectivity_score(density: torch.Tensor) -> torch.Tensor:
    pooled = F.avg_pool3d(density, kernel_size=3, stride=1, padding=1)
    isolated = density * (pooled < density / 27.0 + 1e-6)
    return 1.0 - isolated.mean(dim=(1, 2, 3, 4))


def additive_manufacturing_score(
    density: torch.Tensor, maximum_angle: float = 45.0, minimum_voxels: int = 1
) -> torch.Tensor:
    overhang = overhang_score(density, maximum_angle)
    feature = feature_size_score(density, minimum_voxels)
    connected = connectivity_score(density)
    return torch.stack((overhang, feature, connected), dim=1).mean(dim=1)


def volume_fraction_error(density: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    actual = density.mean(dim=(1, 2, 3, 4))
    return (actual - target).abs()


def anatomy_overlap(density: torch.Tensor, anatomy_roi: torch.Tensor) -> torch.Tensor:
    intersection = (density * anatomy_roi).sum(dim=(1, 2, 3, 4))
    denominator = density.sum(dim=(1, 2, 3, 4)) + anatomy_roi.sum(dim=(1, 2, 3, 4))
    return 2.0 * intersection / denominator.clamp_min(1e-8)
