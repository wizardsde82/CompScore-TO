from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from compscore_to.metrics import (
    coefficient_of_determination,
    compliance_error,
    dice_coefficient,
    mean_absolute_error,
    root_mean_square_error,
)
from compscore_to.statistics import bootstrap_mean_interval


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    standard_deviation: float
    confidence_low: float
    confidence_high: float


@dataclass(frozen=True)
class EvaluationReport:
    compliance_error: MetricSummary
    stress_shielding_ratio: MetricSummary
    manufacturing: MetricSummary
    fatigue_life: MetricSummary
    dice: MetricSummary

    def mapping(self) -> dict[str, object]:
        return asdict(self)


def summarize(values: np.ndarray, resamples: int) -> MetricSummary:
    low, high = bootstrap_mean_interval(values, resamples=resamples)
    return MetricSummary(float(values.mean()), float(values.std(ddof=1)), low, high)


def evaluate_arrays(
    generated_compliance: np.ndarray,
    reference_compliance: np.ndarray,
    stress_ratio: np.ndarray,
    manufacturing: np.ndarray,
    fatigue: np.ndarray,
    generated_geometry: np.ndarray,
    anatomy_roi: np.ndarray,
    bootstrap_resamples: int = 10000,
) -> EvaluationReport:
    generated_tensor = torch.from_numpy(generated_geometry)
    roi_tensor = torch.from_numpy(anatomy_roi)
    compliance_values = compliance_error(
        torch.from_numpy(generated_compliance), torch.from_numpy(reference_compliance)
    ).numpy()
    dice_values = dice_coefficient(generated_tensor, roi_tensor).numpy()
    return EvaluationReport(
        compliance_error=summarize(compliance_values, bootstrap_resamples),
        stress_shielding_ratio=summarize(stress_ratio, bootstrap_resamples),
        manufacturing=summarize(manufacturing, bootstrap_resamples),
        fatigue_life=summarize(fatigue, bootstrap_resamples),
        dice=summarize(dice_values, bootstrap_resamples),
    )


def validate_surrogate(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = torch.from_numpy(prediction)
    expected = torch.from_numpy(target)
    return {
        "mae": float(mean_absolute_error(predicted, expected)),
        "rmse": float(root_mean_square_error(predicted, expected)),
        "r2": float(coefficient_of_determination(predicted, expected)),
    }
