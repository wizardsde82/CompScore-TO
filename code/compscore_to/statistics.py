from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Comparison:
    statistic: float
    p_value: float
    adjusted_p_value: float
    cliffs_delta: float


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    differences = left.reshape(-1, 1) - right.reshape(1, -1)
    return float(
        (np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / differences.size
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def paired_comparisons(reference: np.ndarray, candidates: list[np.ndarray]) -> list[Comparison]:
    raw = [stats.wilcoxon(reference, candidate) for candidate in candidates]
    adjusted = benjamini_hochberg(np.asarray([result.pvalue for result in raw]))
    return [
        Comparison(
            float(result.statistic),
            float(result.pvalue),
            float(correction),
            cliffs_delta(reference, candidate),
        )
        for result, correction, candidate in zip(raw, adjusted, candidates, strict=True)
    ]


def bootstrap_mean_interval(
    values: np.ndarray, resamples: int = 10000, confidence: float = 0.95, seed: int = 2026
) -> tuple[float, float]:
    result = stats.bootstrap(
        (values,),
        np.mean,
        n_resamples=resamples,
        confidence_level=confidence,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)
