from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compscore_to.evaluation.report import evaluate_arrays


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="compscore-evaluate")
    result.add_argument("--results", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--bootstrap-resamples", type=int, default=10000)
    return result


def main() -> None:
    arguments = parser().parse_args()
    arrays = np.load(arguments.results)
    report = evaluate_arrays(
        arrays["generated_compliance"],
        arrays["reference_compliance"],
        arrays["stress_ratio"],
        arrays["manufacturing"],
        arrays["fatigue"],
        arrays["generated_geometry"],
        arrays["anatomy_roi"],
        arguments.bootstrap_resamples,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report.mapping(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
