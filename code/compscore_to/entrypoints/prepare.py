from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from compscore_to.data.preprocess import VolumePreprocessor


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="compscore-prepare")
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--site", required=True)
    return result


def main() -> None:
    arguments = parser().parse_args()
    preprocessor = VolumePreprocessor()
    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in sorted(arguments.input.glob("*.nii.gz")):
        image = nib.load(source)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        mask = np.asarray(image.dataobj)
        processed = preprocessor.process_mask(mask, spacing)
        destination = arguments.output / f"{source.name.removesuffix('.nii.gz')}.npy"
        np.save(destination, processed)
        manifest.append(
            {"identifier": source.name, "anatomy": destination.name, "site": arguments.site}
        )
    (arguments.output / "anatomy_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
