#!/usr/bin/env python3
"""Route saved FireWave predictions to fixed physical-condition experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .core import FIELD_NAMES, ssim_per_image
except ImportError:
    from firewave.core import FIELD_NAMES, ssim_per_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--experts", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-arrays")
    return parser.parse_args()


def field_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    image_mse = np.mean(error * error, axis=(-1, -2))
    with torch.no_grad():
        pred = torch.from_numpy(prediction.astype(np.float32)).flatten(0, 1)
        truth = torch.from_numpy(target.astype(np.float32)).flatten(0, 1)
        values = [
            ssim_per_image(pred[start:start + 64], truth[start:start + 64])
            for start in range(0, pred.shape[0], 64)
        ]
    return {
        "mse": float(np.mean(image_mse)),
        "ssim": float(torch.cat(values).mean()),
        "psnr": float(np.mean(10.0 * np.log10(1.0 / np.maximum(image_mse, 1e-12)))),
    }


def main() -> None:
    args = parse_args()
    base = np.load(args.base)
    prediction = np.asarray(base["prediction"]).copy()
    target = np.asarray(base["target"])
    scenarios = np.asarray(base["scenario"]).astype(np.int64)
    starts = np.asarray(base["start"]).astype(np.int64)
    base_lookup = {
        (int(scenario), int(start)): index
        for index, (scenario, start) in enumerate(zip(scenarios, starts))
    }

    provenance = []
    replaced_keys: set[tuple[int, int]] = set()
    for expert_path in args.experts:
        expert = np.load(expert_path)
        expert_prediction = np.asarray(expert["prediction"])
        expert_target = np.asarray(expert["target"])
        expert_scenarios = np.asarray(expert["scenario"]).astype(np.int64)
        expert_starts = np.asarray(expert["start"]).astype(np.int64)
        replaced = 0
        for expert_index, (scenario, start) in enumerate(
            zip(expert_scenarios, expert_starts)
        ):
            key = (int(scenario), int(start))
            if key not in base_lookup:
                raise ValueError(f"Expert key {key} is absent from the base arrays")
            base_index = base_lookup[key]
            if not np.allclose(
                expert_target[expert_index], target[base_index], atol=1e-6, rtol=0.0
            ):
                raise ValueError(f"Expert target differs at key {key}")
            prediction[base_index] = expert_prediction[expert_index]
            replaced_keys.add(key)
            replaced += 1
        provenance.append({
            "expert": expert_path,
            "scenario_ids": sorted({int(value) for value in expert_scenarios}),
            "replaced_windows": replaced,
        })

    metrics = {
        name: field_metrics(prediction[:, :, channel], target[:, :, channel])
        for channel, name in enumerate(FIELD_NAMES)
    }
    payload = {
        "base": args.base,
        "experts": provenance,
        "routing": "fixed scenario ID derived from known HRR and ventilation conditions",
        "fallback_windows": int(len(base_lookup) - len(replaced_keys)),
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.save_arrays:
        arrays_path = Path(args.save_arrays)
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_path,
            prediction=prediction,
            target=target,
            scenario=scenarios,
            start=starts,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
