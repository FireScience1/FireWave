#!/usr/bin/env python3
"""Validation-only PSNR search on saved visibility predictions.

The input arrays are already produced by a fixed model/checkpoint.  This
script searches only low-dimensional, validation-fitted postprocessing and
never reads the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .core import ssim_per_image
except ImportError:
    from firewave.core import ssim_per_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("arrays")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-spatial-sigma", type=float, default=0.5)
    parser.add_argument("--input-temporal-blend", type=float, default=0.1)
    parser.add_argument("--undo-input-temporal-blend", type=float)
    parser.add_argument("--save-best-arrays")
    return parser.parse_args()


def smooth(prediction: torch.Tensor, sigma: float, temporal_blend: float) -> torch.Tensor:
    result = prediction
    if sigma > 0.0:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinate = torch.arange(
            -radius,
            radius + 1,
            dtype=result.dtype,
            device=result.device,
        )
        kernel_1d = torch.exp(-coordinate.square() / (2.0 * sigma ** 2))
        kernel_1d /= kernel_1d.sum()
        kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
        images = F.pad(result.flatten(0, 1)[:, None], (radius,) * 4, mode="reflect")
        result = F.conv2d(images, kernel).reshape_as(result)
    if temporal_blend > 0.0:
        previous = torch.cat((result[:, :1], result[:, :-1]), dim=1)
        following = torch.cat((result[:, 1:], result[:, -1:]), dim=1)
        result = (1.0 - temporal_blend) * result + temporal_blend * 0.5 * (
            previous + following
        )
    return result


def undo_temporal_smooth(prediction: np.ndarray, blend: float) -> np.ndarray:
    """Invert the fixed 1-D temporal blend used when the arrays were saved."""
    if not 0.0 < blend < 1.0:
        raise ValueError("undo temporal blend must be in (0, 1)")
    frames = prediction.shape[1]
    matrix = np.zeros((frames, frames), dtype=np.float64)
    matrix[np.arange(frames), np.arange(frames)] = 1.0 - blend
    matrix[np.arange(1, frames), np.arange(frames - 1)] = 0.5 * blend
    matrix[np.arange(frames - 1), np.arange(1, frames)] = 0.5 * blend
    matrix[0, 0] += 0.5 * blend
    matrix[-1, -1] += 0.5 * blend
    flattened = prediction.transpose(1, 0, 2, 3).reshape(frames, -1)
    restored = np.linalg.solve(matrix, flattened.astype(np.float64))
    restored = restored.reshape(frames, *prediction.shape[:1], *prediction.shape[2:])
    return np.clip(restored.transpose(1, 0, 2, 3), 0.0, 1.0).astype(np.float32)


def basic_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    image_mse = np.mean(np.square(prediction - target), axis=(-1, -2))
    mse = float(image_mse.mean())
    psnr = float(np.mean(10.0 * np.log10(1.0 / np.maximum(image_mse, 1e-12))))
    return mse, psnr


def ssim_metric(prediction: np.ndarray, target: np.ndarray) -> float:
    values = []
    with torch.no_grad():
        pred = torch.from_numpy(prediction.astype(np.float32)).flatten(0, 1)
        truth = torch.from_numpy(target.astype(np.float32)).flatten(0, 1)
        for start in range(0, pred.shape[0], 64):
            values.append(ssim_per_image(pred[start : start + 64], truth[start : start + 64]))
    return float(torch.cat(values).mean().item())


def best_affine(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = prediction.reshape(-1, prediction.shape[-2] * prediction.shape[-1]).astype(np.float64)
    y = target.reshape(-1, target.shape[-2] * target.shape[-1]).astype(np.float64)
    ex2 = np.mean(x * x, axis=1)
    ex = np.mean(x, axis=1)
    exy = np.mean(x * y, axis=1)
    ey = np.mean(y, axis=1)
    ey2 = np.mean(y * y, axis=1)
    best = (-float("inf"), 1.0, 0.0)
    for scale in np.linspace(0.97, 1.03, 49):
        for bias in np.linspace(-0.005, 0.005, 41):
            image_mse = (
                scale * scale * ex2
                + 2.0 * scale * bias * ex
                + bias * bias
                - 2.0 * scale * exy
                - 2.0 * bias * ey
                + ey2
            )
            psnr = float(np.mean(10.0 * np.log10(1.0 / np.maximum(image_mse, 1e-12))))
            if psnr > best[0]:
                best = (psnr, float(scale), float(bias))
    return best[1], best[2]


def main() -> None:
    args = parse_args()
    arrays = np.load(args.arrays)
    base = arrays["prediction"][:, :, 1].astype(np.float32)
    target = arrays["target"][:, :, 1].astype(np.float32)
    base_mse, base_psnr = basic_metrics(base, target)
    baseline = {
        "extra_spatial_sigma": 0.0,
        "extra_temporal_blend": 0.0,
        "scale": 1.0,
        "bias": 0.0,
        "mse": base_mse,
        "ssim": ssim_metric(base, target),
        "psnr": base_psnr,
    }

    search_base = (
        undo_temporal_smooth(base, args.undo_input_temporal_blend)
        if args.undo_input_temporal_blend is not None
        else base
    )
    tensor = torch.from_numpy(search_base)
    candidates = []
    with torch.no_grad():
        for sigma in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
            for temporal_blend in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
                filtered = smooth(tensor, sigma, temporal_blend).numpy()
                scale, bias = best_affine(filtered, target)
                calibrated = np.clip(filtered * scale + bias, 0.0, 1.0)
                mse, psnr = basic_metrics(calibrated, target)
                candidates.append({
                    "extra_spatial_sigma": sigma,
                    "extra_temporal_blend": temporal_blend,
                    "scale": scale,
                    "bias": bias,
                    "mse": mse,
                    "psnr": psnr,
                })

    top = sorted(candidates, key=lambda item: item["psnr"], reverse=True)[:10]
    for candidate in top:
        filtered = smooth(
            tensor,
            candidate["extra_spatial_sigma"],
            candidate["extra_temporal_blend"],
        ).numpy()
        calibrated = np.clip(
            filtered * candidate["scale"] + candidate["bias"], 0.0, 1.0
        )
        candidate["ssim"] = ssim_metric(calibrated, target)

    result = {
        "source": str(Path(args.arrays)),
        "fit_split": "validation",
        "input_postprocessing": {
            "spatial_sigma": args.input_spatial_sigma,
            "temporal_blend": args.input_temporal_blend,
        },
        "search_input": {
            "undo_input_temporal_blend": args.undo_input_temporal_blend,
        },
        "selection_metric": "mean per-frame PSNR with data_range=1",
        "baseline": baseline,
        "top_candidates": top,
    }
    if args.save_best_arrays:
        selected = top[0]
        filtered = smooth(
            tensor,
            selected["extra_spatial_sigma"],
            selected["extra_temporal_blend"],
        ).numpy()
        calibrated = np.clip(
            filtered * selected["scale"] + selected["bias"], 0.0, 1.0
        )
        full_prediction = arrays["prediction"].copy()
        full_prediction[:, :, 1] = calibrated.astype(full_prediction.dtype)
        arrays_output = Path(args.save_best_arrays)
        arrays_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_output,
            prediction=full_prediction,
            target=arrays["target"],
            scenario=arrays["scenario"],
            start=arrays["start"],
        )
        result["saved_best_arrays"] = str(arrays_output)
        result["saved_candidate"] = selected
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
