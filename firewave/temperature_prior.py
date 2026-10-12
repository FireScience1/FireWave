#!/usr/bin/env python3
"""Strict train-frame-only condition/time prior for FIT temperature fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .metrics import field_metrics
    from .scenario_prior import predict_encoded_trajectory
    from .direct_predictor import (
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from .core import FITStore, make_segment_index, split_segment_index
    from .visibility_encoding import target_indices
except ImportError:
    from firewave.metrics import field_metrics
    from firewave.scenario_prior import predict_encoded_trajectory
    from firewave.direct_predictor import (
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from firewave.core import FITStore, make_segment_index, split_segment_index
    from firewave.visibility_encoding import target_indices


def floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def encode_temperature(temperature: np.ndarray, minimum: float, power: float) -> np.ndarray:
    excess = np.maximum(temperature.astype(np.float32) - minimum, 1e-4)
    if abs(power) < 1e-12:
        return np.log(excess).astype(np.float32)
    return np.power(excess, power).astype(np.float32)


def decode_temperature(encoded: np.ndarray, minimum: float, power: float) -> np.ndarray:
    if abs(power) < 1e-12:
        excess = np.exp(encoded)
    else:
        excess = np.power(np.maximum(encoded, 1e-12), 1.0 / power)
    return (excess + minimum).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-arrays", required=True)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--strict-protected-min-span", type=int, default=640)
    parser.add_argument("--temperature-powers", default="0,0.25,0.5,0.75,1,1.5,2,3")
    parser.add_argument("--methods", default="linear,pchip")
    parser.add_argument("--condition-degrees", default="1,2")
    args = parser.parse_args()

    store = FITStore(args.data_root)
    segment_index = make_segment_index(history_frames=args.history_frames)
    _, validation, heldout = split_segment_index(segment_index, args.split_seed)
    validation = [row for row in validation if row[2] >= args.strict_protected_min_span]
    heldout = [row for row in heldout if row[2] >= args.strict_protected_min_span]
    train = strict_frame_disjoint_variable_training_rows(
        validation,
        heldout,
        args.history_frames,
        int(store.data.shape[1]),
        {row[2] for row in segment_index},
        {row[0] for row in segment_index},
    )
    training_frame_keys = target_keys(train, args.history_frames) | history_keys(
        train, args.history_frames
    )
    protected_frame_keys = target_keys(
        validation + heldout, args.history_frames
    ) | history_keys(validation + heldout, args.history_frames)
    overlap = training_frame_keys & protected_frame_keys
    if overlap:
        raise RuntimeError(f"Strict frame overlap: {sorted(overlap)[:5]}")

    frame_times = {}
    raw_temperature = {}
    for scenario in range(8):
        times = np.asarray(
            sorted(time for key_scenario, time in training_frame_keys if key_scenario == scenario),
            dtype=np.int64,
        )
        if times.size:
            frame_times[scenario] = times
            raw_temperature[scenario] = store.data[scenario, times, 0].astype(np.float32)

    scenarios = np.asarray([int(row[0]) for row in validation], dtype=np.int64)
    starts = np.asarray([int(row[1]) for row in validation], dtype=np.int64)
    query_times = [
        target_indices(start, span, 32, args.history_frames)
        for _, start, span in validation
    ]
    target = np.stack(
        [
            store.normalize(store.data[scenario, times]).astype(np.float32)[:, 0]
            for (scenario, _, _), times in zip(validation, query_times)
        ]
    ).astype(np.float32)

    minimum = float(store.minimum[0])
    scale = float(store.scale[0])
    candidates = []
    candidate_predictions = []
    for method in strings(args.methods):
        for power in floats(args.temperature_powers):
            encoded_bank = {
                scenario: {
                    int(time): field
                    for time, field in zip(
                        times,
                        encode_temperature(raw_temperature[scenario], minimum, power),
                    )
                }
                for scenario, times in frame_times.items()
            }
            for degree_value in floats(args.condition_degrees):
                degree = int(degree_value)
                predictions = []
                for scenario, times in zip(scenarios.tolist(), query_times):
                    encoded = predict_encoded_trajectory(
                        scenario,
                        times,
                        encoded_bank,
                        frame_times,
                        method,
                        degree,
                    )
                    physical = decode_temperature(encoded, minimum, power)
                    predictions.append(
                        np.clip((physical - minimum) / scale, 0.0, 1.0).astype(np.float32)
                    )
                prediction = np.stack(predictions).astype(np.float32)
                full = field_metrics(prediction, target)
                fractions = {
                    fraction: field_metrics(
                        prediction[:, index : index + 1],
                        target[:, index : index + 1],
                    )
                    for fraction, index in zip(
                        ("0.25", "0.5", "0.75", "1.0"), (8, 16, 23, 31)
                    )
                }
                candidates.append(
                    {
                        "method": method,
                        "temperature_power": power,
                        "condition_degree": degree,
                        "full_metrics": full,
                        "fraction_metrics": fractions,
                    }
                )
                candidate_predictions.append(prediction)

    order = sorted(
        range(len(candidates)),
        key=lambda index: candidates[index]["full_metrics"]["ssim"],
        reverse=True,
    )
    candidates = [candidates[index] for index in order]
    candidate_predictions = [candidate_predictions[index] for index in order]
    selected = candidates[0]
    selected_prediction = candidate_predictions[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_split": "validation",
        "test_accessed": False,
        "model": "FireWave strict train-frame-only temperature condition prior",
        "protocol": {
            "minimum_span": args.strict_protected_min_span,
            "training_windows": len(train),
            "validation_samples": len(validation),
            "heldout_samples_reserved": len(heldout),
            "training_frame_keys": len(training_frame_keys),
            "protected_validation_test_frame_keys": len(protected_frame_keys),
            "train_evaluation_frame_overlap": len(overlap),
            "candidate_count": len(candidates),
            "selection_metric": "full 32-frame validation temperature SSIM",
        },
        "selected": selected,
        "candidates": candidates,
        "saved_arrays": args.save_arrays,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    arrays = Path(args.save_arrays)
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays,
        prediction=selected_prediction,
        target=target,
        scenario=scenarios,
        start=starts,
    )
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
