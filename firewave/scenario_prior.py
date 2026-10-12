#!/usr/bin/env python3
"""Strict frame-disjoint scenario/time interpolation diagnostic for FIT.

The retrieval bank is built only from frames used by strict training rows.
Validation and held-out history/target frame keys are removed before any field
is loaded into the bank.  Held-out target arrays are never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .metrics import field_metrics
    from .direct_predictor import (
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from .core import FITStore, make_segment_index, split_segment_index
    from .visibility_encoding import (
        decode_visibility,
        encode_visibility,
        interpolate_frame,
        target_indices,
    )
except ImportError:
    from firewave.metrics import field_metrics
    from firewave.direct_predictor import (
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from firewave.core import FITStore, make_segment_index, split_segment_index
    from firewave.visibility_encoding import (
        decode_visibility,
        encode_visibility,
        interpolate_frame,
        target_indices,
    )


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_string_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


HRR_LEVELS = np.asarray((0.0, 0.3010299957, 0.6020599913, 1.0), dtype=np.float64)
HRR_RATIOS = np.asarray((1.0, 2.0, 4.0, 10.0), dtype=np.float64)


def predict_encoded_trajectory(
    scenario: int,
    query_times: np.ndarray,
    encoded_bank: dict[int, dict[int, np.ndarray]],
    frame_times: dict[int, np.ndarray],
    method: str,
    condition_degree: int,
    condition_coordinate_power: float | None = None,
    condition_time_fallback: bool = False,
    condition_fallback_distance: int = 64,
) -> np.ndarray:
    """Interpolate in time, with target-free HRR extrapolation if needed."""

    if scenario in encoded_bank and not condition_time_fallback:
        return np.stack(
            [
                interpolate_frame(
                    encoded_bank[scenario],
                    frame_times[scenario],
                    int(time),
                    method,
                )
                for time in query_times
            ]
        ).astype(np.float32)

    coordinate_levels = (
        HRR_LEVELS
        if condition_coordinate_power is None
        or abs(condition_coordinate_power) < 1e-12
        else np.power(HRR_RATIOS, condition_coordinate_power)
    )
    predictions = []
    for time in query_times:
        if scenario in encoded_bank:
            scenario_times = frame_times[scenario]
            if (
                not condition_time_fallback
                or int(scenario_times[0]) - condition_fallback_distance
                <= int(time)
                <= int(scenario_times[-1]) + condition_fallback_distance
            ):
                predictions.append(
                    interpolate_frame(
                        encoded_bank[scenario],
                        scenario_times,
                        int(time),
                        method,
                    )
                )
                continue

        ventilation = scenario // 4
        source_scenarios = sorted(
            source
            for source in encoded_bank
            if source // 4 == ventilation and source != scenario
        )
        if len(source_scenarios) < 2:
            raise RuntimeError(
                f"Scenario {scenario} needs at least two same-ventilation source scenarios"
            )
        if condition_time_fallback:
            def temporal_distance(source: int) -> int:
                times = frame_times[source]
                if int(time) < int(times[0]):
                    return int(times[0]) - int(time)
                if int(time) > int(times[-1]):
                    return int(time) - int(times[-1])
                return 0

            source_scenarios = sorted(
                source_scenarios,
                key=lambda source: (
                    temporal_distance(source),
                    abs((source % 4) - (scenario % 4)),
                    source,
                ),
            )[: max(2, condition_degree + 1)]
        degree = min(condition_degree, len(source_scenarios) - 1)
        source_hrr = coordinate_levels[[source % 4 for source in source_scenarios]]
        design = np.vander(source_hrr, N=degree + 1, increasing=True)
        query_design = np.asarray(
            [
                coordinate_levels[scenario % 4] ** exponent
                for exponent in range(degree + 1)
            ],
            dtype=np.float64,
        )
        weights = query_design @ np.linalg.pinv(design)
        source_fields = np.stack(
            [
                interpolate_frame(
                    encoded_bank[source],
                    frame_times[source],
                    int(time),
                    method,
                )
                for source in source_scenarios
            ]
        ).astype(np.float32)
        predictions.append(
            np.tensordot(weights.astype(np.float32), source_fields, axes=(0, 0))
        )
    return np.stack(predictions).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-arrays", default="")
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--strict-protected-min-span", type=int, default=640)
    parser.add_argument("--train-min-span", type=int, default=0)
    parser.add_argument(
        "--visibility-powers",
        default="-2,-1.5,-1,-0.75,-0.5,0,0.5,1,2",
    )
    parser.add_argument("--methods", default="linear,pchip")
    parser.add_argument("--condition-degrees", default="1,2")
    parser.add_argument(
        "--condition-coordinate-powers",
        default="0",
        help="Comma-separated powers of HRR/5; zero selects log10(HRR/5)",
    )
    parser.add_argument("--condition-time-fallback", action="store_true")
    parser.add_argument("--condition-fallback-distance", type=int, default=64)
    args = parser.parse_args()

    if args.frames != 32:
        raise ValueError("Strict FIT comparison currently requires 32 frames")

    store = FITStore(args.data_root)
    segment_index = make_segment_index(history_frames=args.history_frames)
    _, validation, heldout = split_segment_index(segment_index, args.split_seed)
    validation = [
        row for row in validation if row[2] >= args.strict_protected_min_span
    ]
    heldout = [
        row for row in heldout if row[2] >= args.strict_protected_min_span
    ]
    train = strict_frame_disjoint_variable_training_rows(
        validation,
        heldout,
        args.history_frames,
        int(store.data.shape[1]),
        {row[2] for row in segment_index},
        {row[0] for row in segment_index},
    )
    if args.train_min_span:
        train = [row for row in train if row[2] >= args.train_min_span]
    if not train:
        raise RuntimeError("Strict interpolation bank has no training rows")

    training_frame_keys = target_keys(train, args.history_frames) | history_keys(
        train, args.history_frames
    )
    protected_frame_keys = target_keys(
        validation + heldout, args.history_frames
    ) | history_keys(validation + heldout, args.history_frames)
    overlap = training_frame_keys & protected_frame_keys
    if overlap:
        raise RuntimeError(f"Strict frame overlap: {sorted(overlap)[:5]}")

    frame_times: dict[int, np.ndarray] = {}
    raw_visibility: dict[int, np.ndarray] = {}
    for scenario in sorted({int(row[0]) for row in segment_index}):
        times = np.asarray(
            sorted(time for key_scenario, time in training_frame_keys if key_scenario == scenario),
            dtype=np.int64,
        )
        if times.size == 0:
            continue
        frame_times[scenario] = times
        raw_visibility[scenario] = store.data[scenario, times, 1].astype(np.float32)

    target_visibility = []
    validation_times = []
    validation_scenarios = []
    validation_starts = []
    for scenario, start, span in validation:
        times = target_indices(start, span, args.frames, args.history_frames)
        target_visibility.append(
            store.normalize(store.data[scenario, times]).astype(np.float32)[:, 1]
        )
        validation_times.append(times)
        validation_scenarios.append(int(scenario))
        validation_starts.append(int(start))
    target = np.stack(target_visibility).astype(np.float32)

    candidates = []
    for method in parse_string_list(args.methods):
        if method not in {"linear", "pchip"}:
            raise ValueError(f"Unsupported method: {method}")
        for power in parse_float_list(args.visibility_powers):
            for condition_degree in (
                int(value) for value in parse_float_list(args.condition_degrees)
            ):
                for coordinate_power in parse_float_list(
                    args.condition_coordinate_powers
                ):
                    encoded_bank = {}
                    for scenario, times in frame_times.items():
                        encoded_values = encode_visibility(
                            raw_visibility[scenario], power
                        )
                        encoded_bank[scenario] = {
                            int(time): frame
                            for time, frame in zip(times, encoded_values)
                        }

                    predictions = []
                    for scenario, times in zip(
                        validation_scenarios, validation_times
                    ):
                        encoded_prediction = predict_encoded_trajectory(
                            scenario,
                            times,
                            encoded_bank,
                            frame_times,
                            method,
                            condition_degree,
                            coordinate_power,
                            args.condition_time_fallback,
                            args.condition_fallback_distance,
                        )
                        physical_prediction = decode_visibility(
                            encoded_prediction, power
                        )
                        normalized_prediction = (
                            physical_prediction - float(store.minimum[1])
                        ) / float(store.scale[1])
                        predictions.append(
                            np.clip(normalized_prediction, 0.0, 1.0).astype(
                                np.float32
                            )
                        )
                    prediction = np.stack(predictions).astype(np.float32)
                    full = field_metrics(prediction, target)
                    fraction_metrics = {}
                    for fraction, index in zip(
                        ("0.25", "0.5", "0.75", "1.0"),
                        (8, 16, 23, 31),
                    ):
                        fraction_metrics[fraction] = field_metrics(
                            prediction[:, index : index + 1],
                            target[:, index : index + 1],
                        )
                    candidate = {
                        "method": method,
                        "visibility_power": power,
                        "condition_degree": condition_degree,
                        "condition_coordinate_power": coordinate_power,
                        "full_metrics": full,
                        "fraction_metrics": fraction_metrics,
                    }
                    candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item["full_metrics"]["ssim"]
            if np.isfinite(item["full_metrics"]["ssim"])
            else -float("inf")
        ),
        reverse=True,
    )
    best = candidates[0]

    # Reconstruct only the selected target-independent candidate for optional saving.
    method = str(best["method"])
    power = float(best["visibility_power"])
    condition_degree = int(best["condition_degree"])
    condition_coordinate_power = float(best["condition_coordinate_power"])
    encoded_bank = {
        scenario: {
            int(time): frame
            for time, frame in zip(times, encode_visibility(raw_visibility[scenario], power))
        }
        for scenario, times in frame_times.items()
    }
    selected_predictions = []
    for scenario, times in zip(validation_scenarios, validation_times):
        encoded_prediction = predict_encoded_trajectory(
            scenario,
            times,
            encoded_bank,
            frame_times,
            method,
            condition_degree,
            condition_coordinate_power,
            args.condition_time_fallback,
            args.condition_fallback_distance,
        )
        physical_prediction = decode_visibility(encoded_prediction, power)
        selected_predictions.append(
            np.clip(
                (physical_prediction - float(store.minimum[1]))
                / float(store.scale[1]),
                0.0,
                1.0,
            ).astype(np.float32)
        )
    selected_prediction = np.stack(selected_predictions).astype(np.float32)

    payload = {
        "evaluation_split": "validation",
        "test_accessed": False,
        "model": "strict train-frame-only scenario-time interpolation diagnostic",
        "protocol": {
            "minimum_span": args.strict_protected_min_span,
            "training_minimum_span": args.train_min_span,
            "training_windows": len(train),
            "validation_samples": len(validation),
            "heldout_samples_reserved": len(heldout),
            "training_frame_keys": len(training_frame_keys),
            "protected_validation_test_frame_keys": len(protected_frame_keys),
            "train_evaluation_frame_overlap": len(overlap),
            "candidate_count": len(candidates),
            "selection_metric": "full 32-frame validation visibility SSIM",
            "condition_time_fallback": args.condition_time_fallback,
            "condition_fallback_distance": args.condition_fallback_distance,
        },
        "selected": best,
        "candidates": candidates,
        "saved_arrays": args.save_arrays or None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.save_arrays:
        arrays = Path(args.save_arrays)
        arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays,
            prediction=selected_prediction,
            target=target,
            scenario=np.asarray(validation_scenarios, dtype=np.int64),
            start=np.asarray(validation_starts, dtype=np.int64),
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
