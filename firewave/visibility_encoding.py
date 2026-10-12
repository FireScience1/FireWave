#!/usr/bin/env python3
"""Train-only scenario/time interpolation prior for FIT trajectory fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .metrics import field_metrics
    from .core import (
        FIELD_NAMES,
        FITStore,
        make_segment_index,
        split_segment_index,
    )
except ImportError:
    from firewave.metrics import field_metrics
    from firewave.core import FIELD_NAMES, FITStore, make_segment_index, split_segment_index


def target_indices(start: int, span: int, frames: int, history_frames: int) -> np.ndarray:
    forecast_start = start + history_frames
    return np.rint(
        np.linspace(forecast_start, forecast_start + span, frames)
    ).astype(np.int64)


def interpolate_frame(
    frame_by_time: dict[int, np.ndarray],
    sorted_times: np.ndarray,
    query_time: int,
    method: str,
) -> np.ndarray:
    position = int(np.searchsorted(sorted_times, query_time))
    if position == 0:
        return frame_by_time[int(sorted_times[0])]
    if position == len(sorted_times):
        return frame_by_time[int(sorted_times[-1])]
    before = int(sorted_times[position - 1])
    after = int(sorted_times[position])
    if before == after:
        return frame_by_time[before]
    weight = float(query_time - before) / float(after - before)
    linear = (
        frame_by_time[before] * (1.0 - weight)
        + frame_by_time[after] * weight
    )
    if method == "linear" or position < 2 or position + 1 >= len(sorted_times):
        return linear

    earlier = int(sorted_times[position - 2])
    later = int(sorted_times[position + 1])
    y0 = frame_by_time[earlier]
    y1 = frame_by_time[before]
    y2 = frame_by_time[after]
    y3 = frame_by_time[later]
    h0 = float(before - earlier)
    h1 = float(after - before)
    h2 = float(later - after)
    d0 = (y1 - y0) / h0
    d1 = (y2 - y1) / h1
    d2 = (y3 - y2) / h2

    def monotone_slope(
        left: np.ndarray,
        right: np.ndarray,
        left_width: float,
        right_width: float,
    ) -> np.ndarray:
        slope = np.zeros_like(left, dtype=np.float32)
        mask = left * right > 0.0
        if np.any(mask):
            first_weight = 2.0 * right_width + left_width
            second_weight = right_width + 2.0 * left_width
            slope[mask] = (first_weight + second_weight) / (
                first_weight / left[mask] + second_weight / right[mask]
            )
        return slope

    slope_before = monotone_slope(d0, d1, h0, h1)
    slope_after = monotone_slope(d1, d2, h1, h2)
    t = weight
    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y1
        + (t3 - 2.0 * t2 + t) * h1 * slope_before
        + (-2.0 * t3 + 3.0 * t2) * y2
        + (t3 - t2) * h1 * slope_after
    ).astype(np.float32)


def encode_visibility(visibility: np.ndarray, power: float) -> np.ndarray:
    """Map positive physical visibility to a temporal interpolation domain."""

    safe = np.maximum(visibility.astype(np.float32), 1e-6)
    if abs(power) < 1e-12:
        return np.log(safe).astype(np.float32)
    return np.power(safe, power).astype(np.float32)


def decode_visibility(encoded: np.ndarray, power: float) -> np.ndarray:
    """Invert :func:`encode_visibility` after temporal interpolation."""

    if abs(power) < 1e-12:
        return np.exp(encoded).astype(np.float32)
    safe = np.maximum(encoded.astype(np.float32), 1e-12)
    return np.power(safe, 1.0 / power).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--evaluation-split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-arrays", required=True)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument(
        "--visibility-power",
        type=float,
        default=1.0,
        help=(
            "Power-domain interpolation for positive physical visibility; "
            "0 selects log visibility and -1 selects inverse visibility "
            "(proportional to optical extinction)."
        ),
    )
    parser.add_argument(
        "--interpolation-method",
        choices=("linear", "pchip"),
        default="linear",
        help="Temporal interpolation rule applied independently at each pixel.",
    )
    args = parser.parse_args()

    store = FITStore(args.data_root)
    train, validation, test = split_segment_index(
        make_segment_index(history_frames=args.history_frames), args.split_seed
    )
    train = [row for row in train if row[2] == args.frames - 1]
    evaluation = validation if args.evaluation_split == "validation" else test
    evaluation = [row for row in evaluation if row[2] == args.frames - 1]

    frame_maps: dict[int, dict[int, np.ndarray]] = {}
    train_keys: set[tuple[int, int]] = set()
    for scenario, start, span in train:
        indices = target_indices(start, span, args.frames, args.history_frames)
        normalized = store.normalize(store.data[scenario, indices]).astype(np.float32)
        normalized[:, 1] = encode_visibility(
            store.data[scenario, indices, 1], args.visibility_power
        )
        scenario_map = frame_maps.setdefault(int(scenario), {})
        for time_index, frame in zip(indices, normalized):
            key = (int(scenario), int(time_index))
            if key in train_keys:
                raise ValueError(f"Duplicate train target time index: {key}")
            train_keys.add(key)
            scenario_map[int(time_index)] = frame

    sorted_times = {
        scenario: np.asarray(sorted(frame_map), dtype=np.int64)
        for scenario, frame_map in frame_maps.items()
    }
    predictions = []
    targets = []
    scenarios = []
    starts = []
    evaluation_keys: set[tuple[int, int]] = set()
    for scenario, start, span in evaluation:
        indices = target_indices(start, span, args.frames, args.history_frames)
        evaluation_keys.update((int(scenario), int(value)) for value in indices)
        frame_map = frame_maps[int(scenario)]
        times = sorted_times[int(scenario)]
        prediction = np.stack([
            interpolate_frame(
                frame_map,
                times,
                int(value),
                args.interpolation_method,
            )
            for value in indices
        ]).astype(np.float32)
        physical_visibility = decode_visibility(
            prediction[:, 1], args.visibility_power
        )
        prediction[:, 1] = (
            physical_visibility - float(store.minimum[1])
        ) / float(store.scale[1])
        predictions.append(prediction)
        targets.append(
            store.normalize(store.data[scenario, indices]).astype(np.float32)
        )
        scenarios.append(int(scenario))
        starts.append(int(start))

    overlap = train_keys & evaluation_keys
    if overlap:
        raise RuntimeError(
            f"Train/evaluation target-time overlap detected: {sorted(overlap)[:5]}"
        )

    prediction = np.stack(predictions).astype(np.float32)
    target = np.stack(targets).astype(np.float32)
    scenario_array = np.asarray(scenarios, dtype=np.int64)
    start_array = np.asarray(starts, dtype=np.int64)
    metrics = {
        name: field_metrics(prediction[:, :, channel], target[:, :, channel])
        for channel, name in enumerate(FIELD_NAMES)
    }
    payload = {
        "model": "train-only scenario-time power-domain interpolation prior",
        "evaluation_split": args.evaluation_split,
        "split_seed": args.split_seed,
        "history_frames": args.history_frames,
        "forecast_frames": args.frames,
        "interpolation_method": args.interpolation_method,
        "visibility_power": args.visibility_power,
        "visibility_domain": (
            "log-visibility"
            if abs(args.visibility_power) < 1e-12
            else (
                "inverse-visibility optical-extinction proxy"
                if abs(args.visibility_power + 1.0) < 1e-12
                else f"physical-visibility power {args.visibility_power:g}"
            )
        ),
        "training_target_frames": len(train_keys),
        "evaluation_target_frames": len(evaluation_keys),
        "train_evaluation_target_overlap": 0,
        "selection_or_fitting_on_evaluation_split": False,
        "metrics": metrics,
        "saved_arrays": args.save_arrays,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    arrays_path = Path(args.save_arrays)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        prediction=prediction,
        target=target,
        scenario=scenario_array,
        start=start_array,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
