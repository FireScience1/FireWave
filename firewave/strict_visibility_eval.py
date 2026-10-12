#!/usr/bin/env python3
"""One-shot frozen held-out evaluation of a validation-selected strict prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .metrics import field_metrics
    from .postprocess import smooth
    from .scenario_prior import predict_encoded_trajectory
    from .direct_predictor import (
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from .core import FITStore, make_segment_index, split_segment_index
    from .visibility_encoding import (
        decode_visibility,
        encode_visibility,
        target_indices,
    )
except ImportError:
    from firewave.metrics import field_metrics
    from firewave.postprocess import smooth
    from firewave.scenario_prior import predict_encoded_trajectory
    from firewave.direct_predictor import (
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from firewave.core import FITStore, make_segment_index, split_segment_index
    from firewave.visibility_encoding import (
        decode_visibility,
        encode_visibility,
        target_indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--strict-protected-min-span", type=int, default=640)
    args = parser.parse_args()

    selection_path = Path(args.selection_json)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection_payload.get("test_accessed") is not False:
        raise RuntimeError("Selection source must be validation-only")
    configuration = selection_payload["full"]["configuration"]

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
    raw_visibility = {}
    for scenario in range(8):
        times = np.asarray(
            sorted(time for key_scenario, time in training_frame_keys if key_scenario == scenario),
            dtype=np.int64,
        )
        if times.size:
            frame_times[scenario] = times
            raw_visibility[scenario] = store.data[scenario, times, 1].astype(np.float32)

    power = float(configuration["power"])
    method = str(configuration["method"])
    condition_degree = int(configuration["condition_degree"])
    encoded_bank = {
        scenario: {
            int(time): field
            for time, field in zip(
                times,
                encode_visibility(raw_visibility[scenario], power),
            )
        }
        for scenario, times in frame_times.items()
    }

    scenarios = np.asarray([int(row[0]) for row in heldout], dtype=np.int64)
    starts = np.asarray([int(row[1]) for row in heldout], dtype=np.int64)
    predictions = []
    targets = []
    for scenario, start, span in heldout:
        times = target_indices(start, span, 32, args.history_frames)
        encoded = predict_encoded_trajectory(
            int(scenario),
            times,
            encoded_bank,
            frame_times,
            method,
            condition_degree,
        )
        physical = decode_visibility(encoded, power)
        predictions.append(
            np.clip(
                (physical - float(store.minimum[1])) / float(store.scale[1]),
                0.0,
                1.0,
            ).astype(np.float32)
        )
        targets.append(
            store.normalize(store.data[scenario, times]).astype(np.float32)[:, 1]
        )
    target = np.stack(targets).astype(np.float32)
    with torch.no_grad():
        prediction = smooth(
            torch.from_numpy(np.stack(predictions).astype(np.float32)),
            float(configuration["spatial_sigma"]),
            float(configuration["temporal_blend"]),
        ).numpy()
    prediction = np.clip(
        prediction * float(configuration["scale"]), 0.0, 1.0
    ).astype(np.float32)

    fractions = {
        fraction: field_metrics(
            prediction[:, index : index + 1],
            target[:, index : index + 1],
        )
        for fraction, index in zip(
            ("0.25", "0.5", "0.75", "1.0"), (8, 16, 23, 31)
        )
    }
    per_scenario = {}
    for scenario in sorted(set(scenarios.tolist())):
        selected = scenarios == scenario
        per_scenario[str(int(scenario))] = field_metrics(
            prediction[selected], target[selected]
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "heldout_arrays.npz",
        prediction=prediction,
        target=target,
        scenario=scenarios,
        start=starts,
    )
    result = {
        "evaluation_split": "heldout",
        "heldout_accessed": True,
        "test_accessed": True,
        "selection_or_calibration_on_evaluation_split": False,
        "model": "FireWave strict conditional physical prior frozen evaluation",
        "selection_source": str(selection_path),
        "configuration": configuration,
        "full_metrics": field_metrics(prediction, target),
        "fraction_metrics": fractions,
        "per_scenario": per_scenario,
        "data_audit": {
            "protocol": "strict frame-disjoint dense variable-span windows",
            "protected_min_span": args.strict_protected_min_span,
            "training_windows": len(train),
            "validation_windows_reserved": len(validation),
            "heldout_windows": len(heldout),
            "training_frame_keys": len(training_frame_keys),
            "protected_validation_test_frame_keys": len(protected_frame_keys),
            "train_evaluation_frame_overlap": len(overlap),
        },
    }
    (output / "heldout_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
