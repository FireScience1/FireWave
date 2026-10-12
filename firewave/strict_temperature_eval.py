#!/usr/bin/env python3
"""One-shot frozen held-out evaluation of the strict temperature prior."""

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
    from .temperature_prior import (
        decode_temperature,
        encode_temperature,
    )
    from .direct_predictor import (
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from .core import FITStore, make_segment_index, split_segment_index
    from .visibility_encoding import target_indices
except ImportError:
    from firewave.metrics import field_metrics
    from firewave.postprocess import smooth
    from firewave.scenario_prior import predict_encoded_trajectory
    from firewave.temperature_prior import (
        decode_temperature,
        encode_temperature,
    )
    from firewave.direct_predictor import (
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from firewave.core import FITStore, make_segment_index, split_segment_index
    from firewave.visibility_encoding import target_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prior-selection-json", required=True)
    parser.add_argument("--postprocess-selection-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--strict-protected-min-span", type=int, default=640)
    args = parser.parse_args()

    prior_payload = json.loads(
        Path(args.prior_selection_json).read_text(encoding="utf-8")
    )
    gate_payload = json.loads(
        Path(args.postprocess_selection_json).read_text(encoding="utf-8")
    )
    if prior_payload.get("test_accessed") is not False:
        raise RuntimeError("Prior selection must be validation-only")
    if gate_payload.get("test_accessed") is not False:
        raise RuntimeError("Postprocessing selection must be validation-only")
    prior_config = prior_payload["selected"]
    postprocess = gate_payload["full"]["configuration"]

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

    minimum = float(store.minimum[0])
    scale = float(store.scale[0])
    power = float(prior_config["temperature_power"])
    method = str(prior_config["method"])
    degree = int(prior_config["condition_degree"])
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

    scenarios = np.asarray([int(row[0]) for row in heldout], dtype=np.int64)
    starts = np.asarray([int(row[1]) for row in heldout], dtype=np.int64)
    predictions = []
    targets = []
    for scenario, start, span in heldout:
        times = target_indices(start, span, 32, args.history_frames)
        encoded = predict_encoded_trajectory(
            int(scenario), times, encoded_bank, frame_times, method, degree
        )
        physical = decode_temperature(encoded, minimum, power)
        predictions.append(
            np.clip((physical - minimum) / scale, 0.0, 1.0).astype(np.float32)
        )
        targets.append(
            store.normalize(store.data[scenario, times]).astype(np.float32)[:, 0]
        )
    target = np.stack(targets).astype(np.float32)
    with torch.no_grad():
        prediction = smooth(
            torch.from_numpy(np.stack(predictions).astype(np.float32)),
            float(postprocess["spatial_sigma"]),
            float(postprocess["temporal_blend"]),
        ).numpy()
    prediction = np.power(
        np.clip(prediction * float(postprocess["scale"]), 0.0, 1.0),
        float(postprocess["gamma"]),
    ).astype(np.float32)

    fractions = {
        fraction: field_metrics(
            prediction[:, index : index + 1], target[:, index : index + 1]
        )
        for fraction, index in zip(
            ("0.25", "0.5", "0.75", "1.0"), (8, 16, 23, 31)
        )
    }
    per_scenario = {
        str(int(scenario)): field_metrics(
            prediction[scenarios == scenario], target[scenarios == scenario]
        )
        for scenario in sorted(set(scenarios.tolist()))
    }

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
        "model": "FireWave strict temperature condition prior frozen evaluation",
        "prior_selection_source": args.prior_selection_json,
        "postprocess_selection_source": args.postprocess_selection_json,
        "prior_configuration": {
            "method": method,
            "temperature_power": power,
            "condition_degree": degree,
        },
        "postprocessing": postprocess,
        "full_metrics": field_metrics(prediction, target),
        "fraction_metrics": fractions,
        "per_scenario": per_scenario,
        "data_audit": {
            "protocol": "strict frame-disjoint dense variable-span windows",
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
