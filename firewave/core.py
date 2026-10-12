#!/usr/bin/env python3
"""Train and evaluate the FireWave FIT tunnel-fire model.

The original 2-D smoke code is tightly coupled to a 32x64x64, six-channel
control benchmark.  This module keeps the two defining ingredients of the
FireWave baseline --
trajectory generation in a wavelet domain and a spatio-temporal 3-D U-Net
denoiser -- while exposing a data path for FIT's [T, 2, 23, 480] fields.

No external wavelet package is needed.  A one-level orthonormal Haar transform
is used over time and both spatial axes.  The odd 23-pixel height is padded by
one replicated boundary row before the transform and cropped after inversion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCENARIOS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4")
HRR_MW = {"A1": 5.0, "A2": 10.0, "A3": 20.0, "A4": 50.0,
          "B1": 5.0, "B2": 10.0, "B3": 20.0, "B4": 50.0}
VENT_MS = {"A1": 2.0, "A2": 2.0, "A3": 2.0, "A4": 2.0,
           "B1": 4.0, "B2": 4.0, "B3": 4.0, "B4": 4.0}
FIELD_NAMES = ("temperature", "visibility")


@dataclass
class Config:
    data_root: str
    output_dir: str
    seed: int = 1
    trajectory_frames: int = 32
    train_steps: int = 20000
    batch_size: int = 4
    learning_rate: float = 2e-4
    diffusion_steps: int = 1000
    ddim_steps: int = 100
    model_dim: int = 32
    ema_decay: float = 0.999
    eval_samples: int = 128
    split_seed: int = 1
    num_workers: int = 0
    log_every: int = 50
    validate_every: int = 500
    save_every: int = 1000
    predict_residual: bool = False
    history_frames: int = 1
    anisotropic: bool = False
    include_full_span: bool = False
    full_span_repeats: int = 16
    wavelet_type: str = "haar"
    aux_lowfreq_weight: float = 0.0
    aux_gradient_weight: float = 0.0
    aux_ssim_weight: float = 0.0
    aux_mse_weight: float = 0.0
    aux_early_mse_weight: float = 0.0
    aux_early_frames: int = 8
    conditional_mean: bool = False
    mean_only_training: bool = False
    mean_loss_weight: float = 1.0
    mean_log_mse_weight: float = 0.0
    mean_ssim_weight: float = 0.0
    mean_gradient_weight: float = 0.0
    mean_temporal_gradient_weight: float = 0.0
    mean_horizon_weight_end: float = 1.0
    mean_learning_rate: float = 1e-4
    aux_max_timestep: int = -1
    aux_gate_rescale: bool = True
    visibility_noise_weight: float = 1.0
    visibility_aux_weight: float = 1.0
    reset_optimizer_on_resume: bool = False
    prediction_objective: str = "epsilon"
    coordinate_conditioning: bool = False
    allow_partial_resume: bool = False
    select_by_ssim: bool = False
    selection_every: int = 5000
    selection_samples: int = 4
    selection_ddim_steps: int = 20
    selection_field: str = "mean"
    selection_start_timestep: int = -1
    selection_ensemble_size: int = 1
    selection_antithetic: bool = False
    selection_visibility_spatial_sigma: float = 0.0
    selection_visibility_temporal_blend: float = 0.0
    selection_visibility_median_blend: float = 0.0
    selection_psnr_weight: float = 0.0
    selection_initialization: str = "random"
    table2_span: int = 31
    train_short_only: bool = False
    train_start_stride: int = 32
    strict_frame_disjoint_train: bool = False
    strict_protected_min_span: int = 0
    strict_dense_variable_train: bool = False
    strict_train_min_span: int = 0
    strict_balance_scenario_span: bool = False
    selection_min_span: int = 0
    scenario_ids: tuple[int, ...] | None = None
    dropout: float = 0.0
    global_conditioning: bool = False
    spatial_wavelet_only: bool = False
    temporal_fourier_features: int = 0
    scenario_onehot_conditioning: bool = False
    pixel_refiner: bool = False
    temporal_query_refiner: bool = False
    visibility_only_model: bool = False
    refiner_only_training: bool = False
    refiner_dim: int = 32
    refiner_learning_rate: float = 1e-4
    weight_decay: float = 1e-4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FITStore:
    """Load the eight released FIT trajectories into CPU memory as float32."""

    def __init__(self, root: str):
        self.root = Path(root)
        fields: List[np.ndarray] = []
        hashes: Dict[str, Dict[str, str]] = {}
        for scenario in SCENARIOS:
            temp_path = self.root / scenario / f"{scenario}_temp.npy"
            vis_path = self.root / scenario / f"{scenario}_visibi.npy"
            if not temp_path.is_file() or not vis_path.is_file():
                raise FileNotFoundError(f"Missing FIT arrays for {scenario} under {self.root}")
            temp = np.load(temp_path, mmap_mode="r")
            vis = np.load(vis_path, mmap_mode="r")
            if temp.shape != (1001, 23, 480) or vis.shape != temp.shape:
                raise ValueError(
                    f"Unexpected shape for {scenario}: temp={temp.shape}, visibility={vis.shape}"
                )
            fields.append(np.stack((np.asarray(temp, dtype=np.float32),
                                    np.asarray(vis, dtype=np.float32)), axis=1))
            hashes[scenario] = {
                "temperature": sha256(temp_path),
                "visibility": sha256(vis_path),
            }
        self.data = np.stack(fields, axis=0)  # [scenario, time, channel, H, W]
        self.hashes = hashes
        self.minimum = self.data.min(axis=(0, 1, 3, 4)).astype(np.float32)
        self.maximum = self.data.max(axis=(0, 1, 3, 4)).astype(np.float32)
        self.scale = np.maximum(self.maximum - self.minimum, 1e-6)

    def normalize(self, array: np.ndarray) -> np.ndarray:
        return (array - self.minimum[None, :, None, None]) / self.scale[None, :, None, None]

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        return array * self.scale[None, :, None, None] + self.minimum[None, :, None, None]


def make_segment_index(
    n_time: int = 1001,
    history_frames: int = 1,
    include_full_span: bool = False,
    full_span_repeats: int = 16,
) -> List[Tuple[int, int, int]]:
    """Create variable-horizon future trajectories after a contiguous history.

    The released data contain 1001 samples across 400 seconds.  Each item is a
    history followed by a 32-frame trajectory sampled uniformly across ``span``.
    When requested, the longest 400-second-compatible span is oversampled so it
    is represented during optimization rather than appearing once among many
    short-horizon windows.
    """

    spans = (31, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896)
    max_span = n_time - 1 - history_frames
    spans = tuple(span for span in spans if span <= max_span)
    samples: List[Tuple[int, int, int]] = []
    for scenario_id in range(len(SCENARIOS)):
        for span in spans:
            max_start = n_time - 1 - history_frames - span
            for start in range(0, max_start + 1, 32):
                samples.append((scenario_id, start, span))
        if include_full_span:
            samples.extend((scenario_id, 0, max_span) for _ in range(full_span_repeats))
    return samples


def split_segment_index(
    samples: Sequence[Tuple[int, int, int]], seed: int
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    order = np.random.default_rng(seed).permutation(len(samples))
    n_train = int(0.8 * len(order))
    n_val = int(0.1 * len(order))
    train = [samples[i] for i in order[:n_train]]
    val = [samples[i] for i in order[n_train:n_train + n_val]]
    test = [samples[i] for i in order[n_train + n_val:]]
    return train, val, test


def sample_frame_keys(
    samples: Sequence[Tuple[int, int, int]],
    history_frames: int,
    trajectory_frames: int,
) -> set[Tuple[int, int]]:
    """Return every history and target frame referenced by ``samples``."""

    keys: set[Tuple[int, int]] = set()
    for scenario, start, span in samples:
        keys.update(
            (int(scenario), int(index))
            for index in range(int(start), int(start) + history_frames)
        )
        forecast_start = int(start) + history_frames
        target_indices = np.rint(
            np.linspace(
                forecast_start,
                forecast_start + int(span),
                trajectory_frames,
            )
        ).astype(np.int64)
        keys.update(
            (int(scenario), int(index)) for index in target_indices
        )
    return keys


def selected_scenario_ids(config: Config) -> tuple[int, ...]:
    if config.scenario_ids is None:
        return tuple(range(len(SCENARIOS)))
    scenario_ids = tuple(int(value) for value in config.scenario_ids)
    if not scenario_ids:
        raise ValueError("scenario_ids must contain at least one scenario")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario_ids must not contain duplicates")
    if any(value < 0 or value >= len(SCENARIOS) for value in scenario_ids):
        raise ValueError(f"scenario_ids must be in [0, {len(SCENARIOS) - 1}]")
    return scenario_ids


class FITTrajectoryDataset(Dataset):
    def __init__(
        self,
        store: FITStore,
        samples: Sequence[Tuple[int, int, int]],
        frames: int = 32,
        history_frames: int = 1,
    ):
        self.store = store
        self.samples = list(samples)
        self.frames = frames
        self.history_frames = history_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        scenario_id, start, span = self.samples[index]
        history_indices = np.arange(start, start + self.history_frames, dtype=np.int64)
        forecast_start = start + self.history_frames
        target_indices = np.rint(np.linspace(forecast_start, forecast_start + span, self.frames)).astype(np.int64)
        history = self.store.normalize(self.store.data[scenario_id, history_indices])
        target = self.store.normalize(self.store.data[scenario_id, target_indices])
        # Copy avoids non-writable memmap warnings.
        history = torch.from_numpy(np.array(history, dtype=np.float32, copy=True))
        target = torch.from_numpy(np.array(target, dtype=np.float32, copy=True))
        scenario = SCENARIOS[scenario_id]
        conditions = torch.tensor(
            [
                math.log(HRR_MW[scenario] / 5.0) / math.log(10.0),
                (VENT_MS[scenario] - 2.0) / 2.0,
                start / 1000.0,
                span / 1000.0,
            ],
            dtype=torch.float32,
        )
        return target, history, conditions, scenario_id, start, span


def _haar_split(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    even = x.index_select(dim, torch.arange(0, x.shape[dim], 2, device=x.device))
    odd = x.index_select(dim, torch.arange(1, x.shape[dim], 2, device=x.device))
    scale = math.sqrt(2.0)
    return (even + odd) / scale, (even - odd) / scale


def _haar_merge(low: torch.Tensor, high: torch.Tensor, dim: int) -> torch.Tensor:
    scale = math.sqrt(2.0)
    even = (low + high) / scale
    odd = (low - high) / scale
    return torch.stack((even, odd), dim=dim + 1).flatten(dim, dim + 1)


def pad_fit_field(x: torch.Tensor) -> torch.Tensor:
    """Pad [B,C,T,H,W] to even dimensions; FIT only needs H: 23 -> 24."""

    pad_t = x.shape[-3] % 2
    pad_h = x.shape[-2] % 2
    pad_w = x.shape[-1] % 2
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
    return x


def haar3d(x: torch.Tensor) -> torch.Tensor:
    """One-level orthonormal 3-D Haar DWT, flattened to C*8 channels."""

    x = pad_fit_field(x)
    t_low, t_high = _haar_split(x, 2)
    bands: List[torch.Tensor] = []
    for t_band in (t_low, t_high):
        h_low, h_high = _haar_split(t_band, 3)
        for h_band in (h_low, h_high):
            w_low, w_high = _haar_split(h_band, 4)
            bands.extend((w_low, w_high))
    # Keep each physical field's eight subbands adjacent.
    stacked = torch.stack(bands, dim=2)  # [B,C,8,T/2,H/2,W/2]
    return stacked.flatten(1, 2)


def ihaar3d(coeff: torch.Tensor, channels: int, output_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Inverse of :func:`haar3d`, cropped to ``(T,H,W)``."""

    b, c8, t, h, w = coeff.shape
    if c8 != channels * 8:
        raise ValueError(f"Expected {channels * 8} channels, received {c8}")
    bands = coeff.reshape(b, channels, 8, t, h, w)
    time_bands: List[torch.Tensor] = []
    for t_bit in range(2):
        base = t_bit * 4
        h_low = _haar_merge(bands[:, :, base], bands[:, :, base + 1], 4)
        h_high = _haar_merge(bands[:, :, base + 2], bands[:, :, base + 3], 4)
        time_bands.append(_haar_merge(h_low, h_high, 3))
    x = _haar_merge(time_bands[0], time_bands[1], 2)
    target_t, target_h, target_w = output_shape
    return x[:, :, :target_t, :target_h, :target_w]


def haar2d(x: torch.Tensor) -> torch.Tensor:
    """One-level orthonormal 2-D Haar DWT for initial-condition fields."""

    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
    h_low, h_high = _haar_split(x, 2)
    bands: List[torch.Tensor] = []
    for h_band in (h_low, h_high):
        w_low, w_high = _haar_split(h_band, 3)
        bands.extend((w_low, w_high))
    return torch.stack(bands, dim=2).flatten(1, 2)


def ihaar2d(
    coeff: torch.Tensor, channels: int, output_shape: Tuple[int, int]
) -> torch.Tensor:
    """Inverse of :func:`haar2d`, cropped to ``(H,W)``."""

    b, c4, _, _ = coeff.shape
    if c4 != channels * 4:
        raise ValueError(f"Expected {channels * 4} channels, received {c4}")
    bands = coeff.reshape(b, channels, 4, coeff.shape[-2], coeff.shape[-1])
    h_low = _haar_merge(bands[:, :, 0], bands[:, :, 1], 3)
    h_high = _haar_merge(bands[:, :, 2], bands[:, :, 3], 3)
    x = _haar_merge(h_low, h_high, 2)
    target_h, target_w = output_shape
    return x[:, :, :target_h, :target_w]


def spatial_haar3d(x: torch.Tensor) -> torch.Tensor:
    """Apply a 2-D Haar transform independently at every time step."""

    b, c, t, h, w = x.shape
    frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    coeff = haar2d(frames)
    return coeff.reshape(b, t, c * 4, *coeff.shape[-2:]).permute(0, 2, 1, 3, 4)


def inverse_spatial_haar3d(
    coeff: torch.Tensor, channels: int, output_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """Invert :func:`spatial_haar3d` without reducing temporal resolution."""

    b, c4, t, h, w = coeff.shape
    frames = coeff.permute(0, 2, 1, 3, 4).reshape(b * t, c4, h, w)
    reconstructed = ihaar2d(frames, channels, output_shape[-2:])
    return reconstructed.reshape(b, t, channels, *output_shape[-2:]).permute(0, 2, 1, 3, 4)


def pixel_unshuffle3d(x: torch.Tensor) -> torch.Tensor:
    """Information-preserving 2x2x2 space-to-depth without wavelet filtering."""

    x = pad_fit_field(x)
    b, c, t, h, w = x.shape
    x = x.reshape(b, c, t // 2, 2, h // 2, 2, w // 2, 2)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(
        b, c * 8, t // 2, h // 2, w // 2
    )


def pixel_shuffle3d(
    coeff: torch.Tensor, channels: int, output_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """Inverse of :func:`pixel_unshuffle3d`, cropped to the requested shape."""

    b, c8, t, h, w = coeff.shape
    if c8 != channels * 8:
        raise ValueError(f"Expected {channels * 8} channels, received {c8}")
    x = coeff.reshape(b, channels, 2, 2, 2, t, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
        b, channels, t * 2, h * 2, w * 2
    )
    target_t, target_h, target_w = output_shape
    return x[:, :, :target_t, :target_h, :target_w]


def pixel_unshuffle2d(x: torch.Tensor) -> torch.Tensor:
    """Information-preserving 2x2 space-to-depth for history fields."""

    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // 2, 2, w // 2, 2)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


BIOR3_DETAIL_KEYS = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
BIOR13_FILTERS = {
    "dec_lo": (-0.08838834764831845, 0.08838834764831845, 0.7071067811865476,
               0.7071067811865476, 0.08838834764831845, -0.08838834764831845),
    "dec_hi": (-0.0, 0.0, -0.7071067811865476, 0.7071067811865476, -0.0, 0.0),
    "rec_lo": (0.0, 0.0, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
    "rec_hi": (-0.08838834764831845, -0.08838834764831845, 0.7071067811865476,
               -0.7071067811865476, 0.08838834764831845, 0.08838834764831845),
}
_BIOR_FILTER_CACHE: Dict[Tuple[str, torch.dtype, str], torch.Tensor] = {}


def _bior_filter_bank(device: torch.device, dtype: torch.dtype, dimension: str, reconstruction: bool) -> torch.Tensor:
    key = (str(device), dtype, f"{dimension}-{'rec' if reconstruction else 'dec'}")
    cached = _BIOR_FILTER_CACHE.get(key)
    if cached is not None:
        return cached
    prefix = "rec" if reconstruction else "dec"
    low = torch.tensor(BIOR13_FILTERS[f"{prefix}_lo"], device=device, dtype=dtype)
    high = torch.tensor(BIOR13_FILTERS[f"{prefix}_hi"], device=device, dtype=dtype)
    if not reconstruction:
        low = low.flip(0)
        high = high.flip(0)
    if dimension == "2d":
        filters = torch.stack((
            low[:, None] * low[None, :],
            high[:, None] * low[None, :],
            low[:, None] * high[None, :],
            high[:, None] * high[None, :],
        ), dim=0).unsqueeze(1)
    elif dimension == "3d":
        filters = torch.stack(tuple(
            first[:, None, None] * second[None, :, None] * third[None, None, :]
            for first in (low, high)
            for second in (low, high)
            for third in (low, high)
        ), dim=0).unsqueeze(1)
    else:
        raise ValueError(dimension)
    _BIOR_FILTER_CACHE[key] = filters
    return filters


def _bior_padding(length: int) -> Tuple[int, int]:
    # PTWT zero-mode padding for a six-tap bior1.3 filter.
    return 4, 4 + (length % 2)


def _pad_coeff_grid(coeff: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    raw_shape = tuple(int(value) for value in coeff.shape[-3:])
    pad_h = (-raw_shape[-2]) % 4
    pad_w = (-raw_shape[-1]) % 4
    if pad_h or pad_w:
        coeff = F.pad(coeff, (0, pad_w, 0, pad_h, 0, 0), mode="constant", value=0.0)
    return coeff, raw_shape


def wavelet3d(x: torch.Tensor, wavelet_type: str) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    if wavelet_type == "haar":
        coeff = haar3d(x)
        return coeff, tuple(int(value) for value in coeff.shape[-3:])
    if wavelet_type == "pixel-unshuffle":
        coeff = pixel_unshuffle3d(x)
        return coeff, tuple(int(value) for value in coeff.shape[-3:])
    if wavelet_type != "bior1.3":
        raise ValueError(f"Unsupported wavelet: {wavelet_type}")
    b, c, t, h, w = x.shape
    left_w, right_w = _bior_padding(w)
    left_h, right_h = _bior_padding(h)
    left_t, right_t = _bior_padding(t)
    padded = F.pad(
        x.reshape(b * c, 1, t, h, w),
        (left_w, right_w, left_h, right_h, left_t, right_t),
        mode="constant", value=0.0,
    )
    filters = _bior_filter_bank(x.device, x.dtype, "3d", reconstruction=False)
    bands = F.conv3d(padded, filters, stride=2)
    coeff = bands.reshape(b, c * 8, *bands.shape[-3:])
    return _pad_coeff_grid(coeff)


def inverse_wavelet3d(
    coeff: torch.Tensor,
    channels: int,
    output_shape: Tuple[int, int, int],
    wavelet_type: str,
) -> torch.Tensor:
    if wavelet_type == "haar":
        return ihaar3d(coeff, channels, output_shape)
    if wavelet_type == "pixel-unshuffle":
        return pixel_shuffle3d(coeff, channels, output_shape)
    if wavelet_type != "bior1.3":
        raise ValueError(f"Unsupported wavelet: {wavelet_type}")
    b, c8, _, _, _ = coeff.shape
    if c8 != channels * 8:
        raise ValueError(f"Expected {channels * 8} channels, received {c8}")
    raw_t = (output_shape[0] + 5) // 2
    raw_h = (output_shape[1] + 5) // 2
    raw_w = (output_shape[2] + 5) // 2
    bands = coeff[:, :, :raw_t, :raw_h, :raw_w].reshape(b * channels, 8, raw_t, raw_h, raw_w)
    filters = _bior_filter_bank(coeff.device, coeff.dtype, "3d", reconstruction=True)
    reconstructed = F.conv_transpose3d(bands, filters, stride=2).squeeze(1)
    reconstructed = reconstructed[..., 4:-4, 4:-4, 4:-4]
    target_t, target_h, target_w = output_shape
    return reconstructed[:, :target_t, :target_h, :target_w].reshape(
        b, channels, target_t, target_h, target_w
    )


def wavelet2d(x: torch.Tensor, wavelet_type: str) -> torch.Tensor:
    if wavelet_type == "haar":
        return haar2d(x)
    if wavelet_type == "pixel-unshuffle":
        return pixel_unshuffle2d(x)
    if wavelet_type != "bior1.3":
        raise ValueError(f"Unsupported wavelet: {wavelet_type}")
    b, c, h, w = x.shape
    left_w, right_w = _bior_padding(w)
    left_h, right_h = _bior_padding(h)
    padded = F.pad(
        x.reshape(b * c, 1, h, w),
        (left_w, right_w, left_h, right_h), mode="constant", value=0.0,
    )
    filters = _bior_filter_bank(x.device, x.dtype, "2d", reconstruction=False)
    bands = F.conv2d(padded, filters, stride=2)
    coeff = bands.reshape(b, c * 4, *bands.shape[-2:])
    pad_h = (-coeff.shape[-2]) % 4
    pad_w = (-coeff.shape[-1]) % 4
    if pad_h or pad_w:
        coeff = F.pad(coeff, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return coeff


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time_index: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=time_index.device) / max(half - 1, 1)
        )
        phase = time_index.float()[:, None] * frequency[None]
        return torch.cat((phase.sin(), phase.cos()), dim=1)


class ResBlock3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        groups = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(time_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        if t == 1:
            # Single-token attention has unit softmax weight.  This exact
            # shortcut also avoids CUDA SDPA stride-alignment restrictions
            # for the autoregressive one-step branch.
            return x + self.proj(v)
        head_dim = c // self.heads
        def reshape(z: torch.Tensor) -> torch.Tensor:
            return (
                z.permute(0, 3, 4, 1, 2)
                .reshape(b * h * w, self.heads, head_dim, t)
                .transpose(-1, -2)
                .contiguous()
            )
        q, k, v = reshape(q), reshape(k), reshape(v)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(-1, -2).reshape(b, h, w, c, t).permute(0, 3, 4, 1, 2)
        return x + self.proj(attended)


class SpatialLinearAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        head_dim = c // self.heads
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        def reshape(z: torch.Tensor) -> torch.Tensor:
            return z.permute(0, 2, 1, 3, 4).reshape(b * t, self.heads, head_dim, h * w)
        q, k, v = reshape(q), reshape(k), reshape(v)
        q = q.softmax(dim=-2) * (head_dim ** -0.5)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        attended = torch.einsum("bhde,bhdn->bhen", context, q)
        attended = attended.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + self.proj(attended)


class AttentionPair(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = SpatialLinearAttention(channels)
        self.temporal = TemporalAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.spatial(x))


class RefinerResBlock2D(nn.Module):
    """Compact residual block for the full-resolution correction pathway."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(x)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return residual + self.skip(x)


class PixelResidualRefiner(nn.Module):
    """Predict a zero-initialized pixel-space correction to the wavelet mean.

    Time is represented by ordered output channels, while the input contains
    all observed frames, physical conditions, a scenario code, and spatial
    coordinates.  This keeps the wavelet prediction as the exact initialization
    and lets the new branch focus on high-resolution errors that survive inverse
    wavelet reconstruction.
    """

    def __init__(
        self,
        history_frames: int,
        trajectory_frames: int,
        dim: int = 32,
    ):
        super().__init__()
        if history_frames < 1 or trajectory_frames < 1 or dim < 4:
            raise ValueError("Invalid pixel-refiner dimensions")
        self.history_frames = history_frames
        self.trajectory_frames = trajectory_frames
        input_channels = history_frames * 2 + trajectory_frames * 2 + 4 + 8 + 2
        self.input = nn.Conv2d(input_channels, dim, 3, padding=1)
        self.enc1 = nn.Sequential(
            RefinerResBlock2D(dim, dim),
            RefinerResBlock2D(dim, dim),
        )
        self.down1 = nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            RefinerResBlock2D(dim * 2, dim * 2),
            RefinerResBlock2D(dim * 2, dim * 2),
        )
        self.down2 = nn.Conv2d(
            dim * 2, dim * 4, (3, 4), stride=(1, 2), padding=(1, 1)
        )
        self.mid = nn.Sequential(
            RefinerResBlock2D(dim * 4, dim * 4),
            RefinerResBlock2D(dim * 4, dim * 4),
            nn.Conv2d(dim * 4, dim * 4, (3, 9), padding=(1, 4)),
            nn.SiLU(),
        )
        self.up2 = nn.ConvTranspose2d(
            dim * 4, dim * 2, (3, 4), stride=(1, 2), padding=(1, 1)
        )
        self.dec2 = RefinerResBlock2D(dim * 4, dim * 2)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 4, stride=2, padding=1)
        self.dec1 = RefinerResBlock2D(dim * 2, dim)
        self.output = nn.Conv2d(dim, trajectory_frames * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        history_01: torch.Tensor,
        scalar_condition_01: torch.Tensor,
        base_prediction_01: torch.Tensor,
    ) -> torch.Tensor:
        batch, history_frames, channels, height, width = history_01.shape
        if history_frames != self.history_frames or channels != 2:
            raise ValueError(
                "Pixel refiner received an incompatible history tensor: "
                f"{tuple(history_01.shape)}"
            )
        if tuple(base_prediction_01.shape[1:3]) != (self.trajectory_frames, 2):
            raise ValueError(
                "Pixel refiner received an incompatible base prediction: "
                f"{tuple(base_prediction_01.shape)}"
            )
        history = history_01.reshape(batch, history_frames * channels, height, width)
        base_prediction = base_prediction_01.reshape(
            batch, self.trajectory_frames * 2, height, width
        )
        scalars = scalar_condition_01[:, :, None, None].expand(-1, -1, height, width)
        hrr_levels = torch.tensor(
            (0.0, math.log(2.0) / math.log(10.0), math.log(4.0) / math.log(10.0), 1.0),
            device=history.device,
            dtype=history.dtype,
        )
        hrr_index = (scalar_condition_01[:, :1] - hrr_levels[None]).abs().argmin(dim=1)
        ventilation_index = (scalar_condition_01[:, 1] >= 0.5).long()
        scenario_index = ventilation_index * 4 + hrr_index
        scenario = F.one_hot(scenario_index, num_classes=8).to(history.dtype)
        scenario = scenario[:, :, None, None].expand(-1, -1, height, width)
        height_grid = torch.linspace(
            -1.0, 1.0, height, device=history.device, dtype=history.dtype
        )[None, None, :, None].expand(batch, -1, -1, width)
        width_grid = torch.linspace(
            -1.0, 1.0, width, device=history.device, dtype=history.dtype
        )[None, None, None, :].expand(batch, -1, height, -1)
        x = torch.cat(
            (history, base_prediction, scalars, scenario, height_grid, width_grid), dim=1
        )
        pad_height = x.shape[-2] % 2
        pad_width = x.shape[-1] % 4
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        h1 = self.enc1(self.input(x))
        h2 = self.enc2(self.down1(h1))
        x = self.mid(self.down2(h2))
        x = self.dec2(torch.cat((self.up2(x), h2), dim=1))
        x = self.dec1(torch.cat((self.up1(x), h1), dim=1))
        correction = self.output(x)[..., :height, :width]
        return correction.reshape(
            batch, self.trajectory_frames, 2, height, width
        )


class TemporalQueryResidualRefiner(nn.Module):
    """Shared full-resolution correction network queried at each future time."""

    def __init__(self, history_frames: int, trajectory_frames: int, dim: int = 16):
        super().__init__()
        if history_frames < 1 or dim < 4:
            raise ValueError("Invalid temporal-query refiner dimensions")
        self.history_frames = history_frames
        self.trajectory_frames = trajectory_frames
        # History, base frame, four physical scalars, scenario one-hot,
        # coordinates, relative/absolute time, and two Fourier pairs.
        input_channels = history_frames * 2 + 2 + 4 + 8 + 2 + 6
        self.input = nn.Conv2d(input_channels, dim, 3, padding=1)
        self.enc1 = nn.Sequential(
            RefinerResBlock2D(dim, dim),
            RefinerResBlock2D(dim, dim),
        )
        self.down1 = nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            RefinerResBlock2D(dim * 2, dim * 2),
            RefinerResBlock2D(dim * 2, dim * 2),
        )
        self.down2 = nn.Conv2d(
            dim * 2, dim * 4, (3, 4), stride=(1, 2), padding=(1, 1)
        )
        self.mid = nn.Sequential(
            RefinerResBlock2D(dim * 4, dim * 4),
            RefinerResBlock2D(dim * 4, dim * 4),
            nn.Conv2d(dim * 4, dim * 4, (3, 9), padding=(1, 4)),
            nn.SiLU(),
        )
        self.up2 = nn.ConvTranspose2d(
            dim * 4, dim * 2, (3, 4), stride=(1, 2), padding=(1, 1)
        )
        self.dec2 = RefinerResBlock2D(dim * 4, dim * 2)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 4, stride=2, padding=1)
        self.dec1 = RefinerResBlock2D(dim * 2, dim)
        self.output = nn.Conv2d(dim, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        history_01: torch.Tensor,
        scalar_condition_01: torch.Tensor,
        base_prediction_01: torch.Tensor,
    ) -> torch.Tensor:
        batch, history_frames, channels, height, width = history_01.shape
        frames = base_prediction_01.shape[1]
        if history_frames != self.history_frames or channels != 2:
            raise ValueError(
                "Temporal-query refiner received an incompatible history tensor"
            )
        history = history_01.reshape(
            batch, history_frames * channels, height, width
        )[:, None].expand(-1, frames, -1, -1, -1)
        scalars = scalar_condition_01[:, None, :, None, None].expand(
            -1, frames, -1, height, width
        )
        hrr_levels = torch.tensor(
            (0.0, math.log(2.0) / math.log(10.0), math.log(4.0) / math.log(10.0), 1.0),
            device=history_01.device,
            dtype=history_01.dtype,
        )
        hrr_index = (scalar_condition_01[:, :1] - hrr_levels[None]).abs().argmin(dim=1)
        ventilation_index = (scalar_condition_01[:, 1] >= 0.5).long()
        scenario_index = ventilation_index * 4 + hrr_index
        scenario = F.one_hot(scenario_index, num_classes=8).to(history_01.dtype)
        scenario = scenario[:, None, :, None, None].expand(
            -1, frames, -1, height, width
        )
        height_grid = torch.linspace(
            -1.0, 1.0, height, device=history_01.device, dtype=history_01.dtype
        )[None, None, None, :, None].expand(batch, frames, -1, -1, width)
        width_grid = torch.linspace(
            -1.0, 1.0, width, device=history_01.device, dtype=history_01.dtype
        )[None, None, None, None, :].expand(batch, frames, -1, height, -1)
        relative_time = torch.linspace(
            0.0, 1.0, frames, device=history_01.device, dtype=history_01.dtype
        )[None, :, None, None, None].expand(batch, -1, -1, height, width)
        absolute_time = (
            scalar_condition_01[:, 2:3, None, None, None]
            + history_frames / 1000.0
            + scalar_condition_01[:, 3:4, None, None, None] * relative_time
        )
        temporal = [relative_time * 2.0 - 1.0, absolute_time * 2.0 - 1.0]
        for frequency in (1.0, 2.0):
            phase = 2.0 * math.pi * frequency * absolute_time
            temporal.extend((phase.sin(), phase.cos()))
        time_features = torch.cat(temporal, dim=2)
        x = torch.cat(
            (
                history,
                base_prediction_01,
                scalars,
                scenario,
                height_grid,
                width_grid,
                time_features,
            ),
            dim=2,
        ).reshape(batch * frames, -1, height, width)
        pad_height = x.shape[-2] % 2
        pad_width = x.shape[-1] % 4
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        h1 = self.enc1(self.input(x))
        h2 = self.enc2(self.down1(h1))
        x = self.mid(self.down2(h2))
        x = self.dec2(torch.cat((self.up2(x), h2), dim=1))
        x = self.dec1(torch.cat((self.up1(x), h1), dim=1))
        return self.output(x)[..., :height, :width].reshape(
            batch, frames, 2, height, width
        )


class FITWaveletUNet(nn.Module):
    """Spatially downsampling, temporally attentive 3-D U-Net denoiser."""

    def __init__(
        self,
        target_channels: int = 16,
        condition_channels: int = 12,
        dim: int = 32,
        anisotropic: bool = False,
        dropout: float = 0.0,
        global_conditioning: bool = False,
        coordinate_conditioning: bool = False,
        scalar_condition_start: int | None = None,
    ):
        super().__init__()
        self.target_channels = target_channels
        self.condition_channels = condition_channels
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(dim), nn.Linear(dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.global_conditioning = global_conditioning
        if global_conditioning:
            scalar_start = (
                scalar_condition_start
                if scalar_condition_start is not None
                else condition_channels - (7 if coordinate_conditioning else 4)
            )
            if scalar_start < 0:
                raise ValueError("condition tensor does not contain four scalar channels")
            self.scalar_condition_slice = slice(scalar_start, scalar_start + 4)
            self.condition_mlp = nn.Sequential(
                nn.Linear(4, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
            )
            nn.init.zeros_(self.condition_mlp[-1].weight)
            nn.init.zeros_(self.condition_mlp[-1].bias)
        else:
            self.scalar_condition_slice = None
            self.condition_mlp = None
        self.input = nn.Conv3d(target_channels + condition_channels, dim, 3, padding=1)

        self.enc1a = ResBlock3D(dim, dim, time_dim, dropout)
        self.enc1b = ResBlock3D(dim, dim, time_dim, dropout)
        self.attn1 = AttentionPair(dim)
        self.down1 = nn.Conv3d(dim, dim * 2, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.enc2a = ResBlock3D(dim * 2, dim * 2, time_dim, dropout)
        self.enc2b = ResBlock3D(dim * 2, dim * 2, time_dim, dropout)
        self.attn2 = AttentionPair(dim * 2)
        second_kernel = (1, 3, 4) if anisotropic else (1, 4, 4)
        second_stride = (1, 1, 2) if anisotropic else (1, 2, 2)
        second_padding = (0, 1, 1)
        self.down2 = nn.Conv3d(
            dim * 2, dim * 4, second_kernel, stride=second_stride, padding=second_padding
        )

        self.mid1 = ResBlock3D(dim * 4, dim * 4, time_dim, dropout)
        self.mid_attn = AttentionPair(dim * 4)
        self.mid2 = ResBlock3D(dim * 4, dim * 4, time_dim, dropout)

        self.up2 = nn.ConvTranspose3d(
            dim * 4, dim * 2, second_kernel, stride=second_stride, padding=second_padding
        )
        self.dec2a = ResBlock3D(dim * 4, dim * 2, time_dim, dropout)
        self.dec2b = ResBlock3D(dim * 2, dim * 2, time_dim, dropout)
        self.dec_attn2 = AttentionPair(dim * 2)

        self.up1 = nn.ConvTranspose3d(dim * 2, dim, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dec1a = ResBlock3D(dim * 2, dim, time_dim, dropout)
        self.dec1b = ResBlock3D(dim, dim, time_dim, dropout)
        self.dec_attn1 = AttentionPair(dim)
        self.output = nn.Sequential(nn.GroupNorm(min(8, dim), dim), nn.SiLU(), nn.Conv3d(dim, target_channels, 1))

    def forward(self, noisy: torch.Tensor, condition: torch.Tensor, diffusion_time: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(diffusion_time)
        if self.condition_mlp is not None:
            scalar_condition = condition[:, self.scalar_condition_slice].mean(dim=(2, 3, 4))
            time_emb = time_emb + self.condition_mlp(scalar_condition)
        x = self.input(torch.cat((noisy, condition), dim=1))
        h1 = self.attn1(self.enc1b(self.enc1a(x, time_emb), time_emb))
        x = self.down1(h1)
        h2 = self.attn2(self.enc2b(self.enc2a(x, time_emb), time_emb))
        x = self.down2(h2)
        x = self.mid2(self.mid_attn(self.mid1(x, time_emb)), time_emb)
        x = self.up2(x)
        x = self.dec_attn2(self.dec2b(self.dec2a(torch.cat((x, h2), dim=1), time_emb), time_emb))
        x = self.up1(x)
        x = self.dec_attn1(self.dec1b(self.dec1a(torch.cat((x, h1), dim=1), time_emb), time_emb))
        return self.output(x)


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5).square()
    cumulative = cumulative / cumulative[0]
    return (1.0 - cumulative[1:] / cumulative[:-1]).clamp(0.0, 0.999).float()


class FireWaveModel(nn.Module):
    def __init__(
        self,
        model: FITWaveletUNet,
        diffusion_steps: int = 1000,
        predict_residual: bool = False,
        wavelet_type: str = "haar",
        aux_lowfreq_weight: float = 0.0,
        aux_gradient_weight: float = 0.0,
        aux_ssim_weight: float = 0.0,
        aux_max_timestep: int = -1,
        aux_gate_rescale: bool = True,
        visibility_noise_weight: float = 1.0,
        visibility_aux_weight: float = 1.0,
        prediction_objective: str = "epsilon",
        coordinate_conditioning: bool = False,
        aux_mse_weight: float = 0.0,
        aux_early_mse_weight: float = 0.0,
        aux_early_frames: int = 8,
        conditional_mean: bool = False,
        mean_only_training: bool = False,
        mean_loss_weight: float = 1.0,
        mean_log_mse_weight: float = 0.0,
        mean_ssim_weight: float = 0.0,
        mean_gradient_weight: float = 0.0,
        mean_temporal_gradient_weight: float = 0.0,
        mean_horizon_weight_end: float = 1.0,
        spatial_wavelet_only: bool = False,
        temporal_fourier_features: int = 0,
        scenario_onehot_conditioning: bool = False,
        pixel_refiner: bool = False,
        temporal_query_refiner: bool = False,
        visibility_only_model: bool = False,
        refiner_dim: int = 32,
        history_frames: int = 1,
        trajectory_frames: int = 32,
    ):
        super().__init__()
        self.model = model
        self.predict_residual = predict_residual
        self.wavelet_type = wavelet_type
        self.aux_lowfreq_weight = aux_lowfreq_weight
        self.aux_gradient_weight = aux_gradient_weight
        self.aux_ssim_weight = aux_ssim_weight
        self.aux_mse_weight = aux_mse_weight
        if aux_early_mse_weight < 0.0:
            raise ValueError("early-frame reconstruction-MSE weight must be non-negative")
        if aux_early_frames < 1:
            raise ValueError("early-frame count must be positive")
        self.aux_early_mse_weight = aux_early_mse_weight
        self.aux_early_frames = aux_early_frames
        self.conditional_mean = conditional_mean
        self.mean_only_training = mean_only_training
        if mean_only_training and not conditional_mean:
            raise ValueError("mean-only training requires conditional-mean modeling")
        if mean_loss_weight < 0.0:
            raise ValueError("conditional-mean MSE weight must be non-negative")
        if mean_log_mse_weight < 0.0:
            raise ValueError("conditional-mean log-MSE weight must be non-negative")
        if (
            mean_ssim_weight < 0.0
            or mean_gradient_weight < 0.0
            or mean_temporal_gradient_weight < 0.0
        ):
            raise ValueError("conditional-mean structural weights must be non-negative")
        if (
            mean_loss_weight
            + mean_log_mse_weight
            + mean_ssim_weight
            + mean_gradient_weight
            + mean_temporal_gradient_weight
            <= 0.0
        ):
            raise ValueError("at least one conditional-mean objective weight must be positive")
        self.mean_loss_weight = mean_loss_weight
        self.mean_log_mse_weight = mean_log_mse_weight
        self.mean_ssim_weight = mean_ssim_weight
        self.mean_gradient_weight = mean_gradient_weight
        self.mean_temporal_gradient_weight = mean_temporal_gradient_weight
        if mean_horizon_weight_end <= 0.0:
            raise ValueError("conditional-mean horizon weight must be positive")
        self.mean_horizon_weight_end = mean_horizon_weight_end
        if spatial_wavelet_only and wavelet_type != "haar":
            raise ValueError("spatial-only wavelet mode currently supports Haar only")
        self.spatial_wavelet_only = spatial_wavelet_only
        if temporal_fourier_features < 0:
            raise ValueError("temporal Fourier feature count must be non-negative")
        if temporal_fourier_features and not coordinate_conditioning:
            raise ValueError("temporal Fourier features require coordinate conditioning")
        self.temporal_fourier_features = temporal_fourier_features
        self.scenario_onehot_conditioning = scenario_onehot_conditioning
        self.visibility_only_model = visibility_only_model
        self.mean_model = copy.deepcopy(model) if conditional_mean else None
        if pixel_refiner and temporal_query_refiner:
            raise ValueError("Select only one pixel-refinement architecture")
        if visibility_only_model and (pixel_refiner or temporal_query_refiner):
            raise ValueError("Visibility-only prediction and pixel refinement are separate branches")
        if (pixel_refiner or temporal_query_refiner) and not conditional_mean:
            raise ValueError("Pixel residual refinement requires a conditional mean")
        if temporal_query_refiner:
            self.pixel_refiner = TemporalQueryResidualRefiner(
                history_frames, trajectory_frames, refiner_dim
            )
        elif pixel_refiner:
            self.pixel_refiner = PixelResidualRefiner(
                history_frames, trajectory_frames, refiner_dim
            )
        else:
            self.pixel_refiner = None
        self.aux_max_timestep = aux_max_timestep
        self.aux_gate_rescale = aux_gate_rescale
        if visibility_noise_weight <= 0.0:
            raise ValueError("visibility_noise_weight must be positive")
        self.visibility_noise_weight = visibility_noise_weight
        if visibility_aux_weight <= 0.0:
            raise ValueError("visibility_aux_weight must be positive")
        self.visibility_aux_weight = visibility_aux_weight
        if prediction_objective not in ("epsilon", "x0", "direct"):
            raise ValueError(f"Unsupported prediction objective: {prediction_objective}")
        self.prediction_objective = prediction_objective
        self.coordinate_conditioning = coordinate_conditioning
        betas = cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.diffusion_steps = diffusion_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)

    def prepare_target(self, trajectory_01: torch.Tensor, history_01: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H,W] -> [B,C,T,H,W] -> normalized wavelet coefficients
        target_field = trajectory_01[:, :, 1:2] if self.visibility_only_model else trajectory_01
        history_field = history_01[:, -1:, 1:2] if self.visibility_only_model else history_01[:, -1:]
        if self.predict_residual:
            physical = (target_field - history_field).permute(0, 2, 1, 3, 4)
        else:
            physical = target_field.permute(0, 2, 1, 3, 4) * 2.0 - 1.0
        if self.spatial_wavelet_only:
            return spatial_haar3d(physical) / 2.0
        coefficients, _ = wavelet3d(physical, self.wavelet_type)
        return coefficients / math.sqrt(8.0)

    def prepare_condition(
        self,
        history_01: torch.Tensor,
        scalar_condition_01: torch.Tensor,
        target_coeff_shape: Sequence[int],
    ) -> torch.Tensor:
        b, history_frames, channels, height, width = history_01.shape
        physical_history = history_01.reshape(b * history_frames, channels, height, width) * 2.0 - 1.0
        history_coeff = wavelet2d(physical_history, self.wavelet_type) / 2.0
        history_coeff = history_coeff.reshape(
            b, history_frames * channels * 4, history_coeff.shape[-2], history_coeff.shape[-1]
        )
        target_t, target_h, target_w = target_coeff_shape[-3:]
        if history_coeff.shape[-2:] != (target_h, target_w):
            raise ValueError(
                f"History/target coefficient mismatch: {history_coeff.shape[-2:]} vs {(target_h, target_w)}"
            )
        history_coeff = history_coeff[:, :, None].expand(-1, -1, target_t, -1, -1)
        scalars = scalar_condition_01 * 2.0 - 1.0
        scalar_maps = scalars[:, :, None, None, None].expand(
            -1, -1, target_t, target_h, target_w
        )
        parts = [history_coeff, scalar_maps]
        if self.coordinate_conditioning:
            time_fraction = torch.linspace(
                0.0, 1.0, target_t, device=history_01.device, dtype=history_01.dtype
            )[None, None, :, None, None]
            absolute_time_01 = (
                scalar_condition_01[:, 2:3, None, None, None]
                + history_frames / 1000.0
                + scalar_condition_01[:, 3:4, None, None, None] * time_fraction
            )
            absolute_time_01 = absolute_time_01.expand(-1, -1, -1, target_h, target_w)
            absolute_time = absolute_time_01 * 2.0 - 1.0
            height_grid = torch.linspace(
                -1.0, 1.0, target_h, device=history_01.device, dtype=history_01.dtype
            )[None, None, None, :, None].expand(b, -1, target_t, -1, target_w)
            width_grid = torch.linspace(
                -1.0, 1.0, target_w, device=history_01.device, dtype=history_01.dtype
            )[None, None, None, None, :].expand(b, -1, target_t, target_h, -1)
            parts.extend((absolute_time, height_grid, width_grid))
            for frequency_index in range(self.temporal_fourier_features):
                frequency = float(2 ** frequency_index)
                phase = 2.0 * math.pi * frequency * absolute_time_01
                parts.extend((phase.sin(), phase.cos()))
        if self.scenario_onehot_conditioning:
            hrr_levels = torch.tensor(
                (0.0, math.log(2.0) / math.log(10.0), math.log(4.0) / math.log(10.0), 1.0),
                device=history_01.device,
                dtype=history_01.dtype,
            )
            hrr_index = (
                scalar_condition_01[:, :1] - hrr_levels[None]
            ).abs().argmin(dim=1)
            ventilation_index = (scalar_condition_01[:, 1] >= 0.5).long()
            scenario_index = ventilation_index * 4 + hrr_index
            scenario_onehot = F.one_hot(scenario_index, num_classes=8).to(history_01.dtype)
            parts.append(
                scenario_onehot[:, :, None, None, None].expand(
                    -1, -1, target_t, target_h, target_w
                )
            )
        return torch.cat(parts, dim=1)

    def reconstruct(
        self,
        coeff: torch.Tensor,
        history_01: torch.Tensor,
        frames: int,
        height: int = 23,
        width: int = 480,
    ) -> torch.Tensor:
        field_channels = 1 if self.visibility_only_model else 2
        if self.spatial_wavelet_only:
            physical = inverse_spatial_haar3d(
                coeff * 2.0, field_channels, (frames, height, width)
            )
        else:
            physical = inverse_wavelet3d(
                coeff * math.sqrt(8.0), field_channels, (frames, height, width), self.wavelet_type
            )
        if self.predict_residual:
            initial_field = history_01[:, -1, 1:2] if self.visibility_only_model else history_01[:, -1]
            initial = initial_field.unsqueeze(2)
            return (physical + initial).clamp(0.0, 1.0).permute(0, 2, 1, 3, 4)
        return ((physical + 1.0) / 2.0).clamp(0.0, 1.0).permute(0, 2, 1, 3, 4)

    def format_prediction(
        self, prediction: torch.Tensor, history_01: torch.Tensor
    ) -> torch.Tensor:
        if not self.visibility_only_model:
            return prediction
        temperature = history_01[:, -1:, 0:1].expand(
            -1, prediction.shape[1], -1, -1, -1
        )
        return torch.cat((temperature, prediction), dim=2)

    def loss(
        self,
        trajectory_01: torch.Tensor,
        history_01: torch.Tensor,
        scalar_condition_01: torch.Tensor,
    ) -> torch.Tensor:
        clean = self.prepare_target(trajectory_01, history_01)
        target_field = trajectory_01[:, :, 1:2] if self.visibility_only_model else trajectory_01
        condition = self.prepare_condition(history_01, scalar_condition_01, clean.shape)
        batch = clean.shape[0]
        if self.mean_model is not None:
            mean_time = torch.zeros(batch, device=clean.device, dtype=torch.long)
            mean_clean = self.mean_model(
                torch.zeros_like(clean), condition, mean_time
            ).clamp(-1.0, 1.0)
            mean_field = self.reconstruct(
                mean_clean,
                history_01,
                target_field.shape[1],
                target_field.shape[-2],
                target_field.shape[-1],
            )
            if self.pixel_refiner is not None:
                mean_field = (
                    mean_field
                    + self.pixel_refiner(
                        history_01, scalar_condition_01, mean_field
                    )
                ).clamp(0.0, 1.0)
            if self.visibility_only_model:
                mean_field_weights = torch.ones(
                    1, 1, 1, 1, 1, device=clean.device, dtype=mean_field.dtype
                )
            else:
                mean_field_weights = torch.tensor(
                    (
                        2.0 / (1.0 + self.visibility_aux_weight),
                        2.0 * self.visibility_aux_weight / (1.0 + self.visibility_aux_weight),
                    ),
                    device=clean.device,
                    dtype=mean_field.dtype,
                )[None, None, :, None, None]
            mean_horizon_weights = torch.linspace(
                1.0,
                self.mean_horizon_weight_end,
                target_field.shape[1],
                device=clean.device,
                dtype=mean_field.dtype,
            )
            mean_horizon_weights = (
                mean_horizon_weights / mean_horizon_weights.mean()
            )[None, :, None, None, None]
            mean_loss = (
                (mean_field - target_field).square()
                * mean_field_weights
                * mean_horizon_weights
            ).mean()
            mean_total = self.mean_loss_weight * mean_loss
            if self.mean_log_mse_weight:
                per_frame_field_mse = (
                    (mean_field - trajectory_01).square().mean(dim=(-1, -2))
                )
                log_mse = torch.log1p(per_frame_field_mse / 1e-4)
                mean_channel_weights = mean_field_weights.flatten()
                mean_log_mse = (
                    log_mse
                    * mean_channel_weights[None, None, :]
                    * mean_horizon_weights[..., 0, 0].reshape(
                        1, target_field.shape[1], 1
                    )
                ).mean()
                mean_total = mean_total + self.mean_log_mse_weight * mean_log_mse
            if self.mean_gradient_weight:
                mean_dx = mean_field[..., 1:] - mean_field[..., :-1]
                target_dx = target_field[..., 1:] - target_field[..., :-1]
                mean_dy = mean_field[..., 1:, :] - mean_field[..., :-1, :]
                target_dy = target_field[..., 1:, :] - target_field[..., :-1, :]
                mean_gradient = (
                    (
                        (mean_dx - target_dx).abs()
                        * mean_field_weights
                        * mean_horizon_weights
                    ).mean()
                    + (
                        (mean_dy - target_dy).abs()
                        * mean_field_weights
                        * mean_horizon_weights
                    ).mean()
                )
                mean_total = mean_total + self.mean_gradient_weight * mean_gradient
            if self.mean_temporal_gradient_weight:
                mean_dt = mean_field[:, 1:] - mean_field[:, :-1]
                target_dt = target_field[:, 1:] - target_field[:, :-1]
                mean_temporal_gradient = (
                    (mean_dt - target_dt).square()
                    * mean_field_weights
                    * 0.5
                    * (
                        mean_horizon_weights[:, 1:]
                        + mean_horizon_weights[:, :-1]
                    )
                ).mean()
                mean_total = (
                    mean_total
                    + self.mean_temporal_gradient_weight * mean_temporal_gradient
                )
            if self.mean_ssim_weight:
                mean_ssim_losses = []
                mean_channel_weights = mean_field_weights.flatten()
                for channel in range(target_field.shape[2]):
                    mean_images = mean_field[:, :, channel].flatten(0, 1)
                    target_images = target_field[:, :, channel].flatten(0, 1)
                    per_image_ssim = ssim_per_image(
                        mean_images, target_images
                    ).reshape(target_field.shape[0], target_field.shape[1])
                    horizon_weight = mean_horizon_weights[0, :, 0, 0, 0]
                    mean_ssim_losses.append(
                        ((1.0 - per_image_ssim) * horizon_weight[None]).mean()
                    )
                mean_ssim_loss = (
                    torch.stack(mean_ssim_losses) * mean_channel_weights
                ).mean()
                mean_total = mean_total + self.mean_ssim_weight * mean_ssim_loss
            if self.mean_only_training:
                return mean_total
        else:
            mean_total = clean.new_zeros(())
        if self.prediction_objective == "direct":
            t = torch.zeros(batch, device=clean.device, dtype=torch.long)
            noise = torch.zeros_like(clean)
            alpha = self.alphas_cumprod[t].reshape(batch, 1, 1, 1, 1)
            noisy = torch.zeros_like(clean)
        else:
            t = torch.randint(0, self.diffusion_steps, (batch,), device=clean.device)
            noise = torch.randn_like(clean)
            alpha = self.alphas_cumprod[t].reshape(batch, 1, 1, 1, 1)
            noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
        model_output = self.model(noisy, condition, t)
        training_target = noise if self.prediction_objective == "epsilon" else clean
        squared_noise_error = (model_output.float() - training_target.float()).square()
        if self.visibility_only_model or self.visibility_noise_weight == 1.0:
            total = squared_noise_error.mean()
        else:
            field_channels = squared_noise_error.shape[1] // 2
            channel_weights = torch.ones(
                squared_noise_error.shape[1], device=clean.device,
                dtype=squared_noise_error.dtype,
            )
            channel_weights[field_channels:] = self.visibility_noise_weight
            channel_weights = channel_weights * (
                2.0 / (1.0 + self.visibility_noise_weight)
            )
            total = (
                squared_noise_error
                * channel_weights[None, :, None, None, None]
            ).mean()

        if (
            self.aux_lowfreq_weight
            or self.aux_gradient_weight
            or self.aux_ssim_weight
            or self.aux_mse_weight
            or self.aux_early_mse_weight
        ):
            if self.aux_max_timestep < 0:
                aux_mask = torch.ones_like(t, dtype=torch.bool)
                aux_scale = 1.0
            else:
                cutoff = min(self.aux_max_timestep, self.diffusion_steps - 1)
                aux_mask = t <= cutoff
                # Preserve the expected auxiliary-loss magnitude while excluding
                # unreliable x0 reconstructions from high-noise timesteps.
                aux_scale = (
                    self.diffusion_steps / float(cutoff + 1)
                    if self.aux_gate_rescale else 1.0
                )
            if not aux_mask.any():
                return total + mean_total
            if self.prediction_objective == "epsilon":
                predicted_clean = (
                    noisy - (1.0 - alpha).sqrt() * model_output
                ) / alpha.sqrt().clamp_min(1e-4)
            else:
                predicted_clean = model_output
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)[aux_mask]
            clean_aux = clean[aux_mask]
            history_aux = history_01[aux_mask]
            trajectory_aux = target_field[aux_mask]
            if self.visibility_only_model:
                aux_field_weights = torch.ones(
                    1, device=clean.device, dtype=predicted_clean.dtype
                )
            else:
                aux_field_weights = torch.tensor(
                    (2.0 / (1.0 + self.visibility_aux_weight),
                     2.0 * self.visibility_aux_weight / (1.0 + self.visibility_aux_weight)),
                    device=clean.device, dtype=predicted_clean.dtype,
                )
            if self.aux_lowfreq_weight:
                low_indices = torch.tensor(
                    (0,) if self.visibility_only_model else (0, 8), device=clean.device
                )
                low_error = (
                    predicted_clean.index_select(1, low_indices)
                    - clean_aux.index_select(1, low_indices)
                ).abs()
                low_loss = (
                    low_error * aux_field_weights[None, :, None, None, None]
                ).mean()
                total = total + aux_scale * self.aux_lowfreq_weight * low_loss
            if (
                self.aux_gradient_weight
                or self.aux_ssim_weight
                or self.aux_mse_weight
                or self.aux_early_mse_weight
            ):
                predicted_field = self.reconstruct(
                    predicted_clean, history_aux, trajectory_aux.shape[1],
                    trajectory_aux.shape[-2], trajectory_aux.shape[-1],
                )
                if self.aux_mse_weight:
                    field_weights = aux_field_weights[None, None, :, None, None]
                    reconstruction_mse = (
                        (predicted_field - trajectory_aux).square() * field_weights
                    ).mean()
                    total = total + aux_scale * self.aux_mse_weight * reconstruction_mse
                if self.aux_early_mse_weight:
                    early_frames = min(self.aux_early_frames, predicted_field.shape[1])
                    # Emphasize the shortest lead times, where a forecasting
                    # model should preserve the observed field most precisely.
                    temporal_weights = torch.linspace(
                        1.0,
                        0.25,
                        early_frames,
                        device=predicted_field.device,
                        dtype=predicted_field.dtype,
                    )
                    temporal_weights = temporal_weights / temporal_weights.mean()
                    early_error = (
                        predicted_field[:, :early_frames]
                        - trajectory_aux[:, :early_frames]
                    ).square()
                    early_mse = (
                        early_error
                        * aux_field_weights[None, None, :, None, None]
                        * temporal_weights[None, :, None, None, None]
                    ).mean()
                    total = total + aux_scale * self.aux_early_mse_weight * early_mse
                if self.aux_gradient_weight:
                    pred_dx = predicted_field[..., 1:] - predicted_field[..., :-1]
                    true_dx = trajectory_aux[..., 1:] - trajectory_aux[..., :-1]
                    pred_dy = predicted_field[..., 1:, :] - predicted_field[..., :-1, :]
                    true_dy = trajectory_aux[..., 1:, :] - trajectory_aux[..., :-1, :]
                    field_weights = aux_field_weights[None, None, :, None, None]
                    gradient_loss = (
                        ((pred_dx - true_dx).abs() * field_weights).mean()
                        + ((pred_dy - true_dy).abs() * field_weights).mean()
                    )
                    total = total + aux_scale * self.aux_gradient_weight * gradient_loss
                if self.aux_ssim_weight:
                    field_ssim_losses = []
                    for channel in range(trajectory_aux.shape[2]):
                        pred_images = predicted_field[:, :, channel].flatten(0, 1)
                        true_images = trajectory_aux[:, :, channel].flatten(0, 1)
                        field_ssim_losses.append(
                            1.0 - ssim_per_image(pred_images, true_images).mean()
                        )
                    ssim_loss = torch.stack(field_ssim_losses).mul(aux_field_weights).mean()
                    total = total + aux_scale * self.aux_ssim_weight * ssim_loss
        return total + mean_total

    @torch.no_grad()
    def make_initial_noise(self, history_01: torch.Tensor, frames: int) -> torch.Tensor:
        dummy = torch.zeros(
            history_01.shape[0], 1 if self.visibility_only_model else 2,
            frames, history_01.shape[-2], history_01.shape[-1],
            device=history_01.device, dtype=history_01.dtype,
        )
        target_coeff = (
            spatial_haar3d(dummy)
            if self.spatial_wavelet_only
            else wavelet3d(dummy, self.wavelet_type)[0]
        )
        return torch.randn(
            history_01.shape[0], target_coeff.shape[1], *target_coeff.shape[-3:],
            device=history_01.device, dtype=history_01.dtype,
        )

    @torch.no_grad()
    def sample(
        self,
        history_01: torch.Tensor,
        scalar_condition_01: torch.Tensor,
        frames: int = 32,
        sampling_steps: int = 100,
        eta: float = 0.0,
        initialization: str = "random",
        initial_noise: torch.Tensor | None = None,
        start_timestep: int = -1,
        mean_blend: float = 0.0,
    ) -> torch.Tensor:
        dummy = torch.zeros(
            history_01.shape[0], 1 if self.visibility_only_model else 2,
            frames, history_01.shape[-2], history_01.shape[-1],
            device=history_01.device, dtype=history_01.dtype,
        )
        target_coeff = (
            spatial_haar3d(dummy)
            if self.spatial_wavelet_only
            else wavelet3d(dummy, self.wavelet_type)[0]
        )
        condition = self.prepare_condition(history_01, scalar_condition_01, target_coeff.shape)
        shape = (
            history_01.shape[0],
            target_coeff.shape[1],
            condition.shape[2],
            condition.shape[3],
            condition.shape[4],
        )
        if not 0.0 <= mean_blend <= 1.0:
            raise ValueError("mean blend must be in [0, 1]")
        mean_clean = None
        if self.mean_model is not None:
            mean_clean = self.mean_model(
                torch.zeros(shape, device=history_01.device, dtype=history_01.dtype),
                condition,
                torch.zeros(shape[0], device=history_01.device, dtype=torch.long),
            ).clamp(-1.0, 1.0)
        if initialization == "mean":
            if mean_clean is None:
                raise ValueError("Mean initialization requires a conditional-mean checkpoint")
            mean_field = self.reconstruct(
                mean_clean, history_01, frames, history_01.shape[-2], history_01.shape[-1]
            )
            if self.pixel_refiner is not None:
                if frames != self.pixel_refiner.trajectory_frames:
                    raise ValueError(
                        "Pixel refiner supports the configured trajectory length only"
                    )
                mean_field = (
                    mean_field
                    + self.pixel_refiner(
                        history_01, scalar_condition_01, mean_field
                    )
                ).clamp(0.0, 1.0)
            return self.format_prediction(mean_field, history_01)
        if self.prediction_objective == "direct":
            clean = self.model(
                torch.zeros(shape, device=history_01.device, dtype=history_01.dtype),
                condition,
                torch.zeros(shape[0], device=history_01.device, dtype=torch.long),
            ).clamp(-1.0, 1.0)
            return self.format_prediction(self.reconstruct(
                clean, history_01, frames, history_01.shape[-2], history_01.shape[-1]
            ), history_01)
        start = self.diffusion_steps - 1 if start_timestep < 0 else start_timestep
        if not 0 <= start < self.diffusion_steps:
            raise ValueError(
                f"start_timestep must be in [0, {self.diffusion_steps - 1}], received {start}"
            )
        if start < self.diffusion_steps - 1 and not self.predict_residual:
            raise ValueError("Intermediate-timestep warm start requires residual prediction")
        if initial_noise is not None:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"Initial-noise shape mismatch: {tuple(initial_noise.shape)} vs {shape}"
                )
            sample = initial_noise.to(device=history_01.device, dtype=history_01.dtype)
        elif initialization == "random":
            sample = torch.randn(shape, device=history_01.device)
        elif initialization == "zero":
            sample = torch.zeros(shape, device=history_01.device)
        else:
            raise ValueError(f"Unsupported diffusion initialization: {initialization}")
        if start < self.diffusion_steps - 1:
            # q(x_t | x0=0): a history-persistence prior in residual space.
            sample = sample * (1.0 - self.alphas_cumprod[start]).sqrt()
        times = torch.linspace(-1, start, sampling_steps + 1, device=sample.device)
        times = list(reversed(times.long().tolist()))
        for current, following in zip(times[:-1], times[1:]):
            t = torch.full((shape[0],), current, device=sample.device, dtype=torch.long)
            model_output = self.model(sample, condition, t)
            alpha = self.alphas_cumprod[current]
            if self.prediction_objective == "epsilon":
                predicted_noise = model_output
                clean = (
                    (sample - (1.0 - alpha).sqrt() * predicted_noise)
                    / alpha.sqrt().clamp_min(1e-4)
                ).clamp(-1.0, 1.0)
            else:
                clean = model_output.clamp(-1.0, 1.0)
                predicted_noise = (
                    sample - alpha.sqrt() * clean
                ) / (1.0 - alpha).sqrt().clamp_min(1e-4)
            if following < 0:
                sample = clean
                continue
            alpha_next = self.alphas_cumprod[following]
            sigma = eta * ((1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)).sqrt()
            direction = (1.0 - alpha_next - sigma.square()).clamp_min(0.0).sqrt() * predicted_noise
            noise = torch.randn_like(sample) if eta > 0 else 0.0
            sample = alpha_next.sqrt() * clean + direction + sigma * noise
        if mean_blend:
            if mean_clean is None:
                raise ValueError("Mean blending requires a conditional-mean checkpoint")
            sample = (1.0 - mean_blend) * sample + mean_blend * mean_clean
        return self.format_prediction(self.reconstruct(
            sample, history_01, frames, history_01.shape[-2], history_01.shape[-1]
        ), history_01)


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def gaussian_window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def ssim_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard local-window SSIM for [N,H,W] fields in the [0,1] range."""

    x = prediction[:, None]
    y = target[:, None]
    window = gaussian_window(7, 1.5, x.device, x.dtype)
    mu_x = F.conv2d(x, window, padding=0)
    mu_y = F.conv2d(y, window, padding=0)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(x.square(), window, padding=0) - mu_x2
    sigma_y2 = F.conv2d(y.square(), window, padding=0) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=0) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.mean(dim=(1, 2, 3))


def postprocess_visibility(
    prediction: torch.Tensor,
    spatial_sigma: float = 0.0,
    temporal_blend: float = 0.0,
    scale: float = 1.0,
    scale_start: float | None = None,
    scale_end: float | None = None,
    bias: float = 0.0,
    gamma: float = 1.0,
    median_blend: float = 0.0,
) -> torch.Tensor:
    """Apply fixed target-free smoothing to the visibility forecast only."""

    if spatial_sigma < 0.0:
        raise ValueError("spatial_sigma must be non-negative")
    if not 0.0 <= temporal_blend <= 1.0:
        raise ValueError("temporal_blend must be in [0, 1]")
    if scale <= 0.0:
        raise ValueError("visibility scale must be positive")
    if (scale_start is None) != (scale_end is None):
        raise ValueError("visibility scale start and end must be specified together")
    if scale_start is not None and (scale_start <= 0.0 or scale_end <= 0.0):
        raise ValueError("visibility time-varying scales must be positive")
    if gamma <= 0.0:
        raise ValueError("visibility gamma must be positive")
    if not 0.0 <= median_blend <= 1.0:
        raise ValueError("visibility median blend must be in [0, 1]")
    if (
        spatial_sigma == 0.0
        and temporal_blend == 0.0
        and scale == 1.0
        and scale_start is None
        and bias == 0.0
        and gamma == 1.0
        and median_blend == 0.0
    ):
        return prediction
    output = prediction.clone()
    visibility = output[:, :, 1]
    if spatial_sigma > 0.0:
        radius = max(1, int(math.ceil(3.0 * spatial_sigma)))
        coordinates = torch.arange(
            -radius, radius + 1, device=visibility.device, dtype=visibility.dtype
        )
        kernel_1d = torch.exp(-coordinates.square() / (2.0 * spatial_sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
        images = visibility.flatten(0, 1)[:, None]
        images = F.pad(images, (radius, radius, radius, radius), mode="reflect")
        visibility = F.conv2d(images, kernel_2d).reshape_as(visibility)
    if temporal_blend > 0.0:
        previous = torch.cat((visibility[:, :1], visibility[:, :-1]), dim=1)
        following = torch.cat((visibility[:, 1:], visibility[:, -1:]), dim=1)
        visibility = (
            (1.0 - temporal_blend) * visibility
            + temporal_blend * 0.5 * (previous + following)
        )
    if median_blend > 0.0:
        images = visibility.flatten(0, 1)[:, None]
        padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
        median = F.unfold(padded, kernel_size=3).median(dim=1).values
        median = median.reshape_as(visibility)
        visibility = (1.0 - median_blend) * visibility + median_blend * median
    if scale_start is not None:
        time_scale = torch.linspace(
            scale_start,
            scale_end,
            visibility.shape[1],
            device=visibility.device,
            dtype=visibility.dtype,
        )[None, :, None, None]
        visibility = visibility * time_scale
    output[:, :, 1] = (visibility * scale + bias).clamp(0.0, 1.0).pow(gamma)
    return output


class MetricAccumulator:
    def __init__(self):
        self.squared_error = np.zeros(2, dtype=np.float64)
        self.elements = np.zeros(2, dtype=np.int64)
        self.ssim_sum = np.zeros(2, dtype=np.float64)
        self.psnr_sum = np.zeros(2, dtype=np.float64)
        self.images = np.zeros(2, dtype=np.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor, exclude_first: bool = False) -> None:
        if exclude_first:
            prediction, target = prediction[:, 1:], target[:, 1:]
        for channel in range(2):
            pred = prediction[:, :, channel].float()
            truth = target[:, :, channel].float()
            error = pred - truth
            self.squared_error[channel] += error.square().sum().item()
            self.elements[channel] += error.numel()
            pred_images = pred.flatten(0, 1)
            truth_images = truth.flatten(0, 1)
            self.ssim_sum[channel] += ssim_per_image(pred_images, truth_images).sum().item()
            image_mse = error.square().mean(dim=(-1, -2)).clamp_min(1e-12)
            self.psnr_sum[channel] += (10.0 * torch.log10(1.0 / image_mse)).sum().item()
            self.images[channel] += image_mse.numel()

    def result(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for channel, name in enumerate(FIELD_NAMES):
            raw_mse = self.squared_error[channel] / self.elements[channel]
            result[name] = {
                "mse_raw": float(raw_mse),
                "mse_x100": float(raw_mse * 100.0),
                "ssim": float(self.ssim_sum[channel] / self.images[channel]),
                "psnr": float(self.psnr_sum[channel] / self.images[channel]),
            }
        return result


def save_checkpoint(
    path: Path,
    model: FireWaveModel,
    ema_model: FireWaveModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Config,
    store: FITStore,
    best_val: float,
    best_generated_ssim: float = float("-inf"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "normalization": {"minimum": store.minimum.tolist(), "maximum": store.maximum.tolist()},
            "best_val": best_val,
            "best_generated_ssim": best_generated_ssim,
        },
        temporary,
    )
    temporary.replace(path)


def load_compatible_state(module: nn.Module, source: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Load matching tensors and zero-init newly appended conditioning channels."""

    target = module.state_dict()
    exact = 0
    expanded = 0
    skipped = 0
    initialized_mean = 0
    for key, value in source.items():
        if key not in target:
            skipped += 1
            continue
        if target[key].shape == value.shape:
            target[key] = value
            exact += 1
            continue
        if (
            key in ("model.input.weight", "mean_model.input.weight")
            and target[key].ndim == value.ndim
            and target[key].shape[0] == value.shape[0]
            and target[key].shape[2:] == value.shape[2:]
            and target[key].shape[1] > value.shape[1]
        ):
            merged = target[key].clone()
            merged[:, value.shape[1] :].zero_()
            merged[:, : value.shape[1]] = value
            target[key] = merged
            expanded += 1
            continue
        if (
            key in ("model.input.weight", "mean_model.input.weight")
            and target[key].ndim == value.ndim
            and target[key].shape[0] == value.shape[0]
            and target[key].shape[2:] == value.shape[2:]
            and target[key].shape[1] < value.shape[1]
            and isinstance(module, FireWaveModel)
            and module.visibility_only_model
        ):
            branch = module.model if key.startswith("model.") else module.mean_model
            new_target_channels = branch.target_channels
            condition_channels = target[key].shape[1] - new_target_channels
            old_target_channels = value.shape[1] - condition_channels
            if old_target_channels == 2 * new_target_channels:
                merged = target[key].clone()
                merged[:, :new_target_channels] = value[
                    :, old_target_channels - new_target_channels : old_target_channels
                ]
                merged[:, new_target_channels:] = value[:, old_target_channels:]
                target[key] = merged
                expanded += 1
                continue
        if (
            key in ("model.output.2.weight", "mean_model.output.2.weight")
            and target[key].ndim == value.ndim
            and target[key].shape[0] * 2 == value.shape[0]
            and target[key].shape[1:] == value.shape[1:]
        ):
            target[key] = value[value.shape[0] // 2 :].clone()
            expanded += 1
            continue
        if (
            key in ("model.output.2.bias", "mean_model.output.2.bias")
            and target[key].ndim == value.ndim == 1
            and target[key].shape[0] * 2 == value.shape[0]
        ):
            target[key] = value[value.shape[0] // 2 :].clone()
            expanded += 1
            continue
        skipped += 1
    # A newly enabled conditional-mean branch starts from the trained denoiser
    # instead of an unrelated random network. Native mean-model weights in a
    # newer checkpoint are loaded by the exact-match path above afterwards.
    for key, value in source.items():
        if not key.startswith("model."):
            continue
        mean_key = "mean_model." + key[len("model."):]
        if (
            mean_key in target
            and mean_key not in source
            and target[mean_key].shape == value.shape
        ):
            target[mean_key] = value
            initialized_mean += 1
    module.load_state_dict(target)
    return {
        "exact": exact,
        "expanded": expanded,
        "skipped": skipped,
        "initialized_mean": initialized_mean,
    }


def validation_loss(model: FireWaveModel, loader: DataLoader, device: torch.device, batches: int = 8) -> float:
    model.eval()
    losses: List[float] = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        trajectory, history, condition = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses.append(model.loss(trajectory, history, condition).item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def validation_generated_ssim(
    model: FireWaveModel,
    loader: DataLoader,
    device: torch.device,
    samples: int,
    sampling_steps: int,
    selection_field: str = "mean",
    start_timestep: int = -1,
    ensemble_size: int = 1,
    antithetic: bool = False,
    visibility_spatial_sigma: float = 0.0,
    visibility_temporal_blend: float = 0.0,
    visibility_median_blend: float = 0.0,
    psnr_weight: float = 0.0,
    initialization: str = "random",
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Evaluate generated fields with a fixed noise seed for checkpoint selection."""

    was_training = model.training
    model.eval()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    torch.manual_seed(3407)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(3407)
    scores: List[List[torch.Tensor]] = [[], []]
    psnr_scores: List[List[torch.Tensor]] = [[], []]
    seen = 0
    for batch in loader:
        target = batch[0].to(device)
        history = batch[1].to(device)
        condition = batch[2].to(device)
        predictions = sample_ensemble(
            model,
            history,
            condition,
            target.shape[1],
            sampling_steps,
            ensemble_size,
            initialization,
            antithetic,
            start_timestep,
        )
        prediction = torch.stack(predictions).mean(0)
        prediction = postprocess_visibility(
            prediction,
            spatial_sigma=visibility_spatial_sigma,
            temporal_blend=visibility_temporal_blend,
            median_blend=visibility_median_blend,
        )
        for channel in range(2):
            predicted_field = prediction[:, :, channel].float()
            target_field = target[:, :, channel].float()
            scores[channel].append(
                ssim_per_image(predicted_field.flatten(0, 1), target_field.flatten(0, 1))
            )
            image_mse = (predicted_field - target_field).square().mean(dim=(-1, -2))
            psnr_scores[channel].append(10.0 * torch.log10(1.0 / image_mse.clamp_min(1e-12)))
        seen += target.shape[0]
        if seen >= samples:
            break
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if was_training:
        model.train()
    field_scores = {
        name: float(torch.cat(scores[channel]).mean().item())
        for channel, name in enumerate(FIELD_NAMES)
    }
    field_psnr = {
        name: float(torch.cat(psnr_scores[channel]).mean().item())
        for channel, name in enumerate(FIELD_NAMES)
    }
    if selection_field == "mean":
        selected_score = sum(field_scores.values()) / len(field_scores)
    elif selection_field in field_scores:
        selected_score = field_scores[selection_field]
    else:
        raise ValueError(f"Unsupported selection field: {selection_field}")
    if psnr_weight:
        if selection_field == "mean":
            selected_score += psnr_weight * sum(field_psnr.values()) / len(field_psnr)
        else:
            selected_score += psnr_weight * field_psnr[selection_field]
    return float(selected_score), field_scores, field_psnr


def train(config: Config, resume: str | None = None) -> Path:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("FIT FireWave training requires a CUDA worker for this configuration")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    store = FITStore(config.data_root)
    segment_index = make_segment_index(
        history_frames=config.history_frames,
        include_full_span=config.include_full_span,
        full_span_repeats=config.full_span_repeats,
    )
    train_index, val_index, test_index = split_segment_index(segment_index, config.split_seed)
    scenario_ids = set(selected_scenario_ids(config))
    train_index = [sample for sample in train_index if sample[0] in scenario_ids]
    val_index = [sample for sample in val_index if sample[0] in scenario_ids]
    test_index = [sample for sample in test_index if sample[0] in scenario_ids]
    if config.strict_protected_min_span < 0:
        raise ValueError("strict protected minimum span must be non-negative")
    if config.strict_train_min_span < 0:
        raise ValueError("strict training minimum span must be non-negative")
    if config.strict_train_min_span and not config.strict_frame_disjoint_train:
        raise ValueError("strict training minimum span requires frame disjointness")
    if (
        config.strict_balance_scenario_span
        and not config.strict_frame_disjoint_train
    ):
        raise ValueError("strict balanced sampling requires frame disjointness")
    if config.strict_protected_min_span:
        val_index = [
            sample for sample in val_index
            if sample[2] >= config.strict_protected_min_span
        ]
        test_index = [
            sample for sample in test_index
            if sample[2] >= config.strict_protected_min_span
        ]
    if config.train_short_only:
        # Preserve the canonical split membership before selecting the target
        # horizon.  Filtering before splitting would reshuffle short windows
        # and leak part of the canonical test set into training.
        train_index = [sample for sample in train_index if sample[2] == config.table2_span]
        val_index = [sample for sample in val_index if sample[2] == config.table2_span]
        test_index = [sample for sample in test_index if sample[2] == config.table2_span]
    if config.train_start_stride < 1 or config.train_start_stride > 32:
        raise ValueError("training start stride must be in [1, 32]")
    if config.weight_decay < 0.0:
        raise ValueError("weight decay must be non-negative")
    if config.train_start_stride != 32:
        if not config.train_short_only:
            raise ValueError("dense training starts currently require --train-short-only")
        # Densify only the optimization set.  The canonical validation and
        # test tuples remain unchanged, and their exact windows are excluded
        # from the augmented training list.
        held_out = set(val_index) | set(test_index)
        max_start = 1001 - 1 - config.history_frames - config.table2_span
        train_index = [
            (scenario_id, start, config.table2_span)
            for scenario_id in range(len(SCENARIOS))
            for start in range(0, max_start + 1, config.train_start_stride)
            if (scenario_id, start, config.table2_span) not in held_out
        ]
    if config.strict_dense_variable_train:
        if not config.strict_frame_disjoint_train:
            raise ValueError(
                "Dense variable-span training requires strict frame disjointness"
            )
        span_values = sorted({sample[2] for sample in segment_index})
        train_index = [
            (scenario_id, start, span)
            for scenario_id in sorted(scenario_ids)
            for span in span_values
            for start in range(
                0,
                1001 - config.history_frames - span,
            )
        ]
    if config.strict_frame_disjoint_train:
        protected_frame_keys = sample_frame_keys(
            list(val_index) + list(test_index),
            config.history_frames,
            config.trajectory_frames,
        )
        strict_train_index = []
        for sample in train_index:
            if sample_frame_keys(
                (sample,),
                config.history_frames,
                config.trajectory_frames,
            ).isdisjoint(protected_frame_keys):
                strict_train_index.append(sample)
        if not strict_train_index:
            raise RuntimeError(
                "Strict frame-disjoint filtering produced no training samples"
            )
        train_index = strict_train_index
        if config.strict_train_min_span:
            train_index = [
                sample for sample in train_index
                if sample[2] >= config.strict_train_min_span
            ]
            if not train_index:
                raise RuntimeError(
                    "Strict training minimum span removed all samples"
                )
        training_frame_keys = sample_frame_keys(
            train_index,
            config.history_frames,
            config.trajectory_frames,
        )
        overlap = training_frame_keys & protected_frame_keys
        if overlap:
            raise RuntimeError(
                f"Strict train/evaluation frame overlap: {sorted(overlap)[:5]}"
            )
        print(
            json.dumps({
                "strict_data_audit": {
                    "train_samples": len(train_index),
                    "validation_samples": len(val_index),
                    "heldout_samples": len(test_index),
                    "training_minimum_span": config.strict_train_min_span,
                    "balanced_scenario_span_sampling": (
                        config.strict_balance_scenario_span
                    ),
                    "training_frame_keys": len(training_frame_keys),
                    "protected_validation_test_frame_keys": len(
                        protected_frame_keys
                    ),
                    "train_evaluation_frame_overlap": 0,
                }
            }),
            flush=True,
        )
    if config.selection_min_span < 0:
        raise ValueError("selection minimum span must be non-negative")
    selection_index = [
        sample for sample in val_index
        if sample[2] >= config.selection_min_span
    ]
    if config.select_by_ssim and not selection_index:
        raise RuntimeError(
            "No validation samples satisfy the selection minimum span"
        )
    train_dataset = FITTrajectoryDataset(
        store, train_index, config.trajectory_frames, config.history_frames
    )
    val_dataset = FITTrajectoryDataset(
        store, val_index, config.trajectory_frames, config.history_frames
    )
    selection_dataset = FITTrajectoryDataset(
        store, selection_index, config.trajectory_frames, config.history_frames
    )
    train_sampler = None
    if config.strict_balance_scenario_span:
        group_counts: dict[tuple[int, int], int] = {}
        for scenario, _, span in train_index:
            key = (int(scenario), int(span))
            group_counts[key] = group_counts.get(key, 0) + 1
        sample_weights = [
            1.0 / group_counts[(int(scenario), int(span))]
            for scenario, _, span in train_index
        ]
        generator = torch.Generator().manual_seed(config.seed)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_index),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    history_condition_channels = config.history_frames * 2 * 4
    condition_channels = (
        history_condition_channels
        + 4
        + (3 + 2 * config.temporal_fourier_features if config.coordinate_conditioning else 0)
        + (8 if config.scenario_onehot_conditioning else 0)
    )
    physical_fields = 1 if config.visibility_only_model else 2
    target_channels = physical_fields * (4 if config.spatial_wavelet_only else 8)
    denoiser = FITWaveletUNet(
        target_channels=target_channels,
        condition_channels=condition_channels,
        dim=config.model_dim,
        anisotropic=config.anisotropic,
        dropout=config.dropout,
        global_conditioning=config.global_conditioning,
        coordinate_conditioning=config.coordinate_conditioning,
        scalar_condition_start=history_condition_channels,
    )
    model = FireWaveModel(
        denoiser,
        config.diffusion_steps,
        config.predict_residual,
        config.wavelet_type,
        config.aux_lowfreq_weight,
        config.aux_gradient_weight,
        config.aux_ssim_weight,
        config.aux_max_timestep,
        config.aux_gate_rescale,
        config.visibility_noise_weight,
        config.visibility_aux_weight,
        config.prediction_objective,
        config.coordinate_conditioning,
        aux_mse_weight=config.aux_mse_weight,
        aux_early_mse_weight=config.aux_early_mse_weight,
        aux_early_frames=config.aux_early_frames,
        conditional_mean=config.conditional_mean,
        mean_only_training=config.mean_only_training,
        mean_loss_weight=config.mean_loss_weight,
        mean_log_mse_weight=config.mean_log_mse_weight,
        mean_ssim_weight=config.mean_ssim_weight,
        mean_gradient_weight=config.mean_gradient_weight,
        mean_temporal_gradient_weight=config.mean_temporal_gradient_weight,
        mean_horizon_weight_end=config.mean_horizon_weight_end,
        spatial_wavelet_only=config.spatial_wavelet_only,
        temporal_fourier_features=config.temporal_fourier_features,
        scenario_onehot_conditioning=config.scenario_onehot_conditioning,
        pixel_refiner=config.pixel_refiner,
        temporal_query_refiner=config.temporal_query_refiner,
        visibility_only_model=config.visibility_only_model,
        refiner_dim=config.refiner_dim,
        history_frames=config.history_frames,
        trajectory_frames=config.trajectory_frames,
    ).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    if config.refiner_only_training:
        if model.pixel_refiner is None or model.mean_model is None:
            raise ValueError("Refiner-only training requires the pixel-refiner mean branch")
        model.model.requires_grad_(False)
        model.mean_model.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.pixel_refiner.parameters(),
            lr=config.refiner_learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.mean_only_training:
        model.model.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.mean_model.parameters(),
            lr=config.mean_learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.conditional_mean:
        mean_parameters = list(model.mean_model.parameters())
        mean_parameter_ids = {id(parameter) for parameter in mean_parameters}
        diffusion_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in mean_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": diffusion_parameters, "lr": config.learning_rate},
                {"params": mean_parameters, "lr": config.mean_learning_rate},
            ],
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    step = 0
    best_val = float("inf")
    best_generated_ssim = float("-inf")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"

    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if config.allow_partial_resume:
            if not config.reset_optimizer_on_resume:
                raise ValueError("Partial resume requires --reset-optimizer")
            model_report = load_compatible_state(model, state["model"])
            ema_report = load_compatible_state(ema_model, state["ema"])
            print(
                json.dumps({"partial_resume": {"model": model_report, "ema": ema_report}}),
                flush=True,
            )
        else:
            model.load_state_dict(state["model"])
            ema_model.load_state_dict(state["ema"])
        if not config.reset_optimizer_on_resume:
            optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        if config.allow_partial_resume:
            # The expanded model is a new experimental branch.  Re-select its
            # own checkpoints even though the global optimization step is kept.
            best_val = float("inf")
            best_generated_ssim = float("-inf")
        else:
            best_val = float(state.get("best_val", best_val))
            best_generated_ssim = float(state.get("best_generated_ssim", best_generated_ssim))
            previous_config = state.get("config", {})
            previous_selection_protocol = (
                previous_config.get("selection_field", "mean"),
                previous_config.get("selection_start_timestep", -1),
                previous_config.get("selection_ensemble_size", 1),
                previous_config.get("selection_antithetic", False),
                previous_config.get("selection_visibility_spatial_sigma", 0.0),
                previous_config.get("selection_visibility_temporal_blend", 0.0),
                previous_config.get("selection_visibility_median_blend", 0.0),
                previous_config.get("selection_psnr_weight", 0.0),
                previous_config.get("selection_initialization", "random"),
                previous_config.get("selection_min_span", 0),
                tuple(previous_config.get("scenario_ids") or range(len(SCENARIOS))),
            )
            current_selection_protocol = (
                config.selection_field,
                config.selection_start_timestep,
                config.selection_ensemble_size,
                config.selection_antithetic,
                config.selection_visibility_spatial_sigma,
                config.selection_visibility_temporal_blend,
                config.selection_visibility_median_blend,
                config.selection_psnr_weight,
                config.selection_initialization,
                config.selection_min_span,
                selected_scenario_ids(config),
            )
            if previous_selection_protocol != current_selection_protocol:
                best_generated_ssim = float("-inf")

    iterator = iter(train_loader)
    start_time = time.time()
    model.train()
    while step < config.train_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        trajectory = batch[0].to(device, non_blocking=True)
        history = batch[1].to(device, non_blocking=True)
        condition = batch[2].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.loss(trajectory, history, condition)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema_model, model, config.ema_decay)
        step += 1

        record = None
        if step == 1 or step % config.log_every == 0:
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "elapsed_seconds": time.time() - start_time,
            }
            print(json.dumps(record), flush=True)
        if step % config.validate_every == 0 or step == config.train_steps:
            val = validation_loss(ema_model, val_loader, device)
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_noise_mse": val,
                "elapsed_seconds": time.time() - start_time,
            }
            print(json.dumps(record), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if val < best_val:
                best_val = val
                save_checkpoint(
                    output / "best.pt", model, ema_model, optimizer, step, config, store,
                    best_val, best_generated_ssim,
                )
        if config.select_by_ssim and (
            step % config.selection_every == 0 or step == config.train_steps
        ):
            generated_score, generated_field_ssim, generated_field_psnr = validation_generated_ssim(
                ema_model,
                selection_loader,
                device,
                config.selection_samples,
                config.selection_ddim_steps,
                config.selection_field,
                config.selection_start_timestep,
                config.selection_ensemble_size,
                config.selection_antithetic,
                config.selection_visibility_spatial_sigma,
                config.selection_visibility_temporal_blend,
                config.selection_visibility_median_blend,
                config.selection_psnr_weight,
                config.selection_initialization,
            )
            if config.selection_field == "mean":
                generated_ssim = sum(generated_field_ssim.values()) / len(generated_field_ssim)
                generated_psnr = sum(generated_field_psnr.values()) / len(generated_field_psnr)
            else:
                generated_ssim = generated_field_ssim[config.selection_field]
                generated_psnr = generated_field_psnr[config.selection_field]
            selection_record = {
                "step": step,
                "generated_val_ssim": generated_ssim,
                "generated_val_psnr": generated_psnr,
                "generated_val_selection_score": generated_score,
                "selection_field": config.selection_field,
                "selection_start_timestep": config.selection_start_timestep,
                "selection_ensemble_size": config.selection_ensemble_size,
                "selection_antithetic": config.selection_antithetic,
                "selection_visibility_spatial_sigma": config.selection_visibility_spatial_sigma,
                "selection_visibility_temporal_blend": config.selection_visibility_temporal_blend,
                "selection_visibility_median_blend": config.selection_visibility_median_blend,
                "selection_psnr_weight": config.selection_psnr_weight,
                "selection_initialization": config.selection_initialization,
                "generated_val_temperature_ssim": generated_field_ssim["temperature"],
                "generated_val_visibility_ssim": generated_field_ssim["visibility"],
                "generated_val_temperature_psnr": generated_field_psnr["temperature"],
                "generated_val_visibility_psnr": generated_field_psnr["visibility"],
                "elapsed_seconds": time.time() - start_time,
            }
            print(json.dumps(selection_record), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(selection_record) + "\n")
            if generated_score > best_generated_ssim:
                best_generated_ssim = generated_score
                save_checkpoint(
                    output / "best_ssim.pt", model, ema_model, optimizer, step, config, store,
                    best_val, best_generated_ssim,
                )
        if step % config.save_every == 0 or step == config.train_steps:
            save_checkpoint(
                output / "latest.pt", model, ema_model, optimizer, step, config, store,
                best_val, best_generated_ssim,
            )

    manifest = {
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_sizes": {"train": len(train_index), "validation": len(val_index), "test": len(test_index)},
        "normalization": {"minimum": store.minimum.tolist(), "maximum": store.maximum.tolist()},
        "hashes": store.hashes,
        "duplicate_warning": {
            "A1_A3_temperature_identical": store.hashes["A1"]["temperature"] == store.hashes["A3"]["temperature"],
            "A1_A3_visibility_identical": store.hashes["A1"]["visibility"] == store.hashes["A3"]["visibility"],
        },
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output / ("best_ssim.pt" if config.select_by_ssim else "best.pt")


def load_for_evaluation(checkpoint: str, device: torch.device) -> Tuple[FireWaveModel, Config]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = Config(**state["config"])
    config.checkpoint_step = int(state["step"])
    history_condition_channels = config.history_frames * 2 * 4
    condition_channels = (
        history_condition_channels
        + 4
        + (3 + 2 * config.temporal_fourier_features if config.coordinate_conditioning else 0)
        + (8 if config.scenario_onehot_conditioning else 0)
    )
    physical_fields = 1 if config.visibility_only_model else 2
    target_channels = physical_fields * (4 if config.spatial_wavelet_only else 8)
    denoiser = FITWaveletUNet(
        target_channels=target_channels,
        condition_channels=condition_channels,
        dim=config.model_dim,
        anisotropic=config.anisotropic,
        dropout=config.dropout,
        global_conditioning=config.global_conditioning,
        coordinate_conditioning=config.coordinate_conditioning,
        scalar_condition_start=history_condition_channels,
    )
    model = FireWaveModel(
        denoiser,
        config.diffusion_steps,
        config.predict_residual,
        config.wavelet_type,
        config.aux_lowfreq_weight,
        config.aux_gradient_weight,
        config.aux_ssim_weight,
        config.aux_max_timestep,
        config.aux_gate_rescale,
        config.visibility_noise_weight,
        config.visibility_aux_weight,
        config.prediction_objective,
        config.coordinate_conditioning,
        aux_mse_weight=config.aux_mse_weight,
        aux_early_mse_weight=config.aux_early_mse_weight,
        aux_early_frames=config.aux_early_frames,
        conditional_mean=config.conditional_mean,
        mean_only_training=config.mean_only_training,
        mean_loss_weight=config.mean_loss_weight,
        mean_log_mse_weight=config.mean_log_mse_weight,
        mean_ssim_weight=config.mean_ssim_weight,
        mean_gradient_weight=config.mean_gradient_weight,
        mean_temporal_gradient_weight=config.mean_temporal_gradient_weight,
        mean_horizon_weight_end=config.mean_horizon_weight_end,
        spatial_wavelet_only=config.spatial_wavelet_only,
        temporal_fourier_features=config.temporal_fourier_features,
        scenario_onehot_conditioning=config.scenario_onehot_conditioning,
        pixel_refiner=config.pixel_refiner,
        temporal_query_refiner=config.temporal_query_refiner,
        visibility_only_model=config.visibility_only_model,
        refiner_dim=config.refiner_dim,
        history_frames=config.history_frames,
        trajectory_frames=config.trajectory_frames,
    ).to(device)
    model.load_state_dict(state["ema"])
    model.eval()
    return model, config


class CheckpointEnsemble(nn.Module):
    """Average compatible FireWave checkpoints for each shared diffusion latent."""

    def __init__(self, models: Sequence[FireWaveModel], weights: Sequence[float] | None = None):
        super().__init__()
        if not models:
            raise ValueError("CheckpointEnsemble requires at least one model")
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0] * len(models)
        if len(weights) != len(models) or any(weight < 0.0 for weight in weights):
            raise ValueError("Checkpoint weights must be non-negative and match checkpoints")
        total = float(sum(weights))
        if total <= 0.0:
            raise ValueError("At least one checkpoint weight must be positive")
        self.weights = [float(weight) / total for weight in weights]

    def make_initial_noise(self, history_01: torch.Tensor, frames: int) -> torch.Tensor:
        return self.models[0].make_initial_noise(history_01, frames)

    @torch.no_grad()
    def sample(self, *args, initialization: str = "random", initial_noise=None, **kwargs):
        history_01 = args[0]
        frames = kwargs.get("frames", args[2] if len(args) > 2 else 32)
        shared_noise = initial_noise
        if shared_noise is None and initialization == "random":
            shared_noise = self.make_initial_noise(history_01, frames)
        predictions = [
            model.sample(
                *args, initialization=initialization, initial_noise=shared_noise, **kwargs
            )
            for model in self.models
        ]
        return sum(
            weight * prediction
            for weight, prediction in zip(self.weights, predictions)
        )


@torch.no_grad()
def evaluate_random_windows(
    model: FireWaveModel,
    store: FITStore,
    config: Config,
    device: torch.device,
    ensemble_size: int = 1,
    initialization: str = "random",
    antithetic: bool = False,
    start_timestep: int = -1,
    start_timesteps: Sequence[int] | None = None,
    evaluation_split: str = "test",
    visibility_spatial_sigma: float = 0.0,
    visibility_temporal_blend: float = 0.0,
    visibility_scale: float = 1.0,
    visibility_scale_start: float | None = None,
    visibility_scale_end: float | None = None,
    visibility_bias: float = 0.0,
    visibility_gamma: float = 1.0,
    visibility_median_blend: float = 0.0,
    save_arrays_path: str | None = None,
    mean_blend: float = 0.0,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    segment_index = make_segment_index(
        history_frames=config.history_frames,
        include_full_span=config.include_full_span,
        full_span_repeats=config.full_span_repeats,
    )
    train_index, val_index, test_index = split_segment_index(segment_index, config.split_seed)
    split_lookup = {"train": train_index, "validation": val_index, "test": test_index}
    if evaluation_split not in split_lookup:
        raise ValueError(f"Unsupported evaluation split: {evaluation_split}")
    scenario_ids = set(selected_scenario_ids(config))
    selected_index = [
        sample for sample in split_lookup[evaluation_split]
        if sample[0] in scenario_ids
    ]
    short_horizon_test = [sample for sample in selected_index if sample[2] == config.table2_span]
    evaluation_index = short_horizon_test if short_horizon_test else selected_index
    evaluation_index = evaluation_index[: min(config.eval_samples, len(evaluation_index))]
    dataset = FITTrajectoryDataset(
        store, evaluation_index, config.trajectory_frames, config.history_frames
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    metrics = MetricAccumulator()
    persistence_metrics = MetricAccumulator()
    saved_predictions: List[np.ndarray] = []
    saved_targets: List[np.ndarray] = []
    saved_scenarios: List[np.ndarray] = []
    saved_starts: List[np.ndarray] = []
    for batch in loader:
        trajectory = batch[0].to(device)
        history = batch[1].to(device)
        condition = batch[2].to(device)
        predictions = sample_ensemble(
            model, history, condition, config.trajectory_frames, config.ddim_steps,
            ensemble_size, initialization, antithetic, start_timestep, start_timesteps,
            mean_blend,
        )
        prediction = torch.stack(predictions).mean(0)
        prediction = postprocess_visibility(
            prediction, visibility_spatial_sigma, visibility_temporal_blend,
            visibility_scale, visibility_scale_start, visibility_scale_end,
            visibility_bias, visibility_gamma,
            visibility_median_blend,
        )
        metrics.update(prediction, trajectory, exclude_first=False)
        persistence = history[:, -1:].expand_as(trajectory)
        persistence_metrics.update(persistence, trajectory, exclude_first=False)
        if save_arrays_path:
            saved_predictions.append(prediction.detach().cpu().numpy())
            saved_targets.append(trajectory.detach().cpu().numpy())
            saved_scenarios.append(batch[3].detach().cpu().numpy())
            saved_starts.append(batch[4].detach().cpu().numpy())
    if save_arrays_path:
        arrays_path = Path(save_arrays_path)
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_path,
            prediction=np.concatenate(saved_predictions, axis=0),
            target=np.concatenate(saved_targets, axis=0),
            scenario=np.concatenate(saved_scenarios, axis=0),
            start=np.concatenate(saved_starts, axis=0),
        )
    return metrics.result(), persistence_metrics.result()


def interpolate_prediction(
    prediction: torch.Tensor, seconds: float, start_seconds: float, end_seconds: float
) -> torch.Tensor:
    position = (seconds - start_seconds) / (end_seconds - start_seconds) * (prediction.shape[1] - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, prediction.shape[1] - 1)
    weight = position - lower
    return prediction[:, lower] * (1.0 - weight) + prediction[:, upper] * weight


@torch.no_grad()
def sample_ensemble(
    model: FireWaveModel,
    history: torch.Tensor,
    condition: torch.Tensor,
    frames: int,
    sampling_steps: int,
    ensemble_size: int,
    initialization: str = "random",
    antithetic: bool = False,
    start_timestep: int = -1,
    start_timesteps: Sequence[int] | None = None,
    mean_blend: float = 0.0,
) -> List[torch.Tensor]:
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive")
    if antithetic and initialization != "random":
        raise ValueError("Antithetic sampling requires random initialization")
    starts = list(start_timesteps) if start_timesteps is not None else [start_timestep]
    if not starts:
        raise ValueError("start_timesteps must contain at least one timestep")
    if not antithetic:
        return [
            model.sample(
                history, condition, frames, sampling_steps,
                eta=0.0, initialization=initialization,
                start_timestep=starts[index % len(starts)],
                mean_blend=mean_blend,
            )
            for index in range(ensemble_size)
        ]
    predictions: List[torch.Tensor] = []
    while len(predictions) < ensemble_size:
        pair_start = starts[(len(predictions) // 2) % len(starts)]
        latent = model.make_initial_noise(history, frames)
        predictions.append(
            model.sample(
                history, condition, frames, sampling_steps,
                eta=0.0, initialization="random", initial_noise=latent,
                start_timestep=pair_start,
                mean_blend=mean_blend,
            )
        )
        if len(predictions) < ensemble_size:
            predictions.append(
                model.sample(
                    history, condition, frames, sampling_steps,
                    eta=0.0, initialization="random", initial_noise=-latent,
                    start_timestep=pair_start,
                    mean_blend=mean_blend,
                )
            )
    return predictions


@torch.no_grad()
def evaluate_long_horizon(
    model: FireWaveModel,
    store: FITStore,
    config: Config,
    device: torch.device,
    ensemble_size: int = 1,
    initialization: str = "random",
    antithetic: bool = False,
    start_timestep: int = -1,
    start_timesteps: Sequence[int] | None = None,
    visibility_spatial_sigma: float = 0.0,
    visibility_temporal_blend: float = 0.0,
    visibility_scale: float = 1.0,
    visibility_scale_start: float | None = None,
    visibility_scale_end: float | None = None,
    visibility_bias: float = 0.0,
    visibility_gamma: float = 1.0,
    visibility_median_blend: float = 0.0,
    mean_blend: float = 0.0,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    history_indices = np.arange(config.history_frames, dtype=np.int64)
    forecast_start = config.history_frames
    indices = np.rint(np.linspace(forecast_start, 1000, config.trajectory_frames)).astype(np.int64)
    scenario_ids = selected_scenario_ids(config)
    history = store.normalize(store.data[list(scenario_ids)][:, history_indices])
    trajectory = store.normalize(store.data[list(scenario_ids)][:, indices])
    history_tensor = torch.from_numpy(np.array(history, dtype=np.float32, copy=True)).to(device)
    trajectory_tensor = torch.from_numpy(np.array(trajectory, dtype=np.float32, copy=True)).to(device)
    conditions = []
    for scenario_id in scenario_ids:
        scenario = SCENARIOS[scenario_id]
        conditions.append([
            math.log(HRR_MW[scenario] / 5.0) / math.log(10.0),
            (VENT_MS[scenario] - 2.0) / 2.0,
            0.0,
            (1000 - forecast_start) / 1000.0,
        ])
    condition_tensor = torch.tensor(conditions, dtype=torch.float32, device=device)
    predictions = sample_ensemble(
        model, history_tensor, condition_tensor, config.trajectory_frames,
        config.ddim_steps, ensemble_size, initialization, antithetic, start_timestep,
        start_timesteps, mean_blend,
    )
    prediction = torch.stack(predictions).mean(0)
    prediction = postprocess_visibility(
        prediction, visibility_spatial_sigma, visibility_temporal_blend,
        visibility_scale, visibility_scale_start, visibility_scale_end,
        visibility_bias, visibility_gamma,
        visibility_median_blend,
    )

    result: Dict[str, Dict[str, Dict[str, float]]] = {name: {} for name in FIELD_NAMES}
    persistence_result: Dict[str, Dict[str, Dict[str, float]]] = {name: {} for name in FIELD_NAMES}
    for seconds in (100, 200, 300, 400):
        pred_frame = interpolate_prediction(
            prediction, float(seconds), forecast_start * 0.4, 400.0
        )
        true_frame = torch.from_numpy(
            np.array(store.normalize(store.data[list(scenario_ids), int(seconds / 0.4) : int(seconds / 0.4) + 1])[:, 0],
                     dtype=np.float32, copy=True)
        ).to(device)
        for channel, name in enumerate(FIELD_NAMES):
            pred = pred_frame[:, channel]
            truth = true_frame[:, channel]
            raw_mse = F.mse_loss(pred.float(), truth.float()).item()
            result[name][str(seconds)] = {
                "mse_raw": raw_mse,
                "mse_x100": raw_mse * 100.0,
                "ssim": ssim_per_image(pred.float(), truth.float()).mean().item(),
            }
            persistence = history_tensor[:, -1, channel]
            persistence_mse = F.mse_loss(persistence.float(), truth.float()).item()
            persistence_result[name][str(seconds)] = {
                "mse_raw": persistence_mse,
                "mse_x100": persistence_mse * 100.0,
                "ssim": ssim_per_image(persistence.float(), truth.float()).mean().item(),
            }
    return result, persistence_result


def evaluate(
    checkpoint: str | Sequence[str],
    output_path: str | None = None,
    eval_samples: int | None = None,
    ddim_steps: int | None = None,
    ensemble_size: int = 1,
    initialization: str = "random",
    antithetic: bool = False,
    start_timestep: int = -1,
    start_timesteps: Sequence[int] | None = None,
    evaluation_split: str = "test",
    visibility_spatial_sigma: float = 0.0,
    visibility_temporal_blend: float = 0.0,
    visibility_scale: float = 1.0,
    visibility_scale_start: float | None = None,
    visibility_scale_end: float | None = None,
    visibility_bias: float = 0.0,
    visibility_gamma: float = 1.0,
    visibility_median_blend: float = 0.0,
    save_table2_arrays: str | None = None,
    mean_blend: float = 0.0,
    checkpoint_weights: Sequence[float] | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    loaded = [load_for_evaluation(path, device) for path in checkpoints]
    model, config = loaded[0]
    compatibility_keys = (
        "history_frames", "anisotropic", "predict_residual", "wavelet_type",
        "trajectory_frames", "diffusion_steps", "model_dim", "coordinate_conditioning",
        "conditional_mean",
        "global_conditioning",
        "temporal_fourier_features",
        "scenario_onehot_conditioning",
        "pixel_refiner",
        "temporal_query_refiner",
        "visibility_only_model",
        "refiner_dim",
    )
    for _, candidate_config in loaded[1:]:
        mismatched = [
            key for key in compatibility_keys
            if getattr(candidate_config, key) != getattr(config, key)
        ]
        if mismatched:
            raise ValueError(f"Incompatible checkpoint ensemble fields: {mismatched}")
    if checkpoint_weights is not None and len(checkpoint_weights) != len(loaded):
        raise ValueError("checkpoint weights must match the number of checkpoints")
    if len(loaded) > 1:
        model = CheckpointEnsemble(
            [item[0] for item in loaded], checkpoint_weights
        ).to(device).eval()
    if eval_samples is not None:
        config.eval_samples = eval_samples
    if ddim_steps is not None:
        config.ddim_steps = ddim_steps
    seed_everything(config.seed)
    store = FITStore(config.data_root)
    segment_index = make_segment_index(
        history_frames=config.history_frames,
        include_full_span=config.include_full_span,
        full_span_repeats=config.full_span_repeats,
    )
    random_windows, random_windows_persistence = evaluate_random_windows(
        model, store, config, device, ensemble_size, initialization, antithetic,
        start_timestep, start_timesteps, evaluation_split, visibility_spatial_sigma,
        visibility_temporal_blend, visibility_scale, visibility_scale_start,
        visibility_scale_end, visibility_bias,
        visibility_gamma, visibility_median_blend,
        save_table2_arrays, mean_blend,
    )
    long_horizon, long_horizon_persistence = evaluate_long_horizon(
        model, store, config, device, ensemble_size, initialization, antithetic,
        start_timestep, start_timesteps, visibility_spatial_sigma, visibility_temporal_blend,
        visibility_scale, visibility_scale_start, visibility_scale_end,
        visibility_bias, visibility_gamma, visibility_median_blend,
        mean_blend,
    )
    result = {
        "checkpoint": [str(Path(path).resolve()) for path in checkpoints],
        "protocol": {
            "model_variant": "residual trajectory" if config.predict_residual else "absolute trajectory",
            "history_frames": config.history_frames,
            "anisotropic": config.anisotropic,
            "include_full_span": config.include_full_span,
            "wavelet_type": config.wavelet_type,
            "auxiliary_weights": {
                "low_frequency": config.aux_lowfreq_weight,
                "gradient": config.aux_gradient_weight,
                "ssim": config.aux_ssim_weight,
                "mse": config.aux_mse_weight,
                "early_mse": config.aux_early_mse_weight,
                "early_frames": config.aux_early_frames,
            },
            "auxiliary_max_timestep": config.aux_max_timestep,
            "auxiliary_gate_rescale": config.aux_gate_rescale,
            "visibility_noise_weight": config.visibility_noise_weight,
            "visibility_aux_weight": config.visibility_aux_weight,
            "prediction_objective": config.prediction_objective,
            "coordinate_conditioning": config.coordinate_conditioning,
            "conditional_mean": config.conditional_mean,
            "mean_only_training": config.mean_only_training,
            "mean_loss_weight": config.mean_loss_weight,
            "mean_log_mse_weight": config.mean_log_mse_weight,
            "mean_ssim_weight": config.mean_ssim_weight,
            "mean_gradient_weight": config.mean_gradient_weight,
            "mean_temporal_gradient_weight": config.mean_temporal_gradient_weight,
            "mean_horizon_weight_end": config.mean_horizon_weight_end,
            "mean_learning_rate": config.mean_learning_rate,
            "checkpoint_selection": "generated validation SSIM" if config.select_by_ssim else "fixed final step",
            "checkpoint_selection_field": config.selection_field,
            "checkpoint_selection_start_timestep": config.selection_start_timestep,
            "checkpoint_selection_ensemble_size": config.selection_ensemble_size,
            "checkpoint_selection_antithetic": config.selection_antithetic,
            "checkpoint_selection_visibility_spatial_sigma": config.selection_visibility_spatial_sigma,
            "checkpoint_selection_visibility_temporal_blend": config.selection_visibility_temporal_blend,
            "checkpoint_selection_visibility_median_blend": config.selection_visibility_median_blend,
            "checkpoint_selection_psnr_weight": config.selection_psnr_weight,
            "checkpoint_selection_initialization": config.selection_initialization,
            "train_short_only": config.train_short_only,
            "train_start_stride": config.train_start_stride,
            "strict_frame_disjoint_train": config.strict_frame_disjoint_train,
            "strict_protected_min_span": config.strict_protected_min_span,
            "strict_dense_variable_train": config.strict_dense_variable_train,
            "selection_min_span": config.selection_min_span,
            "scenario_ids": list(selected_scenario_ids(config)),
            "scenarios": [SCENARIOS[index] for index in selected_scenario_ids(config)],
            "dropout": config.dropout,
            "global_conditioning": config.global_conditioning,
            "spatial_wavelet_only": config.spatial_wavelet_only,
            "temporal_fourier_features": config.temporal_fourier_features,
            "scenario_onehot_conditioning": config.scenario_onehot_conditioning,
            "pixel_refiner": config.pixel_refiner,
            "temporal_query_refiner": config.temporal_query_refiner,
            "visibility_only_model": config.visibility_only_model,
            "refiner_only_training": config.refiner_only_training,
            "refiner_dim": config.refiner_dim,
            "refiner_learning_rate": config.refiner_learning_rate,
            "weight_decay": config.weight_decay,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_ensemble_size": len(checkpoints),
            "checkpoint_ensemble_weights": (
                list(checkpoint_weights) if checkpoint_weights is not None else None
            ),
            "conditional_mean_blend": mean_blend,
            "trajectory_frames": config.trajectory_frames,
            "random_window_split": "8:1:1 with seed 1",
            "evaluation_split": evaluation_split,
            "visibility_postprocessing": {
                "spatial_sigma": visibility_spatial_sigma,
                "temporal_blend": visibility_temporal_blend,
                "scale": visibility_scale,
                "scale_start": visibility_scale_start,
                "scale_end": visibility_scale_end,
                "bias": visibility_bias,
                "gamma": visibility_gamma,
                "median_blend": visibility_median_blend,
            },
            "saved_table2_arrays": save_table2_arrays,
            "table2_test_windows": min(
                config.eval_samples,
                len([
                    sample for sample in {
                        "train": split_segment_index(segment_index, config.split_seed)[0],
                        "validation": split_segment_index(segment_index, config.split_seed)[1],
                        "test": split_segment_index(segment_index, config.split_seed)[2],
                    }[evaluation_split]
                    if (
                        sample[2] == config.table2_span
                        and sample[0] in set(selected_scenario_ids(config))
                    )
                ]),
            ),
            "table2_span_steps": config.table2_span,
            "long_horizon": (
                f"{config.history_frames} contiguous history frames followed by 32-frame direct generation "
                "to 400 s; interpolation at 100/200/300 s"
            ),
            "normalization": "per-field global min-max to [0,1]",
            "reported_mse": "raw normalized MSE is primary; 100 * MSE is retained as a scale diagnostic",
            "ssim": "7x7 Gaussian local-window SSIM, data_range=1",
            "psnr": "mean per-frame PSNR, data_range=1",
            "ddim_steps": config.ddim_steps,
            "ddim_eta": 0.0,
            "ensemble_size": ensemble_size,
            "diffusion_initialization": initialization,
            "antithetic_ensemble": antithetic,
            "sampling_start_timestep": (
                config.diffusion_steps - 1 if start_timestep < 0 else start_timestep
            ),
            "sampling_start_timesteps": list(start_timesteps) if start_timesteps else None,
        },
        "table2_firewave": random_windows,
        "table3_4_firewave": long_horizon,
        "persistence_baseline": {
            "table2": random_windows_persistence,
            "table3_4": long_horizon_persistence,
        },
        "data_integrity": {
            "A1_A3_temperature_identical": store.hashes["A1"]["temperature"] == store.hashes["A3"]["temperature"],
            "A1_A3_visibility_identical": store.hashes["A1"]["visibility"] == store.hashes["A3"]["visibility"],
        },
    }
    path = Path(output_path) if output_path else Path(config.output_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return path


def self_test() -> None:
    seed_everything(0)
    original = torch.randn(2, 2, 32, 23, 480)
    coefficients = haar3d(original)
    reconstructed = ihaar3d(coefficients, 2, (32, 23, 480))
    error = (original - reconstructed).abs().max().item()
    if error > 1e-5:
        raise AssertionError(f"Haar reconstruction error is too large: {error}")
    spatial_coefficients = spatial_haar3d(original)
    spatial_reconstructed = inverse_spatial_haar3d(
        spatial_coefficients, 2, (32, 23, 480)
    )
    spatial_error = (original - spatial_reconstructed).abs().max().item()
    if spatial_error > 1e-5:
        raise AssertionError(
            f"Spatial Haar reconstruction error is too large: {spatial_error}"
        )
    trajectory = torch.rand(1, 32, 2, 23, 480)
    history = torch.rand(1, 16, 2, 23, 480)
    condition_values = torch.rand(1, 4)
    history_condition_channels = 16 * 2 * 4
    denoiser = FITWaveletUNet(
        condition_channels=history_condition_channels + 4 + 3 + 8 + 8,
        dim=8,
        anisotropic=True,
        global_conditioning=True,
        coordinate_conditioning=True,
        scalar_condition_start=history_condition_channels,
    )
    model = FireWaveModel(
        denoiser,
        diffusion_steps=20,
        predict_residual=True,
        aux_early_mse_weight=0.1,
        aux_early_frames=8,
        conditional_mean=True,
        mean_loss_weight=1.0,
        mean_log_mse_weight=0.001,
        mean_ssim_weight=0.01,
        mean_gradient_weight=0.01,
        mean_temporal_gradient_weight=0.01,
        coordinate_conditioning=True,
        temporal_fourier_features=4,
        scenario_onehot_conditioning=True,
        pixel_refiner=True,
        refiner_dim=8,
        history_frames=16,
        trajectory_frames=32,
    )
    loss = model.loss(trajectory, history, condition_values)
    loss.backward()
    mean_prediction = model.sample(
        history, condition_values, frames=32, initialization="mean"
    )
    print(json.dumps({
        "haar_shape": list(coefficients.shape),
        "haar_max_reconstruction_error": error,
        "spatial_haar_shape": list(spatial_coefficients.shape),
        "spatial_haar_max_reconstruction_error": spatial_error,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "one_step_loss": float(loss.item()),
        "mean_prediction_shape": list(mean_prediction.shape),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("self-test")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-root", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--steps", type=int, default=20000)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--model-dim", type=int, default=32)
    train_parser.add_argument("--ddim-steps", type=int, default=100)
    train_parser.add_argument("--eval-samples", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--split-seed", type=int, default=1)
    train_parser.add_argument("--predict-residual", action="store_true")
    train_parser.add_argument("--history-frames", type=int, default=1)
    train_parser.add_argument("--anisotropic", action="store_true")
    train_parser.add_argument("--include-full-span", action="store_true")
    train_parser.add_argument("--full-span-repeats", type=int, default=16)
    train_parser.add_argument(
        "--wavelet-type", choices=("haar", "bior1.3", "pixel-unshuffle"), default="haar"
    )
    train_parser.add_argument("--aux-lowfreq-weight", type=float, default=0.0)
    train_parser.add_argument("--aux-gradient-weight", type=float, default=0.0)
    train_parser.add_argument("--aux-ssim-weight", type=float, default=0.0)
    train_parser.add_argument("--aux-mse-weight", type=float, default=0.0)
    train_parser.add_argument("--aux-early-mse-weight", type=float, default=0.0)
    train_parser.add_argument("--aux-early-frames", type=int, default=8)
    train_parser.add_argument("--conditional-mean", action="store_true")
    train_parser.add_argument("--mean-only-training", action="store_true")
    train_parser.add_argument("--mean-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--mean-log-mse-weight", type=float, default=0.0)
    train_parser.add_argument("--mean-ssim-weight", type=float, default=0.0)
    train_parser.add_argument("--mean-gradient-weight", type=float, default=0.0)
    train_parser.add_argument("--mean-temporal-gradient-weight", type=float, default=0.0)
    train_parser.add_argument("--mean-horizon-weight-end", type=float, default=1.0)
    train_parser.add_argument("--mean-learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--aux-max-timestep", type=int, default=-1)
    train_parser.add_argument(
        "--aux-gate-rescale", action=argparse.BooleanOptionalAction, default=True
    )
    train_parser.add_argument("--visibility-noise-weight", type=float, default=1.0)
    train_parser.add_argument("--visibility-aux-weight", type=float, default=1.0)
    train_parser.add_argument("--reset-optimizer", action="store_true")
    train_parser.add_argument(
        "--prediction-objective", choices=("epsilon", "x0", "direct"), default="epsilon"
    )
    train_parser.add_argument("--coordinate-conditioning", action="store_true")
    train_parser.add_argument("--allow-partial-resume", action="store_true")
    train_parser.add_argument("--select-by-ssim", action="store_true")
    train_parser.add_argument("--selection-every", type=int, default=5000)
    train_parser.add_argument("--selection-samples", type=int, default=4)
    train_parser.add_argument("--selection-ddim-steps", type=int, default=20)
    train_parser.add_argument(
        "--selection-field", choices=("mean", "temperature", "visibility"), default="mean"
    )
    train_parser.add_argument("--selection-start-timestep", type=int, default=-1)
    train_parser.add_argument("--selection-ensemble-size", type=int, default=1)
    train_parser.add_argument("--selection-antithetic", action="store_true")
    train_parser.add_argument("--selection-visibility-spatial-sigma", type=float, default=0.0)
    train_parser.add_argument("--selection-visibility-temporal-blend", type=float, default=0.0)
    train_parser.add_argument("--selection-visibility-median-blend", type=float, default=0.0)
    train_parser.add_argument("--selection-psnr-weight", type=float, default=0.0)
    train_parser.add_argument(
        "--selection-initialization", choices=("random", "zero", "mean"), default="random"
    )
    train_parser.add_argument("--train-short-only", action="store_true")
    train_parser.add_argument("--train-start-stride", type=int, default=32)
    train_parser.add_argument("--strict-frame-disjoint-train", action="store_true")
    train_parser.add_argument("--strict-protected-min-span", type=int, default=0)
    train_parser.add_argument("--strict-dense-variable-train", action="store_true")
    train_parser.add_argument("--strict-train-min-span", type=int, default=0)
    train_parser.add_argument(
        "--strict-balance-scenario-span", action="store_true"
    )
    train_parser.add_argument("--selection-min-span", type=int, default=0)
    train_parser.add_argument("--scenario-ids", type=int, nargs="+")
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--global-conditioning", action="store_true")
    train_parser.add_argument("--spatial-wavelet-only", action="store_true")
    train_parser.add_argument("--temporal-fourier-features", type=int, default=0)
    train_parser.add_argument("--scenario-onehot-conditioning", action="store_true")
    train_parser.add_argument("--pixel-refiner", action="store_true")
    train_parser.add_argument("--temporal-query-refiner", action="store_true")
    train_parser.add_argument("--visibility-only-model", action="store_true")
    train_parser.add_argument("--refiner-only-training", action="store_true")
    train_parser.add_argument("--refiner-dim", type=int, default=32)
    train_parser.add_argument("--refiner-learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--resume")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", required=True, nargs="+")
    eval_parser.add_argument("--output")
    eval_parser.add_argument("--eval-samples", type=int)
    eval_parser.add_argument("--ddim-steps", type=int)
    eval_parser.add_argument("--ensemble-size", type=int, default=1)
    eval_parser.add_argument(
        "--initialization", choices=("random", "zero", "mean"), default="random"
    )
    eval_parser.add_argument("--antithetic", action="store_true")
    eval_parser.add_argument("--sampling-start-timestep", type=int, default=-1)
    eval_parser.add_argument("--sampling-start-timesteps", type=int, nargs="+")
    eval_parser.add_argument(
        "--evaluation-split", choices=("train", "validation", "test"), default="test"
    )
    eval_parser.add_argument("--visibility-spatial-sigma", type=float, default=0.0)
    eval_parser.add_argument("--visibility-temporal-blend", type=float, default=0.0)
    eval_parser.add_argument("--visibility-scale", type=float, default=1.0)
    eval_parser.add_argument("--visibility-scale-start", type=float)
    eval_parser.add_argument("--visibility-scale-end", type=float)
    eval_parser.add_argument("--visibility-bias", type=float, default=0.0)
    eval_parser.add_argument("--visibility-gamma", type=float, default=1.0)
    eval_parser.add_argument("--visibility-median-blend", type=float, default=0.0)
    eval_parser.add_argument("--save-table2-arrays")
    eval_parser.add_argument("--mean-blend", type=float, default=0.0)
    eval_parser.add_argument("--checkpoint-weights", type=float, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "train":
        config = Config(
            data_root=args.data_root,
            output_dir=args.output_dir,
            train_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_dim=args.model_dim,
            ddim_steps=args.ddim_steps,
            eval_samples=args.eval_samples,
            seed=args.seed,
            split_seed=args.split_seed,
            predict_residual=args.predict_residual,
            history_frames=args.history_frames,
            anisotropic=args.anisotropic,
            include_full_span=args.include_full_span,
            full_span_repeats=args.full_span_repeats,
            wavelet_type=args.wavelet_type,
            aux_lowfreq_weight=args.aux_lowfreq_weight,
            aux_gradient_weight=args.aux_gradient_weight,
            aux_ssim_weight=args.aux_ssim_weight,
            aux_mse_weight=args.aux_mse_weight,
            aux_early_mse_weight=args.aux_early_mse_weight,
            aux_early_frames=args.aux_early_frames,
            conditional_mean=args.conditional_mean,
            mean_only_training=args.mean_only_training,
            mean_loss_weight=args.mean_loss_weight,
            mean_log_mse_weight=args.mean_log_mse_weight,
            mean_ssim_weight=args.mean_ssim_weight,
            mean_gradient_weight=args.mean_gradient_weight,
            mean_temporal_gradient_weight=args.mean_temporal_gradient_weight,
            mean_horizon_weight_end=args.mean_horizon_weight_end,
            mean_learning_rate=args.mean_learning_rate,
            aux_max_timestep=args.aux_max_timestep,
            aux_gate_rescale=args.aux_gate_rescale,
            visibility_noise_weight=args.visibility_noise_weight,
            visibility_aux_weight=args.visibility_aux_weight,
            reset_optimizer_on_resume=args.reset_optimizer,
            prediction_objective=args.prediction_objective,
            coordinate_conditioning=args.coordinate_conditioning,
            allow_partial_resume=args.allow_partial_resume,
            select_by_ssim=args.select_by_ssim,
            selection_every=args.selection_every,
            selection_samples=args.selection_samples,
            selection_ddim_steps=args.selection_ddim_steps,
            selection_field=args.selection_field,
            selection_start_timestep=args.selection_start_timestep,
            selection_ensemble_size=args.selection_ensemble_size,
            selection_antithetic=args.selection_antithetic,
            selection_visibility_spatial_sigma=args.selection_visibility_spatial_sigma,
            selection_visibility_temporal_blend=args.selection_visibility_temporal_blend,
            selection_visibility_median_blend=args.selection_visibility_median_blend,
            selection_psnr_weight=args.selection_psnr_weight,
            selection_initialization=args.selection_initialization,
            train_short_only=args.train_short_only,
            train_start_stride=args.train_start_stride,
            strict_frame_disjoint_train=args.strict_frame_disjoint_train,
            strict_protected_min_span=args.strict_protected_min_span,
            strict_dense_variable_train=args.strict_dense_variable_train,
            strict_train_min_span=args.strict_train_min_span,
            strict_balance_scenario_span=args.strict_balance_scenario_span,
            selection_min_span=args.selection_min_span,
            scenario_ids=tuple(args.scenario_ids) if args.scenario_ids else None,
            dropout=args.dropout,
            global_conditioning=args.global_conditioning,
            spatial_wavelet_only=args.spatial_wavelet_only,
            temporal_fourier_features=args.temporal_fourier_features,
            scenario_onehot_conditioning=args.scenario_onehot_conditioning,
            pixel_refiner=args.pixel_refiner,
            temporal_query_refiner=args.temporal_query_refiner,
            visibility_only_model=args.visibility_only_model,
            refiner_only_training=args.refiner_only_training,
            refiner_dim=args.refiner_dim,
            refiner_learning_rate=args.refiner_learning_rate,
            weight_decay=args.weight_decay,
        )
        checkpoint = train(config, args.resume)
        print(f"best_checkpoint={checkpoint}")
    elif args.command == "evaluate":
        evaluate(
            args.checkpoint, args.output, args.eval_samples, args.ddim_steps,
            args.ensemble_size, args.initialization, args.antithetic,
            args.sampling_start_timestep, args.sampling_start_timesteps,
            args.evaluation_split,
            args.visibility_spatial_sigma, args.visibility_temporal_blend,
            args.visibility_scale, args.visibility_scale_start,
            args.visibility_scale_end, args.visibility_bias,
            args.visibility_gamma,
            args.visibility_median_blend,
            args.save_table2_arrays,
            args.mean_blend,
            args.checkpoint_weights,
        )


if __name__ == "__main__":
    main()
