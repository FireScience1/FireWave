#!/usr/bin/env python3
"""Train a leakage-safe residual refiner on the strict physical visibility prior."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from .metrics import field_metrics
    from .postprocess import smooth
    from .scenario_prior import predict_encoded_trajectory
    from .direct_predictor import (
        Residual2D,
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from .core import (
        FITStore,
        HRR_MW,
        SCENARIOS,
        VENT_MS,
        make_segment_index,
        split_segment_index,
        ssim_per_image,
    )
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
        Residual2D,
        history_keys,
        strict_frame_disjoint_variable_training_rows,
        target_keys,
    )
    from firewave.core import (
        FITStore,
        HRR_MW,
        SCENARIOS,
        VENT_MS,
        make_segment_index,
        split_segment_index,
        ssim_per_image,
    )
    from firewave.visibility_encoding import (
        decode_visibility,
        encode_visibility,
        target_indices,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_scenario_sampling_multipliers(value: str | None) -> tuple[float, ...]:
    """Return one positive strict-training sampler weight per FIT scenario."""

    if value is None:
        return (1.0,) * 8
    multipliers = tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )
    if len(multipliers) != 8 or any(item <= 0.0 for item in multipliers):
        raise ValueError(
            "scenario sampling multipliers must contain eight positive values"
        )
    return multipliers


def haar_wavelet_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    levels: int,
) -> torch.Tensor:
    """Multilevel Haar-domain L1 loss for smoke fronts and boundaries."""

    current_prediction = prediction
    current_target = target
    losses = []
    for _ in range(levels):
        height = 2 * (current_prediction.shape[-2] // 2)
        width = 2 * (current_prediction.shape[-1] // 2)
        if height < 2 or width < 2:
            break
        pred = current_prediction[..., :height, :width]
        truth = current_target[..., :height, :width]

        def subbands(field: torch.Tensor):
            a = field[..., 0::2, 0::2]
            b = field[..., 0::2, 1::2]
            c = field[..., 1::2, 0::2]
            d = field[..., 1::2, 1::2]
            return (
                0.5 * (a + b + c + d),
                0.5 * (a - b + c - d),
                0.5 * (a + b - c - d),
                0.5 * (a - b - c + d),
            )

        pred_ll, pred_lh, pred_hl, pred_hh = subbands(pred)
        true_ll, true_lh, true_hl, true_hh = subbands(truth)
        losses.extend(
            (
                (pred_lh - true_lh).abs().mean(),
                (pred_hl - true_hl).abs().mean(),
                (pred_hh - true_hh).abs().mean(),
            )
        )
        current_prediction = pred_ll
        current_target = true_ll
    losses.append((current_prediction - current_target).abs().mean())
    return torch.stack(losses).mean()


class StrictPriorDataset(Dataset):
    def __init__(
        self,
        store: FITStore,
        rows,
        training_frame_keys: set[tuple[int, int]],
        training: bool,
        visibility_power: float = 16.0,
        history_mode: str = "compact",
        history_anchor_alpha: float = 0.0,
        history_anchor_tau: float = 32.0,
        history_anchor_trend_beta: float = 0.0,
        history_anchor_trend_lag: int = 8,
        training_prior_exclusion: str = "target",
        condition_time_fallback: bool = False,
        condition_fallback_distance: int = 64,
    ):
        self.store = store
        self.rows = list(rows)
        self.training = training
        self.visibility_power = visibility_power
        self.history_mode = history_mode
        self.history_anchor_alpha = history_anchor_alpha
        self.history_anchor_tau = history_anchor_tau
        self.history_anchor_trend_beta = history_anchor_trend_beta
        self.history_anchor_trend_lag = history_anchor_trend_lag
        self.training_prior_exclusion = training_prior_exclusion
        self.condition_time_fallback = condition_time_fallback
        self.condition_fallback_distance = condition_fallback_distance
        self.frame_times = {}
        self.encoded_bank = {}
        for scenario in range(8):
            times = np.asarray(
                sorted(
                    time
                    for key_scenario, time in training_frame_keys
                    if key_scenario == scenario
                ),
                dtype=np.int64,
            )
            if times.size:
                self.frame_times[scenario] = times
                encoded = encode_visibility(
                    store.data[scenario, times, 1], visibility_power
                )
                self.encoded_bank[scenario] = {
                    int(time): field for time, field in zip(times, encoded)
                }

    def __len__(self) -> int:
        return len(self.rows)

    def _prior(
        self,
        scenario: int,
        times: np.ndarray,
        excluded_times: np.ndarray | None = None,
    ) -> np.ndarray:
        frame_times = self.frame_times
        encoded_bank = self.encoded_bank
        if self.training and scenario in encoded_bank:
            held_times = set(
                int(value)
                for value in (
                    times if excluded_times is None else excluded_times
                ).tolist()
            )
            available = np.asarray(
                [value for value in frame_times[scenario] if int(value) not in held_times],
                dtype=np.int64,
            )
            if len(available) < 4:
                raise RuntimeError("Leave-target-out prior has too few source frames")
            local_times = dict(frame_times)
            local_times[scenario] = available
            local_bank = dict(encoded_bank)
            local_bank[scenario] = {
                int(value): encoded_bank[scenario][int(value)] for value in available
            }
        else:
            local_times = frame_times
            local_bank = encoded_bank
        encoded = predict_encoded_trajectory(
            scenario,
            times,
            local_bank,
            local_times,
            "pchip",
            1,
            None,
            self.condition_time_fallback,
            self.condition_fallback_distance,
        )
        physical = decode_visibility(encoded, self.visibility_power)
        return np.clip(
            (physical - float(self.store.minimum[1])) / float(self.store.scale[1]),
            0.0,
            1.0,
        ).astype(np.float32)

    def __getitem__(self, index: int):
        scenario, start, span = self.rows[index]
        times = target_indices(start, span, 32, 16)
        prior_excluded_times = times
        if self.training and self.training_prior_exclusion == "window":
            history_times = np.arange(start, start + 16, dtype=np.int64)
            prior_excluded_times = np.unique(
                np.concatenate((history_times, times))
            )
        prior = self._prior(
            int(scenario),
            times,
            excluded_times=prior_excluded_times,
        )
        target = self.store.normalize(
            self.store.data[scenario, times]
        ).astype(np.float32)[:, 1]
        history = self.store.normalize(
            self.store.data[scenario, start : start + 16]
        ).astype(np.float32)
        last_temperature = history[-1, 0]
        last_visibility = history[-1, 1]
        if self.history_anchor_alpha > 0.0:
            if self.history_anchor_trend_beta > 0.0:
                anchor_times = np.asarray(
                    [
                        int(start) + 15 - self.history_anchor_trend_lag,
                        int(start) + 15,
                    ],
                    dtype=np.int64,
                )
                calibration_excluded_times = np.unique(
                    np.concatenate(
                        (
                            np.arange(start, start + 16, dtype=np.int64),
                            times,
                        )
                    )
                )
                anchor_priors = self._prior(
                    int(scenario),
                    anchor_times,
                    excluded_times=calibration_excluded_times,
                )
                observed_anchors = np.stack(
                    (
                        history[-1 - self.history_anchor_trend_lag, 1],
                        last_visibility,
                    )
                ).astype(np.float32)
                anchor_residuals = observed_anchors - anchor_priors
                residual_trend = (
                    anchor_residuals[1] - anchor_residuals[0]
                ) / float(self.history_anchor_trend_lag)
                anchor_residual = anchor_residuals[1]
            else:
                anchor_prior = self._prior(
                    int(scenario),
                    np.asarray([int(start) + 15], dtype=np.int64),
                    excluded_times=prior_excluded_times,
                )[0]
                anchor_residual = last_visibility - anchor_prior
                residual_trend = np.zeros_like(anchor_residual)
            delta = times.astype(np.float32) - float(start + 15)
            decay = np.exp(-delta / self.history_anchor_tau).astype(np.float32)
            correction = (
                anchor_residual[None]
                + self.history_anchor_trend_beta
                * delta[:, None, None]
                * residual_trend[None]
            )
            prior = np.clip(
                prior
                + self.history_anchor_alpha
                * decay[:, None, None]
                * correction,
                0.0,
                1.0,
            ).astype(np.float32)
        if self.history_mode == "compact":
            history_features = np.stack(
                (
                    last_temperature,
                    last_visibility,
                    history[-1, 1] - history[-2, 1],
                )
            ).astype(np.float32)
        elif self.history_mode == "multiscale":
            features = [last_temperature, last_visibility]
            for field in (0, 1):
                features.extend(
                    history[-1, field] - history[-1 - lag, field]
                    for lag in (1, 4, 8, 15)
                )
            for field in (0, 1):
                features.extend(
                    history[-width:, field].mean(axis=0)
                    for width in (4, 8, 16)
                )
            history_features = np.stack(features).astype(np.float32)
        elif self.history_mode == "full":
            history_features = history.reshape(
                history.shape[0] * history.shape[1],
                history.shape[2],
                history.shape[3],
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown history mode: {self.history_mode}")
        scenario_name = SCENARIOS[int(scenario)]
        conditions = np.asarray(
            [
                math.log(HRR_MW[scenario_name] / 5.0) / math.log(10.0),
                (VENT_MS[scenario_name] - 2.0) / 2.0,
                start / 1000.0,
                span / 1000.0,
            ],
            dtype=np.float32,
        )
        absolute_time = (2.0 * times.astype(np.float32) / 1000.0 - 1.0)
        return (
            torch.from_numpy(prior),
            torch.from_numpy(target),
            torch.from_numpy(history_features),
            torch.from_numpy(conditions),
            torch.from_numpy(absolute_time),
            int(scenario),
            int(start),
        )


class DilatedResidual2D(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(field)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return field + hidden


class TemporalResidual3D(nn.Module):
    def __init__(self, channels: int, temporal_dilation: int = 1):
        super().__init__()
        groups = min(8, channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            3,
            padding=(temporal_dilation, 1, 1),
            dilation=(temporal_dilation, 1, 1),
            groups=channels,
        )
        self.pointwise = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(F.silu(self.norm(field)))
        return field + self.pointwise(hidden)


class LongitudinalResidual2D(nn.Module):
    """Anisotropic residual block aligned with the tunnel axis."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        groups = min(8, channels)
        padding = (1, kernel // 2)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, (3, kernel), padding=padding)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, kernel), padding=padding)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(field)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return field + hidden


class AxialSpectralResidual(nn.Module):
    """Global low-frequency mixing along the tunnel's longitudinal axis."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.weight_real = nn.Parameter(torch.ones(channels, modes))
        self.weight_imag = nn.Parameter(torch.zeros(channels, modes))
        self.projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm(field))
        spectrum = torch.fft.rfft(hidden, dim=-1, norm="ortho")
        mode_count = min(self.modes, spectrum.shape[-1])
        weight = torch.complex(
            self.weight_real[:, :mode_count],
            self.weight_imag[:, :mode_count],
        )
        filtered = torch.zeros_like(spectrum)
        filtered[..., :mode_count] = (
            spectrum[..., :mode_count] * weight[None, :, None]
        )
        mixed = torch.fft.irfft(
            filtered,
            n=field.shape[-1],
            dim=-1,
            norm="ortho",
        )
        return field + self.projection(mixed)


class PriorResidualRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        blocks: int,
        dilated: bool = False,
        temporal_blocks: int = 0,
        history_channels: int = 3,
        time_fourier: int = 0,
        residual_scale: float = 0.20,
        coordinate_channels: bool = False,
        longitudinal_kernel: int = 0,
        temporal_dilated: bool = False,
        relative_time_channel: bool = False,
        condition_film: bool = False,
        learnable_horizon_scale: bool = False,
        horizon_block_heads: int = 0,
        multiscale_output_head: bool = False,
        learned_prior_warp: bool = False,
        max_longitudinal_warp: float = 32.0,
        max_vertical_warp: float = 4.0,
        residual_block_scales: tuple[float, ...] | None = None,
        temporal_attention_heads: int = 0,
        axial_spectral_blocks: int = 0,
        axial_spectral_modes: int = 32,
    ):
        super().__init__()
        self.time_fourier = time_fourier
        self.residual_scale = residual_scale
        self.coordinate_channels = coordinate_channels
        self.relative_time_channel = relative_time_channel
        self.use_condition_film = condition_film
        self.use_learnable_horizon_scale = learnable_horizon_scale
        self.horizon_block_heads = horizon_block_heads
        self.use_multiscale_output_head = multiscale_output_head
        self.use_learned_prior_warp = learned_prior_warp
        self.max_longitudinal_warp = max_longitudinal_warp
        self.max_vertical_warp = max_vertical_warp
        self.residual_block_scales = residual_block_scales
        self.temporal_attention_heads = temporal_attention_heads
        self.axial_spectral_blocks_count = axial_spectral_blocks
        self.axial_spectral_modes = axial_spectral_modes
        if learnable_horizon_scale:
            self.horizon_log_scale = nn.Parameter(torch.zeros(32))
        input_channels = (
            1
            + history_channels
            + 3
            + 1
            + 2 * time_fourier
            + (2 if coordinate_channels else 0)
            + (1 if relative_time_channel else 0)
        )
        self.input = nn.Conv2d(input_channels, dim, 3, padding=1)
        if condition_film:
            self.condition_film = nn.Sequential(
                nn.Linear(5, dim),
                nn.SiLU(),
                nn.Linear(dim, 2 * dim),
            )
            nn.init.zeros_(self.condition_film[-1].weight)
            nn.init.zeros_(self.condition_film[-1].bias)
        if longitudinal_kernel > 0:
            self.blocks = nn.Sequential(
                *(
                    LongitudinalResidual2D(dim, longitudinal_kernel)
                    for _ in range(blocks)
                )
            )
        elif dilated:
            self.blocks = nn.Sequential(
                *(
                    DilatedResidual2D(dim, (1, 2, 4, 8)[index % 4])
                    for index in range(blocks)
                )
            )
        else:
            self.blocks = nn.Sequential(
                *(Residual2D(dim, dim) for _ in range(blocks))
            )
        self.axial_spectral_blocks = nn.Sequential(
            *(
                AxialSpectralResidual(dim, axial_spectral_modes)
                for _ in range(axial_spectral_blocks)
            )
        )
        self.temporal_blocks = nn.Sequential(
            *(
                TemporalResidual3D(
                    dim,
                    (1, 2, 4, 8)[index % 4] if temporal_dilated else 1,
                )
                for index in range(temporal_blocks)
            )
        )
        if temporal_attention_heads > 0:
            self.temporal_attention_norm = nn.LayerNorm(dim)
            self.temporal_attention = nn.MultiheadAttention(
                dim,
                temporal_attention_heads,
                batch_first=True,
            )
            nn.init.zeros_(self.temporal_attention.out_proj.weight)
            nn.init.zeros_(self.temporal_attention.out_proj.bias)
        if horizon_block_heads > 0:
            self.output_norm = nn.GroupNorm(min(8, dim), dim)
            self.output_heads = nn.ModuleList(
                nn.Conv2d(dim, 1, 1) for _ in range(horizon_block_heads)
            )
            for head in self.output_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            output_channels = 3 if learned_prior_warp else 1
            self.output = nn.Sequential(
                nn.GroupNorm(min(8, dim), dim),
                nn.SiLU(),
                nn.Conv2d(dim, output_channels, 1),
            )
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)
            if multiscale_output_head:
                self.low_output = nn.Sequential(
                    nn.GroupNorm(min(8, dim), dim),
                    nn.SiLU(),
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(dim, 1, 1),
                )
                nn.init.zeros_(self.low_output[-1].weight)
                nn.init.zeros_(self.low_output[-1].bias)

    def forward(
        self,
        prior: torch.Tensor,
        history_features: torch.Tensor,
        conditions: torch.Tensor,
        absolute_time: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, height, width = prior.shape
        history = history_features[:, None].expand(-1, frames, -1, -1, -1)
        hrr = conditions[:, None, 0:1, None, None].expand(-1, frames, -1, height, width)
        ventilation = conditions[:, None, 1:2, None, None].expand(
            -1, frames, -1, height, width
        )
        span = conditions[:, None, 3:4, None, None].expand(
            -1, frames, -1, height, width
        )
        time_fields = [
            absolute_time[:, :, None, None, None].expand(
                -1, -1, -1, height, width
            )
        ]
        for index in range(self.time_fourier):
            angle = math.pi * float(2**index) * absolute_time
            time_fields.extend(
                (
                    torch.sin(angle)[:, :, None, None, None].expand(
                        -1, -1, -1, height, width
                    ),
                    torch.cos(angle)[:, :, None, None, None].expand(
                        -1, -1, -1, height, width
                    ),
                )
            )
        if self.relative_time_channel:
            relative_time = torch.linspace(
                0.0,
                1.0,
                frames,
                device=prior.device,
                dtype=prior.dtype,
            )
            time_fields.append(
                relative_time[None, :, None, None, None].expand(
                    batch, -1, -1, height, width
                )
            )
        input_fields = [
            prior[:, :, None],
            history,
            hrr,
            ventilation,
            span,
            *time_fields,
        ]
        if self.coordinate_channels:
            y = torch.linspace(-1.0, 1.0, height, device=prior.device, dtype=prior.dtype)
            x = torch.linspace(-1.0, 1.0, width, device=prior.device, dtype=prior.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack((xx, yy))[None, None].expand(
                batch, frames, -1, -1, -1
            )
            input_fields.append(coordinates)
        inputs = torch.cat(tuple(input_fields), dim=2).flatten(0, 1)
        features = self.input(inputs)
        if self.use_condition_film:
            film_input = torch.cat(
                (
                    conditions[:, None, 0:4].expand(-1, frames, -1),
                    absolute_time[:, :, None],
                ),
                dim=-1,
            ).flatten(0, 1)
            film_scale, film_shift = self.condition_film(film_input).chunk(2, dim=-1)
            features = (
                features * (1.0 + film_scale[:, :, None, None])
                + film_shift[:, :, None, None]
            )
        features = self.blocks(features)
        if len(self.axial_spectral_blocks):
            features = self.axial_spectral_blocks(features)
        if len(self.temporal_blocks):
            features = features.reshape(
                batch, frames, -1, height, width
            ).permute(0, 2, 1, 3, 4)
            features = self.temporal_blocks(features)
            features = features.permute(0, 2, 1, 3, 4).flatten(0, 1)
        if self.temporal_attention_heads > 0:
            features_5d = features.reshape(batch, frames, -1, height, width)
            coarse = F.avg_pool2d(
                features_5d.flatten(0, 1),
                kernel_size=(2, 4),
                stride=(2, 4),
                ceil_mode=True,
            )
            coarse_height, coarse_width = coarse.shape[-2:]
            tokens = coarse.reshape(
                batch, frames, -1, coarse_height, coarse_width
            ).permute(0, 3, 4, 1, 2).reshape(
                batch * coarse_height * coarse_width,
                frames,
                -1,
            )
            normalized_tokens = self.temporal_attention_norm(tokens)
            attended_tokens, _ = self.temporal_attention(
                normalized_tokens,
                normalized_tokens,
                normalized_tokens,
                need_weights=False,
            )
            attended = attended_tokens.reshape(
                batch, coarse_height, coarse_width, frames, -1
            ).permute(0, 3, 4, 1, 2).flatten(0, 1)
            attended = F.interpolate(
                attended,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, frames, -1, height, width)
            features = (features_5d + attended).flatten(0, 1)
        if self.horizon_block_heads > 0:
            if frames % self.horizon_block_heads != 0:
                raise ValueError("forecast frames must divide horizon block heads")
            features = features.reshape(batch, frames, -1, height, width)
            block_size = frames // self.horizon_block_heads
            residual_blocks = []
            for block_index, head in enumerate(self.output_heads):
                begin = block_index * block_size
                end = begin + block_size
                block = features[:, begin:end].flatten(0, 1)
                block = head(F.silu(self.output_norm(block)))[:, 0]
                residual_blocks.append(
                    block.reshape(batch, block_size, height, width)
                )
            residual = torch.cat(residual_blocks, dim=1)
        else:
            output = self.output(features)
            residual = output[:, 0]
            flow = output[:, 1:3] if self.use_learned_prior_warp else None
            if self.use_multiscale_output_head:
                low = F.avg_pool2d(features, kernel_size=(2, 4), stride=(2, 4))
                low = self.low_output(low)
                low = F.interpolate(
                    low,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                residual = residual + low
            residual = residual.reshape(batch, frames, height, width)
        if self.use_learned_prior_warp:
            if self.horizon_block_heads > 0:
                raise ValueError(
                    "learned prior warp is incompatible with horizon block heads"
                )
            assert flow is not None
            flow = flow.reshape(batch, frames, 2, height, width)
            y = torch.linspace(
                -1.0, 1.0, height, device=prior.device, dtype=prior.dtype
            )
            x = torch.linspace(
                -1.0, 1.0, width, device=prior.device, dtype=prior.dtype
            )
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            base_grid = torch.stack((xx, yy), dim=-1)[None].expand(
                batch * frames, -1, -1, -1
            )
            flow_x = (
                2.0
                * self.max_longitudinal_warp
                / max(width - 1, 1)
                * torch.tanh(flow[:, :, 0])
            )
            flow_y = (
                2.0
                * self.max_vertical_warp
                / max(height - 1, 1)
                * torch.tanh(flow[:, :, 1])
            )
            sampling_grid = base_grid + torch.stack(
                (flow_x, flow_y), dim=-1
            ).reshape(batch * frames, height, width, 2)
            warped_prior = F.grid_sample(
                prior.reshape(batch * frames, 1, height, width),
                sampling_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[:, 0].reshape(batch, frames, height, width)
        else:
            warped_prior = prior
        if self.use_learnable_horizon_scale:
            horizon_scale = self.horizon_log_scale[:frames].clamp(-1.0, 1.0).exp()
            residual = residual * horizon_scale[None, :, None, None]
        if self.residual_block_scales is None:
            residual_scale = self.residual_scale
        else:
            if frames % len(self.residual_block_scales) != 0:
                raise ValueError("forecast frames must divide residual block scales")
            residual_scale = residual.new_tensor(
                self.residual_block_scales
            ).repeat_interleave(frames // len(self.residual_block_scales))
            residual_scale = residual_scale[None, :, None, None]
        return (warped_prior + residual_scale * torch.tanh(residual)).clamp(
            0.0, 1.0
        )


def evaluate(model, loader, device, save_arrays: str | None = None):
    model.eval()
    predictions = []
    targets = []
    scenarios = []
    starts = []
    with torch.no_grad():
        for prior, target, history, conditions, times, scenario, start in loader:
            prediction = model(
                prior.to(device),
                history.to(device),
                conditions.to(device),
                times.to(device),
            ).cpu()
            predictions.append(prediction.numpy())
            targets.append(target.numpy())
            scenarios.append(scenario.numpy())
            starts.append(start.numpy())
    prediction = np.concatenate(predictions).astype(np.float32)
    target = np.concatenate(targets).astype(np.float32)
    tensor = torch.from_numpy(prediction)
    processed = np.clip(
        smooth(tensor, 0.5, 0.05).numpy() * 1.05,
        0.0,
        1.0,
    ).astype(np.float32)
    metrics = field_metrics(processed, target)
    if save_arrays:
        np.savez_compressed(
            save_arrays,
            prediction=processed,
            target=target,
            scenario=np.concatenate(scenarios),
            start=np.concatenate(starts),
        )
    return metrics


def atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--initial-checkpoint",
        help=(
            "Optional strict-validation checkpoint used to initialize a fresh "
            "fine-tuning run. Only model weights and the completed step are "
            "restored; the optimizer is intentionally reinitialized."
        ),
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--cosine-lr",
        action="store_true",
        help="Cosine-decay the learning rate to 5% of its initial value.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=(
            "Optional exponential moving average of model weights used for "
            "complete-trajectory validation and checkpoint selection. A value "
            "of zero disables EMA."
        ),
    )
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--dilated", action="store_true")
    parser.add_argument("--temporal-blocks", type=int, default=0)
    parser.add_argument("--temporal-dilated", action="store_true")
    parser.add_argument("--temporal-attention-heads", type=int, default=0)
    parser.add_argument("--axial-spectral-blocks", type=int, default=0)
    parser.add_argument("--axial-spectral-modes", type=int, default=32)
    parser.add_argument("--mse-weight", type=float, default=0.2)
    parser.add_argument("--log-mse-weight", type=float, default=0.0)
    parser.add_argument(
        "--psnr-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the unweighted mean log per-frame MSE. This matches "
            "the optimizable part of mean per-frame PSNR with data_range=1."
        ),
    )
    parser.add_argument("--ssim-weight", type=float, default=3.0)
    parser.add_argument(
        "--metric-aligned-loss",
        action="store_true",
        help=(
            "Apply the fixed validation smoothing and 1.05 intensity scale "
            "inside the differentiable training objective. Evaluation remains "
            "unchanged; this only aligns the optimized prediction with the "
            "reported metric path."
        ),
    )
    parser.add_argument(
        "--smoke-mse-weight",
        type=float,
        default=0.0,
        help="Extra MSE focused on low-visibility smoke regions.",
    )
    parser.add_argument(
        "--gradient-weight",
        type=float,
        default=0.0,
        help="Spatial-gradient L1 weight for smoke-front alignment.",
    )
    parser.add_argument("--wavelet-weight", type=float, default=0.0)
    parser.add_argument("--wavelet-levels", type=int, default=2)
    parser.add_argument(
        "--coarse-ssim-weight",
        type=float,
        default=0.0,
        help=(
            "Optional large-scale structural loss. It averages SSIM after "
            "2x isotropic pooling and 4x longitudinal pooling."
        ),
    )
    parser.add_argument(
        "--history-mode",
        choices=("compact", "multiscale", "full"),
        default="compact",
    )
    parser.add_argument("--time-fourier", type=int, default=0)
    parser.add_argument("--relative-time-channel", action="store_true")
    parser.add_argument("--condition-film", action="store_true")
    parser.add_argument("--learnable-horizon-scale", action="store_true")
    parser.add_argument("--horizon-block-heads", type=int, default=0)
    parser.add_argument("--multiscale-output-head", action="store_true")
    parser.add_argument(
        "--learned-prior-warp",
        action="store_true",
        help=(
            "Predict a zero-initialized spatial flow that warps the physical "
            "prior before adding the bounded intensity residual."
        ),
    )
    parser.add_argument("--max-longitudinal-warp", type=float, default=32.0)
    parser.add_argument("--max-vertical-warp", type=float, default=4.0)
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument(
        "--residual-block-scales",
        help=(
            "Optional comma-separated fixed residual scales over equal forecast "
            "blocks; the count must divide 32."
        ),
    )
    parser.add_argument("--history-anchor-alpha", type=float, default=0.0)
    parser.add_argument("--history-anchor-tau", type=float, default=32.0)
    parser.add_argument(
        "--history-anchor-trend-beta",
        type=float,
        default=0.0,
        help=(
            "Strength of target-independent linear extrapolation of the "
            "physical-prior residual estimated from observed history."
        ),
    )
    parser.add_argument("--history-anchor-trend-lag", type=int, default=8)
    parser.add_argument(
        "--training-prior-exclusion",
        choices=("target", "window"),
        default="target",
        help=(
            "Frames removed from each training sample's interpolation bank. "
            "'window' removes both its 16-frame history and 32 target keys to "
            "better match strict protected-window validation."
        ),
    )
    parser.add_argument("--early-frame-weight", type=float, default=1.0)
    parser.add_argument("--early-frame-count", type=int, default=8)
    parser.add_argument("--early-frame-start", type=int, default=0)
    parser.add_argument(
        "--hard-scenario-id",
        type=int,
        default=-1,
        help="Optional strict-training scenario id receiving extra early-frame loss.",
    )
    parser.add_argument(
        "--hard-scenario-early-weight",
        type=float,
        default=1.0,
        help="Multiplicative early-frame loss weight for --hard-scenario-id.",
    )
    parser.add_argument("--coordinate-channels", action="store_true")
    parser.add_argument("--longitudinal-kernel", type=int, default=0)
    parser.add_argument("--visibility-power", type=float, default=16.0)
    parser.add_argument("--condition-time-fallback", action="store_true")
    parser.add_argument("--condition-fallback-distance", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument(
        "--protected-min-span",
        type=int,
        default=640,
        help="Minimum validation/held-out forecast span protected from training.",
    )
    parser.add_argument(
        "--protected-max-span",
        type=int,
        default=0,
        help="Maximum protected span; zero keeps every span above the minimum.",
    )
    parser.add_argument(
        "--train-scenarios",
        help="Comma-separated strict training scenario ids; default uses all admitted rows",
    )
    parser.add_argument(
        "--train-spans",
        help=(
            "Comma-separated strict training forecast spans; default uses "
            "every canonical FIT span."
        ),
    )
    parser.add_argument(
        "--selection-scenarios",
        help=(
            "Comma-separated validation scenario ids used for complete-trajectory "
            "checkpoint selection; default uses all validation scenarios"
        ),
    )
    parser.add_argument(
        "--scenario-sampling-multipliers",
        help="Eight comma-separated positive strict-training sampler weights",
    )
    parser.add_argument(
        "--balance-train-scenario-span",
        action="store_true",
        help="Sample each observed strict (scenario, span) group equally",
    )
    args = parser.parse_args()

    residual_block_scales = None
    if args.residual_block_scales:
        residual_block_scales = tuple(
            float(value.strip())
            for value in args.residual_block_scales.split(",")
            if value.strip()
        )
        if (
            not residual_block_scales
            or 32 % len(residual_block_scales) != 0
            or any(value <= 0.0 for value in residual_block_scales)
        ):
            raise ValueError(
                "residual block scales must be positive and their count must divide 32"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("Strict prior refiner requires CUDA")
    if (
        args.log_mse_weight < 0.0
        or args.psnr_loss_weight < 0.0
        or args.coarse_ssim_weight < 0.0
        or args.smoke_mse_weight < 0.0
        or args.gradient_weight < 0.0
        or args.wavelet_weight < 0.0
        or args.hard_scenario_id < -1
        or args.hard_scenario_id >= 8
        or args.hard_scenario_early_weight <= 0.0
        or args.wavelet_levels <= 0
        or not 0.0 <= args.ema_decay < 1.0
        or args.time_fourier < 0
        or args.temporal_attention_heads < 0
        or (
            args.temporal_attention_heads > 0
            and args.dim % args.temporal_attention_heads != 0
        )
        or args.axial_spectral_blocks < 0
        or args.axial_spectral_modes <= 0
        or args.horizon_block_heads < 0
        or (
            args.horizon_block_heads > 0
            and 32 % args.horizon_block_heads != 0
        )
        or (
            args.horizon_block_heads > 0
            and args.multiscale_output_head
        )
        or (
            args.horizon_block_heads > 0
            and args.learned_prior_warp
        )
        or (
            residual_block_scales is not None
            and args.learnable_horizon_scale
        )
        or args.residual_scale <= 0.0
        or args.max_longitudinal_warp <= 0.0
        or args.max_vertical_warp <= 0.0
        or args.history_anchor_alpha < 0.0
        or args.history_anchor_tau <= 0.0
        or args.history_anchor_trend_beta < 0.0
        or not 1 <= args.history_anchor_trend_lag <= 15
        or args.early_frame_weight <= 0.0
        or not 0 <= args.early_frame_count <= 32
        or not 0 <= args.early_frame_start <= 32
        or args.early_frame_start + args.early_frame_count > 32
        or not math.isfinite(args.visibility_power)
        or args.condition_fallback_distance < 0
        or args.protected_min_span <= 0
        or (
            args.protected_max_span > 0
            and args.protected_max_span < args.protected_min_span
        )
        or (
            args.longitudinal_kernel != 0
            and (args.longitudinal_kernel < 3 or args.longitudinal_kernel % 2 == 0)
        )
    ):
        raise ValueError(
            "log-MSE weight/time Fourier count must be non-negative and "
            "residual scale/history-anchor tau/early weight must be positive"
        )
    seed_everything(args.seed)
    device = torch.device("cuda")
    store = FITStore(args.data_root)
    segment_index = make_segment_index(history_frames=16)
    _, validation, heldout = split_segment_index(segment_index, args.split_seed)
    span_is_protected = lambda row: (
        row[2] >= args.protected_min_span
        and (
            args.protected_max_span <= 0
            or row[2] <= args.protected_max_span
        )
    )
    validation = [row for row in validation if span_is_protected(row)]
    heldout = [row for row in heldout if span_is_protected(row)]
    if not validation or not heldout:
        raise RuntimeError("Protected span filter removed validation or held-out rows")
    canonical_spans = {row[2] for row in segment_index}
    selected_training_spans = set(canonical_spans)
    if args.train_spans:
        selected_training_spans = {
            int(value.strip())
            for value in args.train_spans.split(",")
            if value.strip()
        }
        if (
            not selected_training_spans
            or not selected_training_spans <= canonical_spans
        ):
            raise ValueError("Training spans must be canonical FIT span values")
    full_train = strict_frame_disjoint_variable_training_rows(
        validation,
        heldout,
        16,
        int(store.data.shape[1]),
        selected_training_spans,
        {row[0] for row in segment_index},
    )
    train = list(full_train)
    scenario_sampling_multipliers = parse_scenario_sampling_multipliers(
        args.scenario_sampling_multipliers
    )
    selected_training_scenarios = None
    if args.train_scenarios:
        selected_training_scenarios = {
            int(value.strip())
            for value in args.train_scenarios.split(",")
            if value.strip()
        }
        if not selected_training_scenarios or any(
            value < 0 or value >= 8 for value in selected_training_scenarios
        ):
            raise ValueError("train scenarios must be non-empty ids in [0, 7]")
        train = [
            row for row in train if int(row[0]) in selected_training_scenarios
        ]
        if not train:
            raise RuntimeError("Scenario filtering removed all strict training rows")
    selected_validation_scenarios = None
    selection_validation = list(validation)
    if args.selection_scenarios:
        selected_validation_scenarios = {
            int(value.strip())
            for value in args.selection_scenarios.split(",")
            if value.strip()
        }
        if not selected_validation_scenarios or any(
            value < 0 or value >= 8 for value in selected_validation_scenarios
        ):
            raise ValueError("selection scenarios must be non-empty ids in [0, 7]")
        selection_validation = [
            row
            for row in validation
            if int(row[0]) in selected_validation_scenarios
        ]
        if not selection_validation:
            raise RuntimeError("Scenario filtering removed all validation rows")
    training_frame_keys = target_keys(full_train, 16) | history_keys(full_train, 16)
    protected_frame_keys = target_keys(validation + heldout, 16) | history_keys(
        validation + heldout, 16
    )
    overlap = training_frame_keys & protected_frame_keys
    if overlap:
        raise RuntimeError(f"Strict frame overlap: {sorted(overlap)[:5]}")
    audit = {
        "protocol": "strict frame-disjoint dense variable-span windows",
        "protected_min_span": args.protected_min_span,
        "protected_max_span": (
            args.protected_max_span if args.protected_max_span > 0 else None
        ),
        "training_windows": len(train),
        "admitted_training_windows": len(full_train),
        "training_scenarios": sorted({int(row[0]) for row in train}),
        "training_spans": sorted({int(row[2]) for row in train}),
        "training_scenario_counts": {
            str(scenario): sum(int(row[0]) == scenario for row in train)
            for scenario in range(8)
        },
        "scenario_sampling_multipliers": list(scenario_sampling_multipliers),
        "balanced_scenario_span_sampling": args.balance_train_scenario_span,
        "validation_windows": len(validation),
        "selection_validation_windows": len(selection_validation),
        "selection_validation_scenarios": sorted(
            {int(row[0]) for row in selection_validation}
        ),
        "heldout_windows_reserved": len(heldout),
        "training_frame_keys": len(training_frame_keys),
        "protected_validation_test_frame_keys": len(protected_frame_keys),
        "train_evaluation_frame_overlap": len(overlap),
        "training_prior": (
            f"leave-{args.training_prior_exclusion}-out "
            f"power-{args.visibility_power:g} "
            "PCHIP condition-time prior"
        ),
        "validation_prior": (
            f"train-frame-only power-{args.visibility_power:g} "
            "PCHIP condition-time prior"
        ),
        "history_mode": args.history_mode,
        "cosine_lr": args.cosine_lr,
        "time_fourier": args.time_fourier,
        "relative_time_channel": args.relative_time_channel,
        "condition_film": args.condition_film,
        "learnable_horizon_scale": args.learnable_horizon_scale,
        "horizon_block_heads": args.horizon_block_heads,
        "multiscale_output_head": args.multiscale_output_head,
        "learned_prior_warp": args.learned_prior_warp,
        "max_longitudinal_warp": args.max_longitudinal_warp,
        "max_vertical_warp": args.max_vertical_warp,
        "log_mse_weight": args.log_mse_weight,
        "psnr_loss_weight": args.psnr_loss_weight,
        "coarse_ssim_weight": args.coarse_ssim_weight,
        "smoke_mse_weight": args.smoke_mse_weight,
        "gradient_weight": args.gradient_weight,
        "wavelet_weight": args.wavelet_weight,
        "wavelet_levels": args.wavelet_levels,
        "residual_scale": args.residual_scale,
        "residual_block_scales": list(residual_block_scales)
        if residual_block_scales is not None
        else None,
        "history_anchor_alpha": args.history_anchor_alpha,
        "history_anchor_tau": args.history_anchor_tau,
        "history_anchor_trend_beta": args.history_anchor_trend_beta,
        "history_anchor_trend_lag": args.history_anchor_trend_lag,
        "training_prior_exclusion": args.training_prior_exclusion,
        "ema_decay": args.ema_decay,
        "early_frame_weight": args.early_frame_weight,
        "early_frame_count": args.early_frame_count,
        "early_frame_start": args.early_frame_start,
        "hard_scenario_id": args.hard_scenario_id,
        "hard_scenario_early_weight": args.hard_scenario_early_weight,
        "coordinate_channels": args.coordinate_channels,
        "longitudinal_kernel": args.longitudinal_kernel,
        "temporal_dilated": args.temporal_dilated,
        "temporal_attention_heads": args.temporal_attention_heads,
        "axial_spectral_blocks": args.axial_spectral_blocks,
        "axial_spectral_modes": args.axial_spectral_modes,
        "visibility_power": args.visibility_power,
        "metric_aligned_loss": args.metric_aligned_loss,
        "condition_time_fallback": args.condition_time_fallback,
        "condition_fallback_distance": args.condition_fallback_distance,
    }

    train_dataset = StrictPriorDataset(
        store,
        train,
        training_frame_keys,
        True,
        visibility_power=args.visibility_power,
        history_mode=args.history_mode,
        history_anchor_alpha=args.history_anchor_alpha,
        history_anchor_tau=args.history_anchor_tau,
        history_anchor_trend_beta=args.history_anchor_trend_beta,
        history_anchor_trend_lag=args.history_anchor_trend_lag,
        training_prior_exclusion=args.training_prior_exclusion,
        condition_time_fallback=args.condition_time_fallback,
        condition_fallback_distance=args.condition_fallback_distance,
    )
    validation_dataset = StrictPriorDataset(
        store,
        validation,
        training_frame_keys,
        False,
        visibility_power=args.visibility_power,
        history_mode=args.history_mode,
        history_anchor_alpha=args.history_anchor_alpha,
        history_anchor_tau=args.history_anchor_tau,
        history_anchor_trend_beta=args.history_anchor_trend_beta,
        history_anchor_trend_lag=args.history_anchor_trend_lag,
        training_prior_exclusion=args.training_prior_exclusion,
        condition_time_fallback=args.condition_time_fallback,
        condition_fallback_distance=args.condition_fallback_distance,
    )
    selection_validation_dataset = StrictPriorDataset(
        store,
        selection_validation,
        training_frame_keys,
        False,
        visibility_power=args.visibility_power,
        history_mode=args.history_mode,
        history_anchor_alpha=args.history_anchor_alpha,
        history_anchor_tau=args.history_anchor_tau,
        history_anchor_trend_beta=args.history_anchor_trend_beta,
        history_anchor_trend_lag=args.history_anchor_trend_lag,
        training_prior_exclusion=args.training_prior_exclusion,
        condition_time_fallback=args.condition_time_fallback,
        condition_fallback_distance=args.condition_fallback_distance,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.balance_train_scenario_span:
        group_counts: dict[tuple[int, int], int] = {}
        for scenario, _, span in train:
            key = (int(scenario), int(span))
            group_counts[key] = group_counts.get(key, 0) + 1
        sample_weights = [
            scenario_sampling_multipliers[int(scenario)]
            / group_counts[(int(scenario), int(span))]
            for scenario, _, span in train
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
    elif any(value != 1.0 for value in scenario_sampling_multipliers):
        sampler = WeightedRandomSampler(
            [scenario_sampling_multipliers[int(row[0])] for row in train],
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    selection_validation_loader = DataLoader(
        selection_validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    history_channels = {
        "compact": 3,
        "multiscale": 16,
        "full": 32,
    }[args.history_mode]
    model = PriorResidualRefiner(
        args.dim,
        args.blocks,
        args.dilated,
        args.temporal_blocks,
        history_channels,
        args.time_fourier,
        args.residual_scale,
        args.coordinate_channels,
        args.longitudinal_kernel,
        args.temporal_dilated,
        args.relative_time_channel,
        args.condition_film,
        args.learnable_horizon_scale,
        args.horizon_block_heads,
        args.multiscale_output_head,
        args.learned_prior_warp,
        args.max_longitudinal_warp,
        args.max_vertical_warp,
        residual_block_scales,
        args.temporal_attention_heads,
        args.axial_spectral_blocks,
        args.axial_spectral_modes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.steps,
            eta_min=0.05 * args.learning_rate,
        )
        if args.cosine_lr
        else None
    )
    initial_step = 0
    if args.initial_checkpoint:
        initial_path = Path(args.initial_checkpoint)
        if "test" in str(initial_path).lower():
            raise ValueError(
                f"held-out/test checkpoint is not permitted: {initial_path}"
            )
        initial_state = torch.load(
            initial_path, map_location=device, weights_only=False
        )
        initial_audit = initial_state.get("data_audit", {})
        if initial_audit.get("train_evaluation_frame_overlap") != 0:
            raise RuntimeError(
                f"initial checkpoint lacks a zero-overlap audit: {initial_path}"
            )
        model.load_state_dict(initial_state["model"], strict=True)
        initial_step = int(initial_state.get("step", 0))
    ema_model = None
    if args.ema_decay > 0.0:
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    iterator = iter(train_loader)
    start_time = time.time()

    evaluation_model = ema_model if ema_model is not None else model
    baseline = evaluate(evaluation_model, selection_validation_loader, device)
    best_score = baseline["ssim"]
    atomic_save(
        output / "best.pt",
        {
            "model": evaluation_model.state_dict(),
            "step": initial_step,
            "best_score": best_score,
            "checkpoint_variant": "ema" if ema_model is not None else "raw",
            "args": vars(args),
            "data_audit": audit,
        },
    )
    print(
        json.dumps(
            {
                "step": initial_step,
                "additional_step": 0,
                "validation": baseline,
            }
        ),
        flush=True,
    )

    for additional_step in range(1, args.steps + 1):
        step = initial_step + additional_step
        try:
            prior, target, history, conditions, times, scenario, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            prior, target, history, conditions, times, scenario, _ = next(iterator)
        prior = prior.to(device)
        target = target.to(device)
        history = history.to(device)
        conditions = conditions.to(device)
        times = times.to(device)
        model.train()
        prediction = model(prior, history, conditions, times)
        loss_prediction = prediction
        if args.metric_aligned_loss:
            loss_prediction = torch.clamp(
                smooth(loss_prediction, 0.5, 0.05) * 1.05,
                0.0,
                1.0,
            )
        frame_mse = (loss_prediction - target).square().mean(dim=(-2, -1))
        frame_ssim = ssim_per_image(
            loss_prediction.flatten(0, 1), target.flatten(0, 1)
        ).reshape(loss_prediction.shape[:2])
        if args.coarse_ssim_weight > 0.0:
            flattened_prediction = loss_prediction.flatten(0, 1)[:, None]
            flattened_target = target.flatten(0, 1)[:, None]
            coarse_ssim_2x = ssim_per_image(
                F.avg_pool2d(flattened_prediction, kernel_size=2, stride=2)[:, 0],
                F.avg_pool2d(flattened_target, kernel_size=2, stride=2)[:, 0],
            ).reshape(loss_prediction.shape[:2])
            coarse_ssim_x4 = ssim_per_image(
                F.avg_pool2d(
                    flattened_prediction, kernel_size=(1, 4), stride=(1, 4)
                )[:, 0],
                F.avg_pool2d(
                    flattened_target, kernel_size=(1, 4), stride=(1, 4)
                )[:, 0],
            ).reshape(loss_prediction.shape[:2])
            frame_coarse_ssim = 0.5 * (coarse_ssim_2x + coarse_ssim_x4)
        else:
            frame_coarse_ssim = frame_ssim
        temporal_weights = torch.ones(
            loss_prediction.shape[1],
            device=loss_prediction.device,
            dtype=loss_prediction.dtype,
        )
        temporal_weights[
            args.early_frame_start : args.early_frame_start
            + args.early_frame_count
        ] = args.early_frame_weight
        temporal_weights = temporal_weights / temporal_weights.mean()
        frame_weights = temporal_weights[None].repeat(loss_prediction.shape[0], 1)
        if args.hard_scenario_id >= 0 and args.hard_scenario_early_weight != 1.0:
            hard_mask = scenario.to(device=prediction.device) == args.hard_scenario_id
            frame_weights[
                hard_mask,
                args.early_frame_start : args.early_frame_start
                + args.early_frame_count,
            ] *= args.hard_scenario_early_weight
            frame_weights = frame_weights / frame_weights.mean(
                dim=1, keepdim=True
            )
        mse = (frame_mse * frame_weights).mean()
        smoke_importance = (1.0 - target).square()
        frame_smoke_mse = (
            (loss_prediction - target).square() * smoke_importance
        ).sum(dim=(-2, -1)) / smoke_importance.sum(dim=(-2, -1)).clamp_min(1e-6)
        smoke_mse = (frame_smoke_mse * frame_weights).mean()
        gradient_x = (
            loss_prediction[..., 1:] - loss_prediction[..., :-1]
            - target[..., 1:] + target[..., :-1]
        ).abs().mean(dim=(-2, -1))
        gradient_y = (
            loss_prediction[..., 1:, :] - loss_prediction[..., :-1, :]
            - target[..., 1:, :] + target[..., :-1, :]
        ).abs().mean(dim=(-2, -1))
        gradient_loss = (
            0.5 * (gradient_x + gradient_y) * frame_weights
        ).mean()
        if args.wavelet_weight > 0.0:
            wavelet_loss = haar_wavelet_l1(
                loss_prediction,
                target,
                args.wavelet_levels,
            )
        else:
            wavelet_loss = prediction.new_zeros(())
        log_mse = (
            torch.log(frame_mse.clamp_min(1e-8)) * frame_weights
        ).mean()
        psnr_surrogate = torch.log(frame_mse.float().clamp_min(1e-8)).mean()
        ssim = (frame_ssim * frame_weights).mean()
        coarse_ssim = (frame_coarse_ssim * frame_weights).mean()
        loss = (
            args.mse_weight * mse
            + args.log_mse_weight * log_mse
            + args.psnr_loss_weight * psnr_surrogate
            + args.ssim_weight * (1.0 - ssim)
            + args.coarse_ssim_weight * (1.0 - coarse_ssim)
            + args.smoke_mse_weight * smoke_mse
            + args.gradient_weight * gradient_loss
            + args.wavelet_weight * wavelet_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema_model is not None:
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), model.parameters()
                ):
                    ema_parameter.mul_(args.ema_decay).add_(
                        parameter, alpha=1.0 - args.ema_decay
                    )
                for ema_buffer, buffer in zip(
                    ema_model.buffers(), model.buffers()
                ):
                    ema_buffer.copy_(buffer)
        if step % 50 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss.detach()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "train_mse": float(mse.detach()),
                        "train_log_mse": float(log_mse.detach()),
                        "train_psnr_surrogate": float(psnr_surrogate.detach()),
                        "train_ssim": float(ssim.detach()),
                        "train_coarse_ssim": float(coarse_ssim.detach()),
                        "train_smoke_mse": float(smoke_mse.detach()),
                        "train_gradient_loss": float(gradient_loss.detach()),
                        "train_wavelet_loss": float(wavelet_loss.detach()),
                        "elapsed_seconds": time.time() - start_time,
                    }
                ),
                flush=True,
            )
        if (
            additional_step % args.validate_every == 0
            or additional_step == args.steps
        ):
            evaluation_model = ema_model if ema_model is not None else model
            metrics = evaluate(
                evaluation_model, selection_validation_loader, device
            )
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
            if metrics["ssim"] > best_score:
                best_score = metrics["ssim"]
                atomic_save(
                    output / "best.pt",
                    {
                        "model": evaluation_model.state_dict(),
                        "step": step,
                        "best_score": best_score,
                        "checkpoint_variant": (
                            "ema" if ema_model is not None else "raw"
                        ),
                        "args": vars(args),
                        "data_audit": audit,
                    },
                )

    state = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    metrics = evaluate(
        model,
        validation_loader,
        device,
        str(output / "val_arrays.npz"),
    )
    selection_metrics = evaluate(model, selection_validation_loader, device)
    payload = {
        "evaluation_split": "validation",
        "test_accessed": False,
        "model": "FireWave strict visibility physical-prior residual refiner",
        "checkpoint_step": int(state["step"]),
        "full_metrics": metrics,
        "selection_metrics": selection_metrics,
        "selection_scenarios": sorted(selected_validation_scenarios)
        if selected_validation_scenarios is not None
        else None,
        "data_audit": audit,
    }
    (output / "val_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
