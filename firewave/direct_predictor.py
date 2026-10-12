#!/usr/bin/env python3
"""Leakage-audited direct 16-to-32 frame predictor for FireWave.

The existing autoregressive prior accumulates one-step errors across the
forecast horizon.  This module instead encodes all observed frames, mixes the
complete history in a SimVP-style latent translator, and decodes all future
visibility fields in one pass.  It keeps the established FIT split and never
uses held-out targets for training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from .core import (
        FITStore,
        FITTrajectoryDataset,
        MetricAccumulator,
        make_segment_index,
        postprocess_visibility,
        seed_everything,
        ssim_per_image,
        split_segment_index,
        update_ema,
    )
except ImportError:
    from firewave.core import (
        FITStore,
        FITTrajectoryDataset,
        MetricAccumulator,
        make_segment_index,
        postprocess_visibility,
        seed_everything,
        ssim_per_image,
        split_segment_index,
        update_ema,
    )


def group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


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


class Residual2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(input_channels), input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(field)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(field) + hidden


class GatedSpatiotemporalBlock(nn.Module):
    """Large-kernel gated block used in the latent temporal translator."""

    def __init__(self, channels: int, width_dilation: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            (3, 7),
            padding=(1, 3 * width_dilation),
            dilation=(1, width_dilation),
            groups=channels,
        )
        self.gate = nn.Conv2d(channels, channels * 2, 1)
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(group_count(channels), channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.residual_scale = nn.Parameter(torch.ones(2))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(F.silu(self.norm1(field)))
        value, gate = self.gate(hidden).chunk(2, dim=1)
        field = field + self.residual_scale[0] * self.project(
            value * torch.sigmoid(gate)
        )
        return field + self.residual_scale[1] * self.mlp(
            F.silu(self.norm2(field))
        )


class FireWaveDirectSimVP(nn.Module):
    """Direct conditional history-to-trajectory model for visibility."""

    def __init__(
        self,
        history_frames: int = 16,
        future_frames: int = 32,
        dim: int = 64,
        translator_blocks: int = 8,
        use_start_time_conditioning: bool = False,
        time_fourier_bands: int = 1,
        centered_history: bool = False,
        predict_increments: bool = False,
        spatial_time_prior_rank: int = 0,
        scenario_horizon_affine: bool = False,
        history_flow_warp: bool = False,
        multiscale_history_trend: bool = False,
        highres_temporal_residual: bool = False,
        scenario_time_keyframes: int = 0,
        factorized_scenario_time_keyframes: int = 0,
        continuous_condition_time_keyframes: int = 0,
        temperature_auxiliary: bool = False,
    ):
        super().__init__()
        if (
            history_frames < 2
            or future_frames < 1
            or dim < 16
            or time_fourier_bands < 1
            or spatial_time_prior_rank < 0
            or scenario_time_keyframes < 0
            or factorized_scenario_time_keyframes < 0
            or continuous_condition_time_keyframes < 0
        ):
            raise ValueError("Invalid direct-predictor dimensions")
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.dim = dim
        self.use_start_time_conditioning = use_start_time_conditioning
        self.time_fourier_bands = time_fourier_bands
        self.centered_history = centered_history
        self.predict_increments = predict_increments
        self.spatial_time_prior_rank = spatial_time_prior_rank
        self.use_scenario_horizon_affine = scenario_horizon_affine
        self.use_history_flow_warp = history_flow_warp
        self.use_multiscale_history_trend = multiscale_history_trend
        self.use_highres_temporal_residual = highres_temporal_residual
        self.scenario_time_keyframes = scenario_time_keyframes
        self.factorized_scenario_time_keyframes = (
            factorized_scenario_time_keyframes
        )
        self.continuous_condition_time_keyframes = (
            continuous_condition_time_keyframes
        )
        self.use_temperature_auxiliary = temperature_auxiliary
        # Per-frame inputs: temperature/visibility, their first differences,
        # and two spatial coordinates.  The centered variant separates the
        # last visibility field from history-relative evolution.
        input_channels = 7 if centered_history else 6
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, dim, 4, stride=2, padding=1),
            Residual2D(dim, dim),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, (3, 4), stride=(1, 2), padding=(1, 1)),
            Residual2D(dim * 2, dim * 2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 3, 4, stride=2, padding=1),
            Residual2D(dim * 3, dim * 3),
        )
        latent_channels = dim * 2
        hidden_channels = dim * 4
        self.latent_compress = nn.Conv2d(dim * 3, latent_channels, 1)
        self.temporal_input = nn.Conv2d(
            history_frames * latent_channels, hidden_channels, 1
        )
        self.condition = nn.Sequential(
            nn.Linear(12, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )
        dilations = (1, 2, 4, 8)
        self.translator = nn.Sequential(
            *(
                GatedSpatiotemporalBlock(
                    hidden_channels, dilations[index % len(dilations)]
                )
                for index in range(translator_blocks)
            )
        )
        self.temporal_output = nn.Conv2d(
            hidden_channels, future_frames * latent_channels, 1
        )
        self.horizon_embedding = nn.Parameter(
            torch.zeros(1, future_frames, latent_channels, 1, 1)
        )
        self.horizon_condition = nn.Sequential(
            nn.Linear(12, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, future_frames * latent_channels * 2),
        )
        nn.init.zeros_(self.horizon_condition[-1].weight)
        nn.init.zeros_(self.horizon_condition[-1].bias)
        if use_start_time_conditioning:
            time_feature_dim = 2 + 2 * time_fourier_bands
            self.start_time_condition = nn.Sequential(
                nn.Linear(time_feature_dim, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels * 2),
            )
            self.horizon_time_condition = nn.Sequential(
                nn.Linear(time_feature_dim, latent_channels * 2),
                nn.SiLU(),
                nn.Linear(latent_channels * 2, latent_channels * 2),
            )
            nn.init.zeros_(self.start_time_condition[-1].weight)
            nn.init.zeros_(self.start_time_condition[-1].bias)
            nn.init.zeros_(self.horizon_time_condition[-1].weight)
            nn.init.zeros_(self.horizon_time_condition[-1].bias)
        if spatial_time_prior_rank:
            time_feature_dim = 2 + 2 * time_fourier_bands
            prior_hidden = max(64, dim * 2)
            self.spatial_time_prior_coefficients = nn.Sequential(
                nn.Linear(12 + time_feature_dim, prior_hidden),
                nn.SiLU(),
                nn.Linear(prior_hidden, spatial_time_prior_rank),
            )
            self.spatial_time_prior_basis = nn.Parameter(
                torch.empty(spatial_time_prior_rank, 23, 480)
            )
            nn.init.normal_(self.spatial_time_prior_basis, mean=0.0, std=0.02)
            nn.init.zeros_(self.spatial_time_prior_coefficients[-1].weight)
            nn.init.zeros_(self.spatial_time_prior_coefficients[-1].bias)
        if scenario_horizon_affine:
            affine_hidden = max(64, dim * 2)
            self.scenario_horizon_affine = nn.Sequential(
                nn.Linear(12, affine_hidden),
                nn.SiLU(),
                nn.Linear(affine_hidden, future_frames * 2),
            )
            nn.init.zeros_(self.scenario_horizon_affine[-1].weight)
            nn.init.zeros_(self.scenario_horizon_affine[-1].bias)
        if multiscale_history_trend:
            if history_frames < 9:
                raise ValueError(
                    "Multiscale history trend requires at least nine history frames"
                )
            time_feature_dim = 2 + 2 * time_fourier_bands
            trend_hidden = max(64, dim * 2)
            self.history_trend_condition = nn.Sequential(
                nn.Linear(12 + time_feature_dim, trend_hidden),
                nn.SiLU(),
                nn.Linear(trend_hidden, 4),
            )
            nn.init.zeros_(self.history_trend_condition[-1].weight)
            nn.init.zeros_(self.history_trend_condition[-1].bias)
        if highres_temporal_residual:
            highres_channels = max(32, dim)
            self.highres_input = nn.Conv2d(
                history_frames * 3,
                highres_channels,
                1,
            )
            self.highres_blocks = nn.Sequential(
                Residual2D(highres_channels, highres_channels),
                GatedSpatiotemporalBlock(highres_channels, 1),
                GatedSpatiotemporalBlock(highres_channels, 2),
            )
            self.highres_condition = nn.Sequential(
                nn.Linear(12, highres_channels * 2),
                nn.SiLU(),
                nn.Linear(highres_channels * 2, highres_channels * 2),
            )
            nn.init.zeros_(self.highres_condition[-1].weight)
            nn.init.zeros_(self.highres_condition[-1].bias)
            if use_start_time_conditioning:
                time_feature_dim = 2 + 2 * time_fourier_bands
                self.highres_time_condition = nn.Sequential(
                    nn.Linear(time_feature_dim, highres_channels * 2),
                    nn.SiLU(),
                    nn.Linear(highres_channels * 2, highres_channels * 2),
                )
                nn.init.zeros_(self.highres_time_condition[-1].weight)
                nn.init.zeros_(self.highres_time_condition[-1].bias)
            self.highres_output = nn.Sequential(
                nn.GroupNorm(group_count(highres_channels), highres_channels),
                nn.SiLU(),
                nn.Conv2d(
                    highres_channels,
                    future_frames,
                    (3, 7),
                    padding=(1, 3),
                ),
            )
            nn.init.zeros_(self.highres_output[-1].weight)
            nn.init.zeros_(self.highres_output[-1].bias)
        if scenario_time_keyframes:
            if scenario_time_keyframes < 2:
                raise ValueError("Scenario-time prior requires at least two keyframes")
            # Eight FIT condition combinations share a temporally interpolated
            # field bank.  Exact validation/held-out frames remain excluded
            # from optimization by the strict row constructor below.
            self.scenario_time_keyframe_prior = nn.Parameter(
                torch.zeros(8, scenario_time_keyframes, 23, 480)
            )
        if factorized_scenario_time_keyframes:
            if factorized_scenario_time_keyframes < 2:
                raise ValueError(
                    "Factorized scenario-time prior requires at least two keyframes"
                )
            # One global, four HRR-specific, and two ventilation-specific
            # components support condition-combination generalization when a
            # complete scenario is absent from strict training rows.
            self.factorized_scenario_time_keyframe_prior = nn.Parameter(
                torch.zeros(
                    7,
                    factorized_scenario_time_keyframes,
                    23,
                    480,
                )
            )
        if continuous_condition_time_keyframes:
            if continuous_condition_time_keyframes < 2:
                raise ValueError(
                    "Continuous condition-time prior requires at least two keyframes"
                )
            # A continuous HRR polynomial with ventilation interaction can
            # extrapolate to strict scenarios whose exact condition has no
            # admitted training frame (notably the 50-MW A4/B4 cases).
            self.continuous_condition_time_keyframe_prior = nn.Parameter(
                torch.zeros(
                    5,
                    continuous_condition_time_keyframes,
                    23,
                    480,
                )
            )

        self.up3 = nn.ConvTranspose2d(
            latent_channels, dim * 2, 4, stride=2, padding=1
        )
        self.dec2 = nn.Sequential(
            Residual2D(dim * 4, dim * 2),
            Residual2D(dim * 2, dim * 2),
        )
        self.up2 = nn.ConvTranspose2d(
            dim * 2, dim, (3, 4), stride=(1, 2), padding=(1, 1)
        )
        self.dec1 = nn.Sequential(
            Residual2D(dim * 2, dim),
            Residual2D(dim, dim),
        )
        self.up1 = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)
        self.output = nn.Sequential(
            Residual2D(dim, dim),
            nn.GroupNorm(group_count(dim), dim),
            nn.SiLU(),
            nn.Conv2d(dim, 1, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        if temperature_auxiliary:
            self.temperature_output = nn.Sequential(
                Residual2D(dim, dim),
                nn.GroupNorm(group_count(dim), dim),
                nn.SiLU(),
                nn.Conv2d(dim, 1, 1),
            )
            nn.init.zeros_(self.temperature_output[-1].weight)
            nn.init.zeros_(self.temperature_output[-1].bias)
        if history_flow_warp:
            self.history_flow_warp = nn.Sequential(
                Residual2D(dim, dim),
                nn.GroupNorm(group_count(dim), dim),
                nn.SiLU(),
                nn.Conv2d(dim, 2, 1),
            )
            nn.init.zeros_(self.history_flow_warp[-1].weight)
            nn.init.zeros_(self.history_flow_warp[-1].bias)
        self.trend_scale = nn.Parameter(torch.zeros(future_frames))

    @staticmethod
    def condition_features(conditions: torch.Tensor) -> torch.Tensor:
        hrr_levels = conditions.new_tensor((0.0, 0.3010299957, 0.6020599913, 1.0))
        hrr_index = (conditions[:, :1] - hrr_levels[None]).abs().argmin(dim=1)
        ventilation_index = (conditions[:, 1] >= 0.5).long()
        scenario_index = ventilation_index * 4 + hrr_index
        scenario = F.one_hot(scenario_index, num_classes=8).to(conditions.dtype)
        return torch.cat((conditions, scenario), dim=1)

    def absolute_time_features(self, times: torch.Tensor) -> torch.Tensor:
        """Encode known absolute FIT frame indices without target values."""

        scaled = 2.0 * times / 1000.0 - 1.0
        frequencies = torch.arange(
            1,
            self.time_fourier_bands + 1,
            device=times.device,
            dtype=times.dtype,
        )
        phase = 2.0 * math.pi * times[..., None] * frequencies / 1000.0
        return torch.cat(
            (
                scaled[..., None],
                scaled.square()[..., None],
                torch.sin(phase),
                torch.cos(phase),
            ),
            dim=-1,
        )

    def future_time_grid(
        self,
        conditions: torch.Tensor,
        forecast_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact target times and elapsed offsets for each sample.

        FIT trajectories always contain 32 target fields, but their physical
        spans range from 31 to 896 source frames.  The former implementation
        treated every target sequence as 32 consecutive frames.  Using the
        known span condition keeps the short-span behavior identical while
        exposing the true absolute times to long-horizon branches.
        """

        fractions = torch.linspace(
            0.0,
            1.0,
            self.future_frames,
            device=conditions.device,
            dtype=conditions.dtype,
        )[None]
        span_frames = (1000.0 * conditions[:, 3:4]).clamp_min(0.0)
        target_offsets = torch.round(span_frames * fractions)
        absolute_times = forecast_start[:, None] + target_offsets
        # The last history frame precedes forecast_start by one source frame.
        elapsed_offsets = target_offsets + 1.0
        return absolute_times, elapsed_offsets

    def forward(
        self,
        history: torch.Tensor,
        conditions: torch.Tensor,
        starts: torch.Tensor | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = history.shape
        if frames != self.history_frames or channels != 2:
            raise ValueError(f"Unexpected history shape: {tuple(history.shape)}")
        previous = torch.cat((history[:, :1], history[:, :-1]), dim=1)
        differences = history - previous
        coordinate_y = torch.linspace(
            -1.0, 1.0, height, device=history.device, dtype=history.dtype
        )[None, None, None, :, None].expand(batch, frames, 1, -1, width)
        coordinate_x = torch.linspace(
            -1.0, 1.0, width, device=history.device, dtype=history.dtype
        )[None, None, None, None, :].expand(batch, frames, 1, height, -1)
        if self.centered_history:
            visibility_anchor = history[:, -1:, 1:2].expand(
                -1, frames, -1, -1, -1
            )
            encoder_fields = torch.cat(
                (
                    history[:, :, 0:1],
                    history[:, :, 1:2] - visibility_anchor,
                    visibility_anchor,
                ),
                dim=2,
            )
        else:
            encoder_fields = history
        encoder_input = torch.cat(
            (encoder_fields, differences, coordinate_y, coordinate_x), dim=2
        ).flatten(0, 1)
        pad_height = height % 2
        if pad_height:
            encoder_input = F.pad(
                encoder_input, (0, 0, 0, pad_height), mode="replicate"
            )
        level1 = self.enc1(encoder_input)
        level2 = self.enc2(level1)
        level3 = self.latent_compress(self.enc3(level2))
        latent_height, latent_width = level3.shape[-2:]
        latent = level3.reshape(
            batch, frames, -1, latent_height, latent_width
        ).flatten(1, 2)
        latent = self.temporal_input(latent)

        condition_features = self.condition_features(conditions)
        scale, shift = self.condition(condition_features).chunk(2, dim=1)
        forecast_start = None
        absolute_times = None
        elapsed_offsets = None
        if starts is not None:
            forecast_start = starts.to(
                device=history.device,
                dtype=history.dtype,
            ).reshape(batch) + float(self.history_frames)
            absolute_times, elapsed_offsets = self.future_time_grid(
                conditions,
                forecast_start,
            )
        if self.use_start_time_conditioning:
            if forecast_start is None:
                raise ValueError("Start indices are required for time conditioning")
            time_scale, time_shift = self.start_time_condition(
                self.absolute_time_features(forecast_start)
            ).chunk(2, dim=1)
            scale = scale + time_scale
            shift = shift + time_shift
        latent = (
            latent * (1.0 + scale[:, :, None, None])
            + shift[:, :, None, None]
        )
        latent = self.translator(latent)
        latent = self.temporal_output(latent).reshape(
            batch,
            self.future_frames,
            self.dim * 2,
            latent_height,
            latent_width,
        )
        horizon_scale, horizon_shift = self.horizon_condition(
            condition_features
        ).reshape(batch, self.future_frames, self.dim * 4).chunk(2, dim=2)
        if self.use_start_time_conditioning:
            if absolute_times is None:
                raise RuntimeError("Missing absolute future-time grid")
            time_scale, time_shift = self.horizon_time_condition(
                self.absolute_time_features(absolute_times)
            ).chunk(2, dim=-1)
            horizon_scale = horizon_scale + time_scale
            horizon_shift = horizon_shift + time_shift
        latent = (
            (latent + self.horizon_embedding)
            * (1.0 + horizon_scale[:, :, :, None, None])
            + horizon_shift[:, :, :, None, None]
        )

        future = self.future_frames
        level2_sequence = level2.reshape(batch, frames, *level2.shape[1:])
        level1_sequence = level1.reshape(batch, frames, *level1.shape[1:])
        skip2 = level2_sequence[:, -1:, ...].expand(
            -1, future, -1, -1, -1
        ).flatten(0, 1)
        skip1 = level1_sequence[:, -1:, ...].expand(
            -1, future, -1, -1, -1
        ).flatten(0, 1)
        decoded = self.up3(latent.flatten(0, 1))
        decoded = self.dec2(torch.cat((decoded, skip2), dim=1))
        decoded = self.up2(decoded)
        decoded = self.dec1(torch.cat((decoded, skip1), dim=1))
        high_resolution = self.up1(decoded)
        residual = self.output(high_resolution)[..., :height, :width]
        residual = residual.reshape(batch, future, 1, height, width)

        if self.use_highres_temporal_residual:
            highres_input = torch.cat(
                (
                    history[:, :, 0],
                    history[:, :, 1],
                    differences[:, :, 1],
                ),
                dim=1,
            )
            highres = self.highres_input(highres_input)
            highres_scale, highres_shift = self.highres_condition(
                condition_features
            ).chunk(2, dim=1)
            if self.use_start_time_conditioning:
                time_scale, time_shift = self.highres_time_condition(
                    self.absolute_time_features(forecast_start)
                ).chunk(2, dim=1)
                highres_scale = highres_scale + time_scale
                highres_shift = highres_shift + time_shift
            highres = (
                highres * (1.0 + highres_scale[:, :, None, None])
                + highres_shift[:, :, None, None]
            )
            highres = self.highres_blocks(highres)
            highres = self.highres_output(highres).reshape(
                batch, future, 1, height, width
            )
            residual = residual + highres

        warped_initial = None
        if self.use_history_flow_warp:
            flow = self.history_flow_warp(high_resolution)[..., :height, :width]
            flow = 0.25 * torch.tanh(flow).reshape(
                batch, future, 2, height, width
            )
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    -1.0, 1.0, height,
                    device=history.device, dtype=history.dtype,
                ),
                torch.linspace(
                    -1.0, 1.0, width,
                    device=history.device, dtype=history.dtype,
                ),
                indexing="ij",
            )
            sampling_grid = torch.stack((grid_x, grid_y), dim=-1)[None, None]
            sampling_grid = sampling_grid + flow.permute(0, 1, 3, 4, 2)
            anchor = history[:, -1:, 1:2].expand(
                -1, future, -1, -1, -1
            )
            warped_initial = F.grid_sample(
                anchor.flatten(0, 1),
                sampling_grid.flatten(0, 1),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).reshape(batch, future, 1, height, width)

        if self.spatial_time_prior_rank:
            if starts is None:
                raise ValueError("Start indices are required for the spatial-time prior")
            if absolute_times is None:
                raise RuntimeError("Missing absolute future-time grid")
            prior_features = torch.cat(
                (
                    condition_features[:, None].expand(-1, future, -1),
                    self.absolute_time_features(absolute_times),
                ),
                dim=-1,
            )
            prior_coefficients = self.spatial_time_prior_coefficients(
                prior_features
            )
            prior_basis = self.spatial_time_prior_basis
            if prior_basis.shape[-2:] != (height, width):
                prior_basis = F.interpolate(
                    prior_basis[:, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            spatial_time_prior = torch.einsum(
                "bfr,rhw->bfhw", prior_coefficients, prior_basis
            )
            residual = residual + spatial_time_prior[:, :, None]

        if self.scenario_time_keyframes:
            if absolute_times is None:
                raise RuntimeError(
                    "Start indices are required for the scenario-time keyframe prior"
                )
            scenario_indices = condition_features[:, -8:].argmax(dim=-1)
            positions = (
                absolute_times.clamp(0.0, 1000.0)
                * float(self.scenario_time_keyframes - 1)
                / 1000.0
            )
            left_indices = positions.floor().long()
            right_indices = (left_indices + 1).clamp_max(
                self.scenario_time_keyframes - 1
            )
            scenario_grid = scenario_indices[:, None].expand(-1, future)
            left_fields = self.scenario_time_keyframe_prior[
                scenario_grid, left_indices
            ]
            right_fields = self.scenario_time_keyframe_prior[
                scenario_grid, right_indices
            ]
            interpolation = (positions - left_indices).to(
                dtype=left_fields.dtype
            )[:, :, None, None]
            keyframe_prior = torch.lerp(
                left_fields,
                right_fields,
                interpolation,
            )
            if keyframe_prior.shape[-2:] != (height, width):
                keyframe_prior = F.interpolate(
                    keyframe_prior.flatten(0, 1)[:, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].reshape(batch, future, height, width)
            residual = residual + 0.5 * torch.tanh(keyframe_prior)[:, :, None]

        if self.factorized_scenario_time_keyframes:
            if absolute_times is None:
                raise RuntimeError(
                    "Start indices are required for the factorized keyframe prior"
                )
            scenario_indices = condition_features[:, -8:].argmax(dim=-1)
            hrr_indices = scenario_indices.remainder(4)
            ventilation_indices = scenario_indices.div(4, rounding_mode="floor")
            positions = (
                absolute_times.clamp(0.0, 1000.0)
                * float(self.factorized_scenario_time_keyframes - 1)
                / 1000.0
            )
            left_indices = positions.floor().long()
            right_indices = (left_indices + 1).clamp_max(
                self.factorized_scenario_time_keyframes - 1
            )
            component_indices = torch.stack(
                (
                    torch.zeros_like(hrr_indices),
                    1 + hrr_indices,
                    5 + ventilation_indices,
                ),
                dim=1,
            )
            component_grid = component_indices[:, :, None].expand(
                -1, -1, future
            )
            left_grid = left_indices[:, None].expand(-1, 3, -1)
            right_grid = right_indices[:, None].expand(-1, 3, -1)
            left_fields = self.factorized_scenario_time_keyframe_prior[
                component_grid, left_grid
            ].sum(dim=1)
            right_fields = self.factorized_scenario_time_keyframe_prior[
                component_grid, right_grid
            ].sum(dim=1)
            interpolation = (positions - left_indices).to(
                dtype=left_fields.dtype
            )[:, :, None, None]
            factorized_prior = torch.lerp(
                left_fields,
                right_fields,
                interpolation,
            )
            if factorized_prior.shape[-2:] != (height, width):
                factorized_prior = F.interpolate(
                    factorized_prior.flatten(0, 1)[:, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].reshape(batch, future, height, width)
            residual = residual + 0.5 * torch.tanh(factorized_prior)[:, :, None]

        if self.continuous_condition_time_keyframes:
            if absolute_times is None:
                raise RuntimeError(
                    "Start indices are required for the continuous condition-time prior"
                )
            positions = (
                absolute_times.clamp(0.0, 1000.0)
                * float(self.continuous_condition_time_keyframes - 1)
                / 1000.0
            )
            left_indices = positions.floor().long()
            right_indices = (left_indices + 1).clamp_max(
                self.continuous_condition_time_keyframes - 1
            )
            hrr = conditions[:, 0]
            ventilation = conditions[:, 1]
            basis_weights = torch.stack(
                (
                    torch.ones_like(hrr),
                    hrr,
                    hrr.square(),
                    ventilation,
                    hrr * ventilation,
                ),
                dim=1,
            )
            component_indices = torch.arange(
                5, device=history.device
            )[None, None].expand(batch, future, -1)
            left_grid = left_indices[:, :, None].expand(-1, -1, 5)
            right_grid = right_indices[:, :, None].expand(-1, -1, 5)
            left_fields = self.continuous_condition_time_keyframe_prior[
                component_indices, left_grid
            ]
            right_fields = self.continuous_condition_time_keyframe_prior[
                component_indices, right_grid
            ]
            interpolation = (positions - left_indices).to(
                dtype=left_fields.dtype
            )[:, :, None, None, None]
            continuous_components = torch.lerp(
                left_fields,
                right_fields,
                interpolation,
            )
            continuous_prior = (
                continuous_components
                * basis_weights[:, None, :, None, None]
            ).sum(dim=2)
            if continuous_prior.shape[-2:] != (height, width):
                continuous_prior = F.interpolate(
                    continuous_prior.flatten(0, 1)[:, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].reshape(batch, future, height, width)
            residual = residual + 0.5 * torch.tanh(continuous_prior)[:, :, None]

        initial = history[:, -1:, 1:2]
        velocity = history[:, -1:, 1:2] - history[:, -2:-1, 1:2]
        if elapsed_offsets is None:
            elapsed_offsets = torch.arange(
                1,
                future + 1,
                device=history.device,
                dtype=history.dtype,
            )[None].expand(batch, -1)
        horizon = elapsed_offsets[:, :, None, None, None]
        trend = torch.tanh(self.trend_scale)[None, :, None, None, None]
        anchor = warped_initial if warped_initial is not None else initial
        if self.predict_increments:
            increments = trend * velocity + residual
            forecast = anchor + increments.cumsum(dim=1)
        else:
            forecast = anchor + trend * horizon * velocity + residual
        if self.use_multiscale_history_trend:
            if starts is None:
                raise ValueError(
                    "Start indices are required for the multiscale history trend"
                )
            if absolute_times is None:
                raise RuntimeError("Missing absolute future-time grid")
            trend_features = torch.cat(
                (
                    condition_features[:, None].expand(-1, future, -1),
                    self.absolute_time_features(absolute_times),
                ),
                dim=-1,
            )
            trend_coefficients = 2.0 * torch.tanh(
                self.history_trend_condition(trend_features)
            )
            offsets = (1, 2, 4, 8)
            velocities = torch.stack(
                tuple(
                    (
                        history[:, -1, 1:2]
                        - history[:, -1 - offset, 1:2]
                    ) / float(offset)
                    for offset in offsets
                ),
                dim=1,
            )
            multiscale_velocity = torch.einsum(
                "bfk,bkchw->bfchw", trend_coefficients, velocities
            )
            forecast = forecast + horizon * multiscale_velocity
        if self.use_scenario_horizon_affine:
            affine = self.scenario_horizon_affine(condition_features).reshape(
                batch, future, 2
            )
            scale = 1.0 + 0.1 * torch.tanh(affine[:, :, 0])
            bias = 0.02 * torch.tanh(affine[:, :, 1])
            forecast = (
                forecast * scale[:, :, None, None, None]
                + bias[:, :, None, None, None]
            )
        visibility_forecast = forecast.clamp(0.0, 1.0)
        if not return_auxiliary:
            return visibility_forecast
        if not self.use_temperature_auxiliary:
            raise RuntimeError("Temperature auxiliary output was not configured")
        temperature_residual = self.temperature_output(high_resolution)[
            ..., :height, :width
        ].reshape(batch, future, 1, height, width)
        temperature_forecast = (
            history[:, -1:, 0:1] + temperature_residual
        ).clamp(0.0, 1.0)
        return visibility_forecast, temperature_forecast


def target_keys(rows, history_frames: int) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for scenario, start, span in rows:
        forecast_start = start + history_frames
        indices = np.rint(
            np.linspace(forecast_start, forecast_start + span, 32)
        ).astype(np.int64)
        keys.update((int(scenario), int(index)) for index in indices)
    return keys


def history_keys(rows, history_frames: int) -> set[tuple[int, int]]:
    """Return every exact scenario/time field used as a model input."""

    return {
        (int(scenario), int(index))
        for scenario, start, _ in rows
        for index in range(int(start), int(start) + history_frames)
    }


def strict_frame_disjoint_dense_training_rows(
    canonical_train,
    validation,
    test,
    history_frames: int,
    n_time: int,
    future_frames: int = 32,
):
    """Densify canonical supervision without reusing any evaluation frame.

    The canonical random-window split contains many exact scenario/time fields
    that serve as a training input in one window and an evaluation input or
    target in another.  This stricter index keeps supervision inside the
    canonical training target bank, protects every validation/test history and
    target field, and then enumerates all consecutive 16-to-32 windows that can
    be formed from the remaining training-only fields.
    """

    canonical_target_keys = target_keys(canonical_train, history_frames)
    canonical_history_keys = history_keys(canonical_train, history_frames)
    evaluation_rows = list(validation) + list(test)
    protected_frame_keys = (
        target_keys(evaluation_rows, history_frames)
        | history_keys(evaluation_rows, history_frames)
    )
    available_training_keys = (
        canonical_target_keys | canonical_history_keys
    ) - protected_frame_keys

    span = future_frames - 1
    max_start = n_time - 1 - history_frames - span
    dense_rows = []
    scenario_ids = sorted({int(row[0]) for row in canonical_train})
    for scenario in scenario_ids:
        for start in range(max_start + 1):
            row = (scenario, start, span)
            row_targets = target_keys((row,), history_frames)
            row_history = history_keys((row,), history_frames)
            if (
                row_targets <= canonical_target_keys
                and row_history <= available_training_keys
                and not ((row_targets | row_history) & protected_frame_keys)
            ):
                dense_rows.append(row)
    if not dense_rows:
        raise RuntimeError("Strict frame-disjoint densification produced no windows")
    return dense_rows


def strict_frame_disjoint_variable_training_rows(
    validation,
    test,
    history_frames: int,
    n_time: int,
    span_values,
    scenario_ids,
):
    """Enumerate variable-span training rows disjoint from long evaluation.

    Validation and held-out histories plus targets form one protected frame
    bank.  Candidate rows may use any canonical FIT span, but every input and
    target frame must remain outside that bank.
    """

    evaluation_rows = list(validation) + list(test)
    protected_frame_keys = (
        target_keys(evaluation_rows, history_frames)
        | history_keys(evaluation_rows, history_frames)
    )
    dense_rows = []
    for scenario in sorted(int(value) for value in scenario_ids):
        for span in sorted(int(value) for value in span_values):
            max_start = n_time - 1 - history_frames - span
            for start in range(max_start + 1):
                row = (scenario, start, span)
                row_frames = (
                    target_keys((row,), history_frames)
                    | history_keys((row,), history_frames)
                )
                if row_frames.isdisjoint(protected_frame_keys):
                    dense_rows.append(row)
    if not dense_rows:
        raise RuntimeError(
            "Strict frame-disjoint variable-span training produced no windows"
        )
    return dense_rows


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    spatial_sigma: float,
    temporal_blend: float,
    save_arrays: str | None = None,
    scale: float = 1.0,
    bias: float = 0.0,
    gamma: float = 1.0,
    median_blend: float = 0.0,
):
    model.eval()
    accumulator = MetricAccumulator()
    predictions, targets, scenarios, starts = [], [], [], []
    for target, history, conditions, scenario, start, _ in loader:
        target = target.to(device, non_blocking=True)
        history = history.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        visibility = model(
            history,
            conditions,
            start.to(device=device, non_blocking=True),
        )
        temperature = history[:, -1:, 0:1].expand(
            -1, target.shape[1], -1, -1, -1
        )
        prediction = torch.cat((temperature, visibility), dim=2)
        prediction = postprocess_visibility(
            prediction,
            spatial_sigma=spatial_sigma,
            temporal_blend=temporal_blend,
            scale=scale,
            bias=bias,
            gamma=gamma,
            median_blend=median_blend,
        )
        accumulator.update(prediction, target)
        if save_arrays:
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
            scenarios.append(np.asarray(scenario))
            starts.append(np.asarray(start))
    if save_arrays:
        np.savez_compressed(
            save_arrays,
            prediction=np.concatenate(predictions),
            target=np.concatenate(targets),
            scenario=np.concatenate(scenarios),
            start=np.concatenate(starts),
        )
    return accumulator.result()


def save_checkpoint(path, model, ema, optimizer, step, best_score, args, audit):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_score": best_score,
            "args": vars(args),
            "data_audit": audit,
        },
        temporary,
    )
    temporary.replace(path)


def load_compatible_direct_state(
    module: nn.Module,
    source: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Load a direct predictor while allowing spatial-prior rank expansion."""

    target = module.state_dict()
    loaded: set[str] = set()
    unexpected: list[str] = []
    expandable = {
        "spatial_time_prior_basis",
        "spatial_time_prior_coefficients.2.weight",
        "spatial_time_prior_coefficients.2.bias",
    }
    for key, value in source.items():
        if key not in target:
            unexpected.append(key)
            continue
        if target[key].shape == value.shape:
            target[key] = value
            loaded.add(key)
            continue
        if (
            key in expandable
            and target[key].ndim == value.ndim
            and target[key].shape[0] > value.shape[0]
            and target[key].shape[1:] == value.shape[1:]
        ):
            merged = target[key].clone()
            merged[: value.shape[0]] = value
            if key != "spatial_time_prior_basis":
                merged[value.shape[0] :].zero_()
            target[key] = merged
            loaded.add(key)
            continue
        unexpected.append(
            f"{key}: source={tuple(value.shape)} target={tuple(target[key].shape)}"
        )
    module.load_state_dict(target)
    return {
        "missing_keys": [key for key in target if key not in loaded],
        "unexpected_keys": unexpected,
    }


def direct_forecast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float,
    log_mse_weight: float,
    psnr_loss_weight: float,
    ssim_loss_weight: float,
    horizon_weight_end: float,
    psnr_epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a training loss whose PSNR term matches the reported metric.

    The reported visibility PSNR is the mean of per-frame PSNR values with
    ``data_range=1``.  Its optimizable part is therefore the equally weighted
    mean of ``log(frame_mse)``; unlike the legacy horizon-weighted objectives,
    this term applies no extra temporal reweighting.
    """

    error = (prediction - target).square()
    per_frame_mse = error.mean(dim=(-1, -2))
    horizon_weights = torch.linspace(
        1.0,
        horizon_weight_end,
        per_frame_mse.shape[1],
        device=per_frame_mse.device,
        dtype=per_frame_mse.dtype,
    )[None, :, None]
    horizon_weights = horizon_weights / horizon_weights.mean()
    mse = (per_frame_mse * horizon_weights).mean()
    log_mse = (
        torch.log1p(per_frame_mse / 1e-4) * horizon_weights
    ).mean()
    psnr_surrogate = torch.log(
        per_frame_mse.float().clamp_min(psnr_epsilon)
    ).mean()
    per_frame_ssim = ssim_per_image(
        prediction[:, :, 0].flatten(0, 1).float(),
        target[:, :, 0].flatten(0, 1).float(),
    ).reshape(prediction.shape[0], prediction.shape[1])
    ssim_loss = (
        (1.0 - per_frame_ssim) * horizon_weights[..., 0].float()
    ).mean()
    loss = (
        mse_weight * mse
        + log_mse_weight * log_mse
        + psnr_loss_weight * psnr_surrogate
        + ssim_loss_weight * ssim_loss
    )
    return loss, {
        "mse": mse,
        "log_mse": log_mse,
        "psnr_surrogate": psnr_surrogate,
        "ssim_loss": ssim_loss,
    }


def differentiable_visibility_postprocess(
    prediction: torch.Tensor,
    spatial_sigma: float,
    temporal_blend: float,
    scale: float,
    bias: float,
    gamma: float,
    median_blend: float,
) -> torch.Tensor:
    """Apply the validation transform without in-place writes for training."""

    if prediction.ndim != 5 or prediction.shape[2] != 1:
        raise ValueError(
            "Differentiable visibility postprocess expects [B,F,1,H,W]"
        )
    visibility = prediction[:, :, 0]
    if spatial_sigma > 0.0:
        radius = max(1, int(math.ceil(3.0 * spatial_sigma)))
        coordinates = torch.arange(
            -radius,
            radius + 1,
            device=visibility.device,
            dtype=visibility.dtype,
        )
        kernel_1d = torch.exp(
            -coordinates.square() / (2.0 * spatial_sigma ** 2)
        )
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
        images = visibility.flatten(0, 1)[:, None]
        images = F.pad(
            images,
            (radius, radius, radius, radius),
            mode="reflect",
        )
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
        visibility = (
            (1.0 - median_blend) * visibility
            + median_blend * median
        )
    visibility = (visibility * scale + bias).clamp(0.0, 1.0).pow(gamma)
    return visibility[:, :, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--translator-blocks", type=int, default=8)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--log-mse-weight", type=float, default=0.05)
    parser.add_argument("--psnr-loss-weight", type=float, default=0.0)
    parser.add_argument("--ssim-loss-weight", type=float, default=0.0)
    parser.add_argument("--psnr-epsilon", type=float, default=1e-8)
    parser.add_argument("--selection-ssim-weight", type=float, default=1.0)
    parser.add_argument("--selection-psnr-weight", type=float, default=1.0)
    parser.add_argument("--use-start-time-conditioning", action="store_true")
    parser.add_argument("--time-fourier-bands", type=int, default=1)
    parser.add_argument("--centered-history", action="store_true")
    parser.add_argument("--predict-increments", action="store_true")
    parser.add_argument(
        "--spatial-time-prior-rank",
        type=int,
        default=0,
        help=(
            "Rank of a scenario- and absolute-time-conditioned spatial field "
            "prior; zero disables the branch"
        ),
    )
    parser.add_argument(
        "--scenario-horizon-affine",
        action="store_true",
        help="Learn a bounded scenario-conditioned scale and bias per horizon",
    )
    parser.add_argument(
        "--history-flow-warp",
        action="store_true",
        help=(
            "Warp the last observed visibility field with a learned bounded "
            "per-horizon flow before adding the forecast residual"
        ),
    )
    parser.add_argument(
        "--multiscale-history-trend",
        action="store_true",
        help=(
            "Add a scenario/time-conditioned mixture of 1, 2, 4, and 8-frame "
            "history velocities"
        ),
    )
    parser.add_argument(
        "--history-module-learning-rate-multiplier",
        type=float,
        default=1.0,
        help=(
            "Learning-rate multiplier for the optional history-flow and "
            "multiscale-trend modules"
        ),
    )
    parser.add_argument(
        "--highres-temporal-residual",
        action="store_true",
        help=(
            "Add a zero-initialized full-resolution temporal residual branch "
            "that preserves fine spatial dynamics from the observed history"
        ),
    )
    parser.add_argument(
        "--highres-module-learning-rate-multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for the full-resolution residual branch",
    )
    parser.add_argument(
        "--freeze-backbone-for-highres",
        action="store_true",
        help="Optimize only the full-resolution residual branch",
    )
    parser.add_argument(
        "--scenario-time-keyframes",
        type=int,
        default=0,
        help=(
            "Number of temporally interpolated per-scenario field keyframes; "
            "zero disables the strict scenario-time prior"
        ),
    )
    parser.add_argument(
        "--factorized-scenario-time-keyframes",
        type=int,
        default=0,
        help=(
            "Number of global+HRR+ventilation keyframes used by the "
            "condition-factorized strict scenario-time prior"
        ),
    )
    parser.add_argument(
        "--continuous-condition-time-keyframes",
        type=int,
        default=0,
        help=(
            "Number of continuous HRR/ventilation polynomial field keyframes; "
            "zero disables the extrapolating strict condition-time prior"
        ),
    )
    parser.add_argument(
        "--keyframe-prior-learning-rate-multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for the scenario-time keyframe prior",
    )
    parser.add_argument(
        "--continuous-prior-learning-rate-multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for the continuous condition-time prior",
    )
    parser.add_argument(
        "--freeze-backbone-for-keyframe-prior",
        action="store_true",
        help="Optimize only the scenario-time keyframe prior",
    )
    parser.add_argument(
        "--freeze-backbone-for-continuous-prior",
        action="store_true",
        help="Optimize only the continuous HRR/ventilation condition-time prior",
    )
    parser.add_argument(
        "--temperature-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of a training-only future-temperature forecasting loss "
            "that regularizes the shared visibility representation"
        ),
    )
    parser.add_argument(
        "--strict-frame-disjoint-dense-train",
        action="store_true",
        help=(
            "Use dense canonical-training windows while excluding every exact "
            "scenario/time field present in validation or held-out histories "
            "and targets"
        ),
    )
    parser.add_argument(
        "--strict-dense-variable-train",
        action="store_true",
        help=(
            "Train on all canonical span values while protecting every frame "
            "in the long-span validation and held-out banks"
        ),
    )
    parser.add_argument(
        "--strict-protected-min-span",
        type=int,
        default=0,
        help="Minimum validation/held-out span protected in variable-span mode",
    )
    parser.add_argument(
        "--train-min-span",
        type=int,
        default=0,
        help="Minimum optimization span after strict frame filtering",
    )
    parser.add_argument(
        "--balance-train-scenario-span",
        action="store_true",
        help="Use inverse-frequency sampling over scenario/span groups",
    )
    parser.add_argument(
        "--scenario-sampling-multipliers",
        help=(
            "Comma-separated positive sampling multipliers for FIT scenarios "
            "0--7, applied after scenario/span balancing"
        ),
    )
    parser.add_argument("--horizon-weight-end", type=float, default=1.0)
    parser.add_argument("--eval-spatial-sigma", type=float, default=0.0)
    parser.add_argument("--eval-temporal-blend", type=float, default=0.1)
    parser.add_argument("--eval-scale", type=float, default=1.0)
    parser.add_argument("--eval-bias", type=float, default=0.0)
    parser.add_argument("--eval-gamma", type=float, default=1.0)
    parser.add_argument("--eval-median-blend", type=float, default=0.0)
    parser.add_argument(
        "--train-postprocess-aligned",
        action="store_true",
        help=(
            "Compute the training loss after applying the same fixed, "
            "target-free visibility transform used for validation"
        ),
    )
    parser.add_argument("--resume")
    parser.add_argument("--reset-optimizer", action="store_true")
    args = parser.parse_args()
    if args.history_frames < 2:
        raise ValueError("At least two history frames are required")
    if args.horizon_weight_end <= 0.0:
        raise ValueError("horizon weight must be positive")
    if min(
        args.mse_weight,
        args.log_mse_weight,
        args.psnr_loss_weight,
        args.ssim_loss_weight,
    ) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if (
        args.mse_weight
        + args.log_mse_weight
        + args.psnr_loss_weight
        + args.ssim_loss_weight
        <= 0.0
    ):
        raise ValueError("at least one training loss weight must be positive")
    if args.psnr_epsilon <= 0.0:
        raise ValueError("psnr epsilon must be positive")
    if args.selection_ssim_weight < 0.0:
        raise ValueError("selection SSIM weight must be non-negative")
    if args.selection_psnr_weight < 0.0:
        raise ValueError("selection PSNR weight must be non-negative")
    if args.selection_ssim_weight + args.selection_psnr_weight <= 0.0:
        raise ValueError("at least one checkpoint-selection weight must be positive")
    if args.time_fourier_bands < 1:
        raise ValueError("time Fourier bands must be positive")
    if args.spatial_time_prior_rank < 0:
        raise ValueError("spatial-time prior rank must be non-negative")
    if args.history_module_learning_rate_multiplier <= 0.0:
        raise ValueError("history-module learning-rate multiplier must be positive")
    if args.highres_module_learning_rate_multiplier <= 0.0:
        raise ValueError("high-resolution module learning-rate multiplier must be positive")
    if args.keyframe_prior_learning_rate_multiplier <= 0.0:
        raise ValueError("keyframe-prior learning-rate multiplier must be positive")
    if args.continuous_prior_learning_rate_multiplier <= 0.0:
        raise ValueError("continuous-prior learning-rate multiplier must be positive")
    if args.scenario_time_keyframes < 0 or args.scenario_time_keyframes == 1:
        raise ValueError("scenario-time keyframes must be zero or at least two")
    if (
        args.factorized_scenario_time_keyframes < 0
        or args.factorized_scenario_time_keyframes == 1
    ):
        raise ValueError(
            "factorized scenario-time keyframes must be zero or at least two"
        )
    if (
        args.continuous_condition_time_keyframes < 0
        or args.continuous_condition_time_keyframes == 1
    ):
        raise ValueError(
            "continuous condition-time keyframes must be zero or at least two"
        )
    if (
        args.scenario_time_keyframes
        and args.factorized_scenario_time_keyframes
    ):
        raise ValueError("Select only one scenario-time keyframe prior")
    if args.weight_decay < 0.0:
        raise ValueError("weight decay must be non-negative")
    if args.temperature_aux_weight < 0.0:
        raise ValueError("temperature auxiliary weight must be non-negative")
    if args.strict_protected_min_span < 0:
        raise ValueError("strict protected minimum span must be non-negative")
    if args.train_min_span < 0:
        raise ValueError("training minimum span must be non-negative")
    if args.strict_dense_variable_train and args.strict_protected_min_span <= 0:
        raise ValueError(
            "variable-span strict training requires --strict-protected-min-span"
        )
    if (
        args.strict_dense_variable_train
        and args.strict_frame_disjoint_dense_train
    ):
        raise ValueError("Select only one strict densification mode")
    if args.train_min_span and not args.strict_dense_variable_train:
        raise ValueError("--train-min-span requires strict variable-span training")
    scenario_sampling_multipliers = parse_scenario_sampling_multipliers(
        args.scenario_sampling_multipliers
    )
    if (
        args.scenario_sampling_multipliers is not None
        and not args.balance_train_scenario_span
    ):
        raise ValueError(
            "scenario sampling multipliers require balanced scenario/span sampling"
        )
    if (
        args.history_module_learning_rate_multiplier != 1.0
        and not (args.history_flow_warp or args.multiscale_history_trend)
    ):
        raise ValueError(
            "history-module learning-rate multiplier requires a history module"
        )
    if (
        args.highres_module_learning_rate_multiplier != 1.0
        and not args.highres_temporal_residual
    ):
        raise ValueError(
            "high-resolution learning-rate multiplier requires its module"
        )
    if args.freeze_backbone_for_highres and not args.highres_temporal_residual:
        raise ValueError("freezing the backbone requires the high-resolution module")
    if (
        args.keyframe_prior_learning_rate_multiplier != 1.0
        and not (
            args.scenario_time_keyframes
            or args.factorized_scenario_time_keyframes
            or args.continuous_condition_time_keyframes
        )
    ):
        raise ValueError(
            "keyframe-prior learning-rate multiplier requires its module"
        )
    if (
        args.continuous_prior_learning_rate_multiplier != 1.0
        and not args.continuous_condition_time_keyframes
    ):
        raise ValueError(
            "continuous-prior learning-rate multiplier requires its module"
        )
    if (
        args.freeze_backbone_for_keyframe_prior
        and not (
            args.scenario_time_keyframes
            or args.factorized_scenario_time_keyframes
            or args.continuous_condition_time_keyframes
        )
    ):
        raise ValueError(
            "freezing the backbone requires the scenario-time keyframe prior"
        )
    if (
        args.freeze_backbone_for_continuous_prior
        and not args.continuous_condition_time_keyframes
    ):
        raise ValueError(
            "freezing the backbone for the continuous prior requires its module"
        )
    if sum(
        (
            bool(args.freeze_backbone_for_highres),
            bool(args.freeze_backbone_for_keyframe_prior),
            bool(args.freeze_backbone_for_continuous_prior),
        )
    ) > 1:
        raise ValueError("Select only one frozen optional module")

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Direct SimVP training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    store = FITStore(args.data_root)
    segment_index = make_segment_index(history_frames=args.history_frames)
    train, validation, test = split_segment_index(
        segment_index, args.split_seed
    )
    if args.strict_dense_variable_train:
        validation = [
            row for row in validation
            if row[2] >= args.strict_protected_min_span
        ]
        test = [
            row for row in test
            if row[2] >= args.strict_protected_min_span
        ]
    else:
        train = [row for row in train if row[2] == 31]
        validation = [row for row in validation if row[2] == 31]
        test = [row for row in test if row[2] == 31]
    canonical_train = list(train)
    canonical_train_frame_keys = (
        target_keys(canonical_train, args.history_frames)
        | history_keys(canonical_train, args.history_frames)
    )
    evaluation_rows = validation + test
    protected_frame_keys = (
        target_keys(evaluation_rows, args.history_frames)
        | history_keys(evaluation_rows, args.history_frames)
    )
    canonical_frame_overlap = canonical_train_frame_keys & protected_frame_keys
    if args.strict_frame_disjoint_dense_train:
        train = strict_frame_disjoint_dense_training_rows(
            canonical_train,
            validation,
            test,
            args.history_frames,
            int(store.data.shape[1]),
        )
    elif args.strict_dense_variable_train:
        train = strict_frame_disjoint_variable_training_rows(
            validation,
            test,
            args.history_frames,
            int(store.data.shape[1]),
            {row[2] for row in segment_index},
            {row[0] for row in segment_index},
        )
        if args.train_min_span:
            train = [row for row in train if row[2] >= args.train_min_span]
            if not train:
                raise RuntimeError("Training minimum span removed all strict rows")
    train_target_keys = target_keys(train, args.history_frames)
    protected_target_keys = target_keys(validation + test, args.history_frames)
    train_history_keys = history_keys(train, args.history_frames)
    target_overlap = train_target_keys & protected_target_keys
    frame_overlap = (train_target_keys | train_history_keys) & protected_frame_keys
    if target_overlap:
        raise RuntimeError(
            f"Train/evaluation target overlap: {sorted(target_overlap)[:5]}"
        )
    strict_frame_disjoint = (
        args.strict_frame_disjoint_dense_train
        or args.strict_dense_variable_train
    )
    if strict_frame_disjoint and frame_overlap:
        raise RuntimeError(
            f"Strict train/evaluation frame overlap: {sorted(frame_overlap)[:5]}"
        )
    audit = {
        "protocol": (
            "strict frame-disjoint dense variable-span windows"
            if args.strict_dense_variable_train
            else (
                "strict frame-disjoint dense canonical-training windows"
                if args.strict_frame_disjoint_dense_train
                else "canonical random-window split"
            )
        ),
        "protected_min_span": args.strict_protected_min_span,
        "training_min_span": args.train_min_span,
        "balanced_scenario_span_sampling": args.balance_train_scenario_span,
        "training_span_values": sorted({row[2] for row in train}),
        "canonical_train_windows": len(canonical_train),
        "train_windows": len(train),
        "validation_windows": len(validation),
        "heldout_windows": len(test),
        "training_target_keys": len(train_target_keys),
        "training_history_keys": len(train_history_keys),
        "protected_validation_test_target_keys": len(protected_target_keys),
        "protected_validation_test_frame_keys": len(protected_frame_keys),
        "canonical_train_protected_frame_overlap": len(canonical_frame_overlap),
        "train_evaluation_target_overlap": 0,
        "train_evaluation_frame_overlap": len(frame_overlap),
    }
    print(json.dumps({"direct_data_audit": audit}), flush=True)

    train_sampler = None
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
        generator = torch.Generator().manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        FITTrajectoryDataset(store, train, 32, args.history_frames),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        FITTrajectoryDataset(store, validation, 32, args.history_frames),
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = FireWaveDirectSimVP(
        args.history_frames,
        32,
        args.model_dim,
        args.translator_blocks,
        args.use_start_time_conditioning,
        args.time_fourier_bands,
        args.centered_history,
        args.predict_increments,
        args.spatial_time_prior_rank,
        args.scenario_horizon_affine,
        args.history_flow_warp,
        args.multiscale_history_trend,
        args.highres_temporal_residual,
        args.scenario_time_keyframes,
        args.factorized_scenario_time_keyframes,
        args.continuous_condition_time_keyframes,
        args.temperature_aux_weight > 0.0,
    ).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    history_module_prefixes = (
        "history_flow_warp.",
        "history_trend_condition.",
    )
    highres_module_prefixes = (
        "highres_input.",
        "highres_blocks.",
        "highres_condition.",
        "highres_time_condition.",
        "highres_output.",
    )
    legacy_keyframe_module_prefixes = (
        "scenario_time_keyframe_prior",
        "factorized_scenario_time_keyframe_prior",
    )
    continuous_prior_prefixes = (
        "continuous_condition_time_keyframe_prior",
    )
    keyframe_module_prefixes = (
        legacy_keyframe_module_prefixes + continuous_prior_prefixes
    )
    if args.freeze_backbone_for_highres:
        highres_parameters = []
        for name, parameter in model.named_parameters():
            is_highres = name.startswith(highres_module_prefixes)
            parameter.requires_grad_(is_highres)
            if is_highres:
                highres_parameters.append(parameter)
        if not highres_parameters:
            raise RuntimeError("No high-resolution module parameters found")
        optimizer_parameters = (
            {
                "params": highres_parameters,
                "lr": (
                    args.learning_rate
                    * args.highres_module_learning_rate_multiplier
                ),
            },
        )
    elif args.freeze_backbone_for_continuous_prior:
        continuous_parameters = []
        for name, parameter in model.named_parameters():
            is_continuous = name.startswith(continuous_prior_prefixes)
            parameter.requires_grad_(is_continuous)
            if is_continuous:
                continuous_parameters.append(parameter)
        if not continuous_parameters:
            raise RuntimeError("No continuous condition-time prior parameters found")
        optimizer_parameters = (
            {
                "params": continuous_parameters,
                "lr": (
                    args.learning_rate
                    * args.continuous_prior_learning_rate_multiplier
                ),
            },
        )
    elif args.freeze_backbone_for_keyframe_prior:
        keyframe_parameters = []
        for name, parameter in model.named_parameters():
            is_keyframe = name.startswith(keyframe_module_prefixes)
            parameter.requires_grad_(is_keyframe)
            if is_keyframe:
                keyframe_parameters.append(parameter)
        if not keyframe_parameters:
            raise RuntimeError("No scenario-time keyframe parameters found")
        optimizer_parameters = (
            {
                "params": keyframe_parameters,
                "lr": (
                    args.learning_rate
                    * args.keyframe_prior_learning_rate_multiplier
                ),
            },
        )
    elif (
        args.history_module_learning_rate_multiplier == 1.0
        and args.highres_module_learning_rate_multiplier == 1.0
        and args.keyframe_prior_learning_rate_multiplier == 1.0
        and args.continuous_prior_learning_rate_multiplier == 1.0
    ):
        optimizer_parameters = model.parameters()
    else:
        base_parameters = []
        history_module_parameters = []
        highres_module_parameters = []
        keyframe_module_parameters = []
        continuous_prior_parameters = []
        for name, parameter in model.named_parameters():
            if name.startswith(highres_module_prefixes):
                destination = highres_module_parameters
            elif name.startswith(history_module_prefixes):
                destination = history_module_parameters
            elif name.startswith(continuous_prior_prefixes):
                destination = continuous_prior_parameters
            elif name.startswith(legacy_keyframe_module_prefixes):
                destination = keyframe_module_parameters
            else:
                destination = base_parameters
            destination.append(parameter)
        if (
            args.history_module_learning_rate_multiplier != 1.0
            and not history_module_parameters
        ):
            raise RuntimeError("No history-module parameters found")
        if (
            args.highres_module_learning_rate_multiplier != 1.0
            and not highres_module_parameters
        ):
            raise RuntimeError("No high-resolution module parameters found")
        if (
            args.keyframe_prior_learning_rate_multiplier != 1.0
            and not keyframe_module_parameters
        ):
            raise RuntimeError("No scenario-time keyframe parameters found")
        if (
            args.continuous_prior_learning_rate_multiplier != 1.0
            and not continuous_prior_parameters
        ):
            raise RuntimeError("No continuous condition-time prior parameters found")
        optimizer_parameters = [{"params": base_parameters}]
        if history_module_parameters:
            optimizer_parameters.append({
                "params": history_module_parameters,
                "lr": (
                    args.learning_rate
                    * args.history_module_learning_rate_multiplier
                ),
            })
        if highres_module_parameters:
            optimizer_parameters.append({
                "params": highres_module_parameters,
                "lr": (
                    args.learning_rate
                    * args.highres_module_learning_rate_multiplier
                ),
            })
        if keyframe_module_parameters:
            optimizer_parameters.append({
                "params": keyframe_module_parameters,
                "lr": (
                    args.learning_rate
                    * args.keyframe_prior_learning_rate_multiplier
                ),
            })
        if continuous_prior_parameters:
            optimizer_parameters.append({
                "params": continuous_prior_parameters,
                "lr": (
                    args.learning_rate
                    * args.continuous_prior_learning_rate_multiplier
                ),
            })
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    step, best_score = 0, -float("inf")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model_result = load_compatible_direct_state(model, state["model"])
        ema_result = load_compatible_direct_state(ema, state["ema"])
        allowed_missing_prefixes = (
            "start_time_condition.",
            "horizon_time_condition.",
        ) if args.use_start_time_conditioning else ()
        if args.spatial_time_prior_rank:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "spatial_time_prior_coefficients.",
                "spatial_time_prior_basis",
            )
        if args.scenario_horizon_affine:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "scenario_horizon_affine.",
            )
        if args.history_flow_warp:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "history_flow_warp.",
            )
        if args.multiscale_history_trend:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "history_trend_condition.",
            )
        if args.highres_temporal_residual:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "highres_input.",
                "highres_blocks.",
                "highres_condition.",
                "highres_time_condition.",
                "highres_output.",
            )
        if args.scenario_time_keyframes:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "scenario_time_keyframe_prior",
            )
        if args.factorized_scenario_time_keyframes:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "factorized_scenario_time_keyframe_prior",
            )
        if args.continuous_condition_time_keyframes:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "continuous_condition_time_keyframe_prior",
            )
        if args.temperature_aux_weight > 0.0:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "temperature_output.",
            )
        for label, result in (("model", model_result), ("ema", ema_result)):
            disallowed_missing = [
                key for key in result["missing_keys"]
                if not key.startswith(allowed_missing_prefixes)
            ]
            if disallowed_missing or result["unexpected_keys"]:
                raise RuntimeError(
                    f"Incompatible {label} checkpoint: "
                    f"missing={disallowed_missing}, "
                    f"unexpected={result['unexpected_keys']}"
                )
        if not args.reset_optimizer:
            optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        best_score = -float("inf") if args.reset_optimizer else float(state["best_score"])

    iterator = iter(train_loader)
    start_time = time.time()
    while step < args.steps:
        try:
            target, history, conditions, _, start, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            target, history, conditions, _, start, _ = next(iterator)
        target = target.to(device, non_blocking=True)
        history = history.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_output = model(
                history,
                conditions,
                start.to(device=device, non_blocking=True),
                return_auxiliary=args.temperature_aux_weight > 0.0,
            )
            if args.temperature_aux_weight > 0.0:
                prediction, temperature_prediction = model_output
            else:
                prediction = model_output
                temperature_prediction = None
            loss_prediction = prediction
            if args.train_postprocess_aligned:
                loss_prediction = differentiable_visibility_postprocess(
                    prediction,
                    spatial_sigma=args.eval_spatial_sigma,
                    temporal_blend=args.eval_temporal_blend,
                    scale=args.eval_scale,
                    bias=args.eval_bias,
                    gamma=args.eval_gamma,
                    median_blend=args.eval_median_blend,
                )
            loss, loss_terms = direct_forecast_loss(
                loss_prediction,
                target[:, :, 1:2],
                args.mse_weight,
                args.log_mse_weight,
                args.psnr_loss_weight,
                args.ssim_loss_weight,
                args.horizon_weight_end,
                args.psnr_epsilon,
            )
            temperature_aux_loss = prediction.new_zeros(())
            if temperature_prediction is not None:
                temperature_aux_loss, _ = direct_forecast_loss(
                    temperature_prediction,
                    target[:, :, 0:1],
                    1.0,
                    args.log_mse_weight,
                    0.0,
                    0.0,
                    1.0,
                    args.psnr_epsilon,
                )
                loss = loss + args.temperature_aux_weight * temperature_aux_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, 0.999)
        step += 1
        if step == 1 or step % 50 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss),
                        "train_mse": float(loss_terms["mse"]),
                        "train_log_mse": float(loss_terms["log_mse"]),
                        "train_psnr_surrogate": float(
                            loss_terms["psnr_surrogate"]
                        ),
                        "train_ssim_loss": float(loss_terms["ssim_loss"]),
                        "temperature_aux_loss": float(temperature_aux_loss),
                        "elapsed_seconds": time.time() - start_time,
                    }
                ),
                flush=True,
            )
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(
                ema,
                validation_loader,
                device,
                args.eval_spatial_sigma,
                args.eval_temporal_blend,
                scale=args.eval_scale,
                bias=args.eval_bias,
                gamma=args.eval_gamma,
                median_blend=args.eval_median_blend,
            )
            visibility = metrics["visibility"]
            score = (
                args.selection_psnr_weight * visibility["psnr"]
                + args.selection_ssim_weight * visibility["ssim"]
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "validation": metrics,
                        "selection_score": score,
                        "elapsed_seconds": time.time() - start_time,
                    }
                ),
                flush=True,
            )
            if score > best_score:
                best_score = score
                save_checkpoint(
                    output / "best.pt",
                    model,
                    ema,
                    optimizer,
                    step,
                    best_score,
                    args,
                    audit,
                )
        if step % 500 == 0 or step == args.steps:
            save_checkpoint(
                output / "latest.pt",
                model,
                ema,
                optimizer,
                step,
                best_score,
                args,
                audit,
            )

    state = torch.load(output / "best.pt", map_location=device, weights_only=False)
    ema.load_state_dict(state["ema"])
    metrics = evaluate(
        ema,
        validation_loader,
        device,
        args.eval_spatial_sigma,
        args.eval_temporal_blend,
        str(output / "val_arrays.npz"),
        scale=args.eval_scale,
        bias=args.eval_bias,
        gamma=args.eval_gamma,
        median_blend=args.eval_median_blend,
    )
    payload = {
        "checkpoint_step": int(state["step"]),
        "evaluation_split": "validation",
        "test_accessed": False,
        "model": "FireWave direct conditional SimVP-style prior",
        "data_audit": audit,
        "postprocessing": {
            "spatial_sigma": args.eval_spatial_sigma,
            "temporal_blend": args.eval_temporal_blend,
            "scale": args.eval_scale,
            "bias": args.eval_bias,
            "gamma": args.eval_gamma,
            "median_blend": args.eval_median_blend,
        },
        "metrics": metrics,
        "saved_arrays": str(output / "val_arrays.npz"),
    }
    (output / "val_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
