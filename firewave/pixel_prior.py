#!/usr/bin/env python3
"""Train a deterministic pixel-space prior for FireWave visibility forecasts."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .core import (
        FITStore,
        FITTrajectoryDataset,
        MetricAccumulator,
        make_segment_index,
        postprocess_visibility,
        seed_everything,
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
        split_segment_index,
        update_ema,
    )


class Residual2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(x)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return x + residual


class SeparableSpatioTemporalBlock(nn.Module):
    def __init__(self, channels: int, width_dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            (3, 3, 7),
            padding=(1, 1, 3 * width_dilation),
            dilation=(1, 1, width_dilation),
            groups=channels,
        )
        self.pointwise = nn.Conv3d(channels, channels * 2, 1)
        self.output = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(F.silu(self.norm(x)))
        value, gate = self.pointwise(hidden).chunk(2, dim=1)
        return x + self.output(value * torch.sigmoid(gate))


class TemporalAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        heads = 8 if channels % 8 == 0 else 4
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        sequence = x.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width, frames, channels
        )
        normalized = self.norm1(sequence)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        sequence = sequence + attended
        sequence = sequence + self.feed_forward(self.norm2(sequence))
        return sequence.reshape(batch, height, width, frames, channels).permute(
            0, 4, 3, 1, 2
        )


class TemporalMixerBlock(nn.Module):
    def __init__(self, frames: int):
        super().__init__()
        self.norm = nn.LayerNorm(frames)
        self.mixer = nn.Sequential(
            nn.Linear(frames, frames * 2),
            nn.SiLU(),
            nn.Linear(frames * 2, frames),
        )
        nn.init.zeros_(self.mixer[-1].weight)
        nn.init.zeros_(self.mixer[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.permute(0, 1, 3, 4, 2)
        sequence = sequence + self.mixer(self.norm(sequence))
        return sequence.permute(0, 1, 4, 2, 3)


class ConditionalPixelPrior(nn.Module):
    def __init__(
        self,
        history_frames: int = 16,
        future_frames: int = 32,
        dim: int = 32,
        visibility_transform: str = "linear",
        joint_output: bool = False,
        coordinate_conditioning: bool = False,
        translator_blocks: int = 4,
        history_differences: bool = False,
        history_summary_skip: bool = False,
        temporal_attention_layers: int = 0,
        temporal_mixer_layers: int = 0,
        increment_output: bool = False,
        scenario_onehot_conditioning: bool = False,
        condition_time_modulation: bool = False,
        scenario_time_template: bool = False,
        scenario_time_template_rank: int = 8,
        scenario_time_template_height: int = 6,
        scenario_time_template_width: int = 30,
        time_fourier_modulation: bool = False,
        time_fourier_bands: int = 4,
    ):
        super().__init__()
        if visibility_transform not in ("linear", "optical-depth", "logit"):
            raise ValueError(f"Unsupported visibility transform: {visibility_transform}")
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.visibility_transform = visibility_transform
        self.joint_output = joint_output
        self.coordinate_conditioning = coordinate_conditioning
        self.translator_blocks = translator_blocks
        self.history_differences = history_differences
        self.history_summary_skip = history_summary_skip
        self.temporal_attention_layers = temporal_attention_layers
        self.temporal_mixer_layers = temporal_mixer_layers
        self.increment_output = increment_output
        self.scenario_onehot_conditioning = scenario_onehot_conditioning
        self.condition_time_modulation = condition_time_modulation
        self.scenario_time_template = scenario_time_template
        self.scenario_time_template_rank = scenario_time_template_rank
        self.scenario_time_template_height = scenario_time_template_height
        self.scenario_time_template_width = scenario_time_template_width
        self.time_fourier_modulation = time_fourier_modulation
        self.time_fourier_bands = time_fourier_bands
        if translator_blocks < 1:
            raise ValueError("translator_blocks must be positive")
        if temporal_attention_layers < 0:
            raise ValueError("temporal_attention_layers must be non-negative")
        if temporal_mixer_layers < 0:
            raise ValueError("temporal_mixer_layers must be non-negative")
        if scenario_time_template_rank < 1:
            raise ValueError("scenario_time_template_rank must be positive")
        if scenario_time_template_height < 1 or scenario_time_template_width < 1:
            raise ValueError("scenario-time template dimensions must be positive")
        if time_fourier_bands < 1:
            raise ValueError("time_fourier_bands must be positive")
        encoder_channels = 2
        if coordinate_conditioning:
            encoder_channels += 2
        if history_differences:
            encoder_channels += 2
        self.enc1 = nn.Sequential(
            nn.Conv2d(encoder_channels, dim, 4, stride=2, padding=1),
            Residual2D(dim),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, (3, 4), stride=(1, 2), padding=(1, 1)),
            Residual2D(dim * 2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 3, (3, 4), stride=(1, 2), padding=(1, 1)),
            Residual2D(dim * 3),
        )
        channels = dim * 3
        condition_dim = 4 + (8 if scenario_onehot_conditioning else 0)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )
        if condition_time_modulation:
            self.condition_time = nn.Sequential(
                nn.Linear(condition_dim, channels),
                nn.SiLU(),
                nn.Linear(channels, future_frames * 2),
            )
            nn.init.zeros_(self.condition_time[-1].weight)
            nn.init.zeros_(self.condition_time[-1].bias)
        if scenario_time_template:
            self.scenario_time_basis = nn.Parameter(
                torch.empty(
                    8,
                    scenario_time_template_rank,
                    scenario_time_template_height,
                    scenario_time_template_width,
                )
            )
            nn.init.normal_(self.scenario_time_basis, mean=0.0, std=0.02)
            self.scenario_time_coeff = nn.Sequential(
                nn.Linear(condition_dim, channels),
                nn.SiLU(),
                nn.Linear(channels, future_frames * scenario_time_template_rank),
            )
            nn.init.zeros_(self.scenario_time_coeff[-1].weight)
            nn.init.zeros_(self.scenario_time_coeff[-1].bias)
        if time_fourier_modulation:
            self.time_fourier = nn.Sequential(
                nn.Linear(condition_dim + 2 * time_fourier_bands, channels),
                nn.SiLU(),
                nn.Linear(channels, channels * 2),
            )
            nn.init.zeros_(self.time_fourier[-1].weight)
            nn.init.zeros_(self.time_fourier[-1].bias)
        self.temporal_up = nn.ConvTranspose3d(
            channels, channels, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, channels, future_frames, 1, 1))
        dilation_cycle = (1, 2, 4, 8)
        self.translator = nn.Sequential(*(
            SeparableSpatioTemporalBlock(
                channels, dilation_cycle[index % len(dilation_cycle)]
            )
            for index in range(translator_blocks)
        ))
        self.temporal_attention = nn.ModuleList(
            TemporalAttentionBlock(channels)
            for _ in range(temporal_attention_layers)
        )
        self.temporal_mixer = nn.ModuleList(
            TemporalMixerBlock(future_frames)
            for _ in range(temporal_mixer_layers)
        )
        self.up3 = nn.ConvTranspose2d(
            channels, dim * 2, (3, 4), stride=(1, 2), padding=(1, 1)
        )
        self.dec2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 2, 1), Residual2D(dim * 2))
        self.up2 = nn.ConvTranspose2d(
            dim * 2, dim, (3, 4), stride=(1, 2), padding=(1, 1)
        )
        self.dec1 = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), Residual2D(dim))
        if history_summary_skip:
            self.skip2_summary = nn.Conv2d(dim * 4, dim * 2, 1)
            self.skip1_summary = nn.Conv2d(dim * 2, dim, 1)
            nn.init.zeros_(self.skip2_summary.weight)
            nn.init.zeros_(self.skip2_summary.bias)
            nn.init.zeros_(self.skip1_summary.weight)
            nn.init.zeros_(self.skip1_summary.bias)
        self.up1 = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)
        self.output = nn.Sequential(
            Residual2D(dim),
            nn.GroupNorm(min(8, dim), dim),
            nn.SiLU(),
            nn.Conv2d(dim, 2 if joint_output else 1, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, history: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = history.shape
        if frames != self.history_frames or channels != 2:
            raise ValueError(f"Unexpected history shape: {tuple(history.shape)}")
        padded = F.pad(history.flatten(0, 1), (0, 0, 0, height % 2), mode="replicate")
        encoder_inputs = [padded]
        if self.history_differences:
            previous = torch.cat((history[:, :1], history[:, :-1]), dim=1)
            differences = F.pad(
                (history - previous).flatten(0, 1),
                (0, 0, 0, height % 2),
                mode="replicate",
            )
            encoder_inputs.append(differences)
        if self.coordinate_conditioning:
            coordinate_y = torch.linspace(
                -1.0, 1.0, padded.shape[-2], device=padded.device, dtype=padded.dtype
            )[None, None, :, None].expand(padded.shape[0], 1, -1, padded.shape[-1])
            coordinate_x = torch.linspace(
                -1.0, 1.0, padded.shape[-1], device=padded.device, dtype=padded.dtype
            )[None, None, None, :].expand(padded.shape[0], 1, padded.shape[-2], -1)
            encoder_inputs.extend((coordinate_y, coordinate_x))
        padded = torch.cat(encoder_inputs, dim=1)
        h1 = self.enc1(padded)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        feature_height, feature_width = h3.shape[-2:]
        latent = h3.reshape(batch, frames, -1, feature_height, feature_width).permute(0, 2, 1, 3, 4)
        condition_features = conditions
        scenario_index = None
        if self.scenario_onehot_conditioning or self.scenario_time_template:
            hrr_levels = conditions.new_tensor((0.0, 0.3010299957, 0.6020599913, 1.0))
            hrr_index = (conditions[:, :1] - hrr_levels[None]).abs().argmin(dim=1)
            ventilation_index = (conditions[:, 1] >= 0.5).long()
            scenario_index = ventilation_index * 4 + hrr_index
        if self.scenario_onehot_conditioning:
            scenario_onehot = F.one_hot(scenario_index, num_classes=8).to(conditions.dtype)
            condition_features = torch.cat((conditions, scenario_onehot), dim=1)
        scale, shift = self.condition(condition_features).chunk(2, dim=1)
        latent = latent * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        latent = self.temporal_up(latent) + self.time_embedding
        if self.condition_time_modulation:
            time_scale, time_shift = self.condition_time(condition_features).chunk(2, dim=1)
            latent = (
                latent * (1.0 + time_scale[:, None, :, None, None])
                + time_shift[:, None, :, None, None]
            )
        if self.time_fourier_modulation:
            future_time = conditions[:, 2:3] + (
                self.history_frames
                + torch.arange(
                    self.future_frames,
                    device=conditions.device,
                    dtype=conditions.dtype,
                )[None]
            ) / 1000.0
            frequencies = 2.0 ** torch.arange(
                self.time_fourier_bands,
                device=conditions.device,
                dtype=conditions.dtype,
            )
            phase = 2.0 * torch.pi * future_time[:, :, None] * frequencies[None, None]
            time_features = torch.cat((phase.sin(), phase.cos()), dim=2)
            modulation_features = torch.cat((
                condition_features[:, None].expand(-1, self.future_frames, -1),
                time_features,
            ), dim=2)
            fourier_scale, fourier_shift = self.time_fourier(
                modulation_features
            ).chunk(2, dim=2)
            latent = (
                latent * (1.0 + fourier_scale.permute(0, 2, 1)[:, :, :, None, None])
                + fourier_shift.permute(0, 2, 1)[:, :, :, None, None]
            )
        latent = self.translator(latent)
        for attention in self.temporal_attention:
            latent = attention(latent)
        for mixer in self.temporal_mixer:
            latent = mixer(latent)

        future = self.future_frames
        latent_2d = latent.permute(0, 2, 1, 3, 4).reshape(
            batch * future, -1, feature_height, feature_width
        )
        h2_sequence = h2.reshape(batch, frames, *h2.shape[1:])
        h1_sequence = h1.reshape(batch, frames, *h1.shape[1:])
        last_h2 = h2_sequence[:, -1]
        last_h1 = h1_sequence[:, -1]
        if self.history_summary_skip:
            last_h2 = last_h2 + self.skip2_summary(torch.cat((
                h2_sequence.mean(dim=1) - last_h2,
                last_h2 - h2_sequence[:, 0],
            ), dim=1))
            last_h1 = last_h1 + self.skip1_summary(torch.cat((
                h1_sequence.mean(dim=1) - last_h1,
                last_h1 - h1_sequence[:, 0],
            ), dim=1))
        skip2 = last_h2[:, None].expand(-1, future, -1, -1, -1).reshape(
            batch * future, *last_h2.shape[1:]
        )
        skip1 = last_h1[:, None].expand(-1, future, -1, -1, -1).reshape(
            batch * future, *last_h1.shape[1:]
        )
        decoded = self.dec2(torch.cat((self.up3(latent_2d), skip2), dim=1))
        decoded = self.dec1(torch.cat((self.up2(decoded), skip1), dim=1))
        delta = self.output(self.up1(decoded))[..., :height, :width]
        output_channels = 2 if self.joint_output else 1
        delta = delta.reshape(batch, future, output_channels, height, width)
        if self.scenario_time_template:
            template_coefficients = self.scenario_time_coeff(condition_features).reshape(
                batch, future, self.scenario_time_template_rank
            )
            template_basis = self.scenario_time_basis[scenario_index]
            template = torch.einsum(
                "bfr,brhw->bfhw", template_coefficients, template_basis
            )
            template = F.interpolate(
                template.reshape(batch * future, 1, *template.shape[-2:]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, future, 1, height, width)
            if self.joint_output:
                delta = torch.cat((delta[:, :, :1], delta[:, :, 1:2] + template), dim=2)
            else:
                delta = delta + template
        if self.joint_output:
            return (history[:, -1:] + delta).clamp(0.0, 1.0)
        initial_visibility = history[:, -1:, 1:2]
        if self.visibility_transform == "optical-depth":
            initial_optical_depth = -torch.log(initial_visibility.clamp_min(1e-4))
            optical_depth = (initial_optical_depth + delta).clamp(0.0, 9.21034)
            return torch.exp(-optical_depth)
        if self.visibility_transform == "logit":
            epsilon = 1e-4
            bounded = initial_visibility.clamp(epsilon, 1.0 - epsilon)
            initial_logit = torch.logit(bounded)
            return torch.sigmoid(initial_logit + delta)
        if self.increment_output:
            return (initial_visibility + torch.cumsum(delta, dim=1)).clamp(0.0, 1.0)
        return (initial_visibility + delta).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    save_arrays: str | None = None,
    spatial_sigma: float = 0.5,
    temporal_blend: float = 0.1,
):
    model.eval()
    accumulator = MetricAccumulator()
    predictions, targets, scenarios, starts = [], [], [], []
    for target, history, condition, scenario, start, _ in loader:
        target = target.to(device, non_blocking=True)
        history = history.to(device, non_blocking=True)
        condition = condition.to(device, non_blocking=True)
        model_prediction = model(history, condition)
        if model_prediction.shape[2] == 2:
            prediction = model_prediction
        else:
            temperature = history[:, -1:, 0:1].expand(
                -1, target.shape[1], -1, -1, -1
            )
            prediction = torch.cat((temperature, model_prediction), dim=2)
        prediction = postprocess_visibility(
            prediction,
            spatial_sigma=spatial_sigma,
            temporal_blend=temporal_blend,
        )
        accumulator.update(prediction, target)
        if save_arrays:
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
            scenarios.append(np.asarray(scenario))
            starts.append(np.asarray(start))
    metrics = accumulator.result()
    if save_arrays:
        np.savez_compressed(
            save_arrays,
            prediction=np.concatenate(predictions),
            target=np.concatenate(targets),
            scenario=np.concatenate(scenarios),
            start=np.concatenate(starts),
        )
    return metrics


def save(path, model, ema, optimizer, step, best_score, args):
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_score": best_score,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--log-mse-weight", type=float, default=0.001)
    parser.add_argument("--horizon-weight-end", type=float, default=1.0)
    parser.add_argument("--multiscale-mse-weight", type=float, default=0.0)
    parser.add_argument("--spatial-gradient-weight", type=float, default=0.0)
    parser.add_argument("--temporal-gradient-weight", type=float, default=0.0)
    parser.add_argument("--eval-spatial-sigma", type=float, default=0.5)
    parser.add_argument("--eval-temporal-blend", type=float, default=0.1)
    parser.add_argument("--joint-output", action="store_true")
    parser.add_argument("--temperature-loss-weight", type=float, default=0.1)
    parser.add_argument("--coordinate-conditioning", action="store_true")
    parser.add_argument("--translator-blocks", type=int, default=4)
    parser.add_argument("--history-differences", action="store_true")
    parser.add_argument("--history-summary-skip", action="store_true")
    parser.add_argument("--temporal-attention-layers", type=int, default=0)
    parser.add_argument("--temporal-mixer-layers", type=int, default=0)
    parser.add_argument("--increment-output", action="store_true")
    parser.add_argument("--scenario-onehot-conditioning", action="store_true")
    parser.add_argument("--condition-time-modulation", action="store_true")
    parser.add_argument("--scenario-time-template", action="store_true")
    parser.add_argument("--scenario-time-template-rank", type=int, default=8)
    parser.add_argument("--scenario-time-template-height", type=int, default=6)
    parser.add_argument("--scenario-time-template-width", type=int, default=30)
    parser.add_argument("--time-fourier-modulation", action="store_true")
    parser.add_argument("--time-fourier-bands", type=int, default=4)
    parser.add_argument(
        "--dense-short-train",
        action="store_true",
        help=(
            "Densify short-horizon training windows while reusing only target "
            "time indices assigned to the original training split."
        ),
    )
    parser.add_argument("--dense-train-stride", type=int, default=1)
    parser.add_argument("--template-only", action="store_true")
    parser.add_argument(
        "--visibility-transform",
        choices=("linear", "optical-depth", "logit"),
        default="linear",
    )
    parser.add_argument("--resume")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--scenario-ids", type=int, nargs="+")
    args = parser.parse_args()

    if args.horizon_weight_end <= 0.0:
        raise ValueError("horizon_weight_end must be positive")
    if args.multiscale_mse_weight < 0.0:
        raise ValueError("multiscale_mse_weight must be non-negative")
    if args.dense_train_stride < 1:
        raise ValueError("dense-training stride must be positive")
    if args.template_only and not args.scenario_time_template:
        raise ValueError("template-only training requires --scenario-time-template")

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Pixel-prior training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    store = FITStore(args.data_root)
    train, validation, test = split_segment_index(
        make_segment_index(history_frames=args.history_frames), args.split_seed
    )
    train = [row for row in train if row[2] == 31]
    validation = [row for row in validation if row[2] == 31]
    test = [row for row in test if row[2] == 31]
    original_train_windows = len(train)
    if args.dense_short_train:
        def target_time_indices(row):
            _, start, span = row
            forecast_start = start + args.history_frames
            return np.rint(
                np.linspace(forecast_start, forecast_start + span, 32)
            ).astype(np.int64)

        allowed_train_targets = {
            (int(scenario), int(time_index))
            for row in train
            for scenario in (row[0],)
            for time_index in target_time_indices(row)
        }
        protected_targets = {
            (int(scenario), int(time_index))
            for row in validation + test
            for scenario in (row[0],)
            for time_index in target_time_indices(row)
        }
        overlap = allowed_train_targets & protected_targets
        if overlap:
            raise RuntimeError(
                "Original short-horizon split contains train/evaluation target "
                f"overlap: {sorted(overlap)[:5]}"
            )
        dense_train = []
        max_start = 1001 - 1 - args.history_frames - 31
        for scenario in range(8):
            for start in range(0, max_start + 1, args.dense_train_stride):
                row = (scenario, start, 31)
                keys = {
                    (scenario, int(time_index))
                    for time_index in target_time_indices(row)
                }
                if keys <= allowed_train_targets:
                    dense_train.append(row)
        if not dense_train:
            raise RuntimeError("Dense short-horizon training produced no windows")
        train = dense_train
        print(json.dumps({
            "dense_short_train": True,
            "dense_train_stride": args.dense_train_stride,
            "original_train_windows": original_train_windows,
            "dense_train_windows": len(train),
            "allowed_train_target_keys": len(allowed_train_targets),
            "protected_validation_test_target_keys": len(protected_targets),
            "train_evaluation_target_overlap": 0,
        }), flush=True)
    if args.scenario_ids:
        scenario_ids = set(args.scenario_ids)
        if any(index < 0 or index > 7 for index in scenario_ids):
            raise ValueError("scenario IDs must be in [0, 7]")
        train = [row for row in train if row[0] in scenario_ids]
        validation = [row for row in validation if row[0] in scenario_ids]
    train_loader = DataLoader(
        FITTrajectoryDataset(store, train, 32, args.history_frames),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        FITTrajectoryDataset(store, validation, 32, args.history_frames),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = ConditionalPixelPrior(
        args.history_frames,
        32,
        args.model_dim,
        args.visibility_transform,
        args.joint_output,
        args.coordinate_conditioning,
        args.translator_blocks,
        args.history_differences,
        args.history_summary_skip,
        args.temporal_attention_layers,
        args.temporal_mixer_layers,
        args.increment_output,
        args.scenario_onehot_conditioning,
        args.condition_time_modulation,
        args.scenario_time_template,
        args.scenario_time_template_rank,
        args.scenario_time_template_height,
        args.scenario_time_template_width,
        args.time_fourier_modulation,
        args.time_fourier_bands,
    ).to(device)
    if args.template_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("scenario_time_basis")
                or name.startswith("scenario_time_coeff.")
            )
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    step, best_score = 0, -float("inf")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_args = state.get("args", {})
        initializing_template = (
            args.scenario_time_template
            and not bool(checkpoint_args.get("scenario_time_template", False))
        )
        initializing_fourier = (
            args.time_fourier_modulation
            and not bool(checkpoint_args.get("time_fourier_modulation", False))
        )
        for module, weights in ((model, state["model"]), (ema, state["ema"])):
            if initializing_template or initializing_fourier:
                missing, unexpected = module.load_state_dict(weights, strict=False)
                allowed_missing = {
                    key
                    for key in module.state_dict()
                    if (
                        initializing_template
                        and (
                            key.startswith("scenario_time_basis")
                            or key.startswith("scenario_time_coeff.")
                        )
                    )
                    or (initializing_fourier and key.startswith("time_fourier."))
                }
                if set(missing) != allowed_missing or unexpected:
                    raise RuntimeError(
                        "Unexpected state mismatch while initializing scenario-time "
                        f"template: missing={missing}, unexpected={unexpected}"
                    )
            else:
                module.load_state_dict(weights)
        if not args.reset_optimizer:
            optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        best_score = (
            -float("inf") if args.reset_optimizer else float(state["best_score"])
        )
    iterator = iter(train_loader)
    start_time = time.time()
    while step < args.steps:
        try:
            target, history, condition, *_ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            target, history, condition, *_ = next(iterator)
        target = target.to(device, non_blocking=True)
        history = history.to(device, non_blocking=True)
        condition = condition.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(history, condition)
            visibility_prediction = (
                prediction[:, :, 1:2] if args.joint_output else prediction
            )
            error = (visibility_prediction - target[:, :, 1:2]).square()
            per_frame_mse = error.mean(dim=(-1, -2))
            horizon_weights = torch.linspace(
                1.0,
                args.horizon_weight_end,
                per_frame_mse.shape[1],
                device=per_frame_mse.device,
                dtype=per_frame_mse.dtype,
            )[None, :, None]
            horizon_weights = horizon_weights / horizon_weights.mean()
            mse = (per_frame_mse * horizon_weights).mean()
            log_mse = (
                torch.log1p(per_frame_mse / 1e-4) * horizon_weights
            ).mean()
            loss = args.mse_weight * mse + args.log_mse_weight * log_mse
            if args.multiscale_mse_weight:
                target_visibility = target[:, :, 1:2]
                multiscale_mse = prediction.new_zeros(())
                flattened_prediction = visibility_prediction.flatten(0, 1)
                flattened_target = target_visibility.flatten(0, 1)
                for scale in (2, 4):
                    pooled_prediction = F.avg_pool2d(
                        flattened_prediction, kernel_size=scale, stride=scale
                    )
                    pooled_target = F.avg_pool2d(
                        flattened_target, kernel_size=scale, stride=scale
                    )
                    multiscale_mse = multiscale_mse + (
                        pooled_prediction - pooled_target
                    ).square().mean()
                loss = loss + args.multiscale_mse_weight * multiscale_mse / 2.0
            if args.joint_output:
                temperature_error = (
                    prediction[:, :, 0:1] - target[:, :, 0:1]
                ).square().mean()
                loss = loss + args.temperature_loss_weight * temperature_error
            if args.spatial_gradient_weight:
                target_visibility = target[:, :, 1:2]
                prediction_dx = (
                    visibility_prediction[..., 1:] - visibility_prediction[..., :-1]
                )
                target_dx = target_visibility[..., 1:] - target_visibility[..., :-1]
                prediction_dy = (
                    visibility_prediction[..., 1:, :]
                    - visibility_prediction[..., :-1, :]
                )
                target_dy = target_visibility[..., 1:, :] - target_visibility[..., :-1, :]
                spatial_gradient = (
                    (prediction_dx - target_dx).abs().mean()
                    + (prediction_dy - target_dy).abs().mean()
                )
                loss = loss + args.spatial_gradient_weight * spatial_gradient
            if args.temporal_gradient_weight:
                prediction_dt = (
                    visibility_prediction[:, 1:] - visibility_prediction[:, :-1]
                )
                target_dt = target[:, 1:, 1:2] - target[:, :-1, 1:2]
                temporal_gradient = (prediction_dt - target_dt).square().mean()
                loss = loss + args.temporal_gradient_weight * temporal_gradient
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, 0.999)
        step += 1
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "train_loss": float(loss), "elapsed_seconds": time.time() - start_time}), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(
                ema,
                validation_loader,
                device,
                spatial_sigma=args.eval_spatial_sigma,
                temporal_blend=args.eval_temporal_blend,
            )
            visibility = metrics["visibility"]
            score = visibility["ssim"] + visibility["psnr"]
            print(json.dumps({"step": step, "validation": metrics, "selection_score": score, "elapsed_seconds": time.time() - start_time}), flush=True)
            if score > best_score:
                best_score = score
                save(output / "best.pt", model, ema, optimizer, step, best_score, args)
        if step % 500 == 0 or step == args.steps:
            save(output / "latest.pt", model, ema, optimizer, step, best_score, args)
    state = torch.load(output / "best.pt", map_location=device, weights_only=False)
    ema.load_state_dict(state["ema"])
    metrics = evaluate(
        ema,
        validation_loader,
        device,
        str(output / "val_arrays.npz"),
        spatial_sigma=args.eval_spatial_sigma,
        temporal_blend=args.eval_temporal_blend,
    )
    payload = {
        "checkpoint_step": int(state["step"]),
        "evaluation_split": "validation",
        "test_accessed": False,
        "model": (
            "joint conditional pixel-space spatiotemporal prior"
            if args.joint_output
            else "conditional pixel-space spatiotemporal prior"
        ),
        "postprocessing": {
            "spatial_sigma": args.eval_spatial_sigma,
            "temporal_blend": args.eval_temporal_blend,
        },
        "metrics": metrics,
        "saved_arrays": str(output / "val_arrays.npz"),
    }
    (output / "val_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
