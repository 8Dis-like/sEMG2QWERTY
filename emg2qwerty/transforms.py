# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch
import torchaudio


T = TypeVar("T")
U = TypeVar("U")


Transform = Callable[[T], U]


@dataclass
class Compose(Generic[T, U]):
    transforms: Sequence[Callable]

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


@dataclass
class ForEach:
    """Apply a transform independently to each element along `dim`."""
    transform: Callable[[torch.Tensor], torch.Tensor]
    dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        xs = tensor.unbind(self.dim)
        ys = [self.transform(x) for x in xs]
        return torch.stack(ys, dim=self.dim)


@dataclass
class ToTensor:
    """Convert structured EMG window (numpy structured array) to tensor.

    Expects fields including emg_left and emg_right, and stacks bands on dim=1:
    output shape: (T, bands=2, C=16)
    """
    fields: Sequence[str] = ("emg_left", "emg_right")

    def __call__(self, window) -> torch.Tensor:
        # window is a numpy structured array slice from EMGSessionData
        xs = [torch.from_numpy(window[f]).to(torch.float32) for f in self.fields]
        # Each field: (T, C)
        return torch.stack(xs, dim=1)  # (T, 2, C)


# -------------------------
# Existing augmentations / feature extraction
# -------------------------

@dataclass
class RandomBandRotation:
    """Randomly roll electrode channels by offset along channel_dim."""
    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(int(offset), dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Randomly jitter left/right alignment by up to max_offset timesteps.

    Input shape: (T, bands=2, ...)
    """
    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal.

    Input: (T, ..., C)
    Output: (T, ..., C, freq)
    """
    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Time/frequency masking (SpecAugment).

    Expects shape (T, ..., C, freq) and masks along time (T) and frequency (freq).
    """
    n_time_masks: int = 2
    time_mask_param: int = 20
    n_freq_masks: int = 2
    freq_mask_param: int = 4

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Work on a view shaped (..., T, freq) by flattening non-time/freq dims.
        # tensor shape: (T, *rest, freq) where *rest may include bands/C, etc.
        if tensor.ndim < 2:
            return tensor

        T = tensor.shape[0]
        freq = tensor.shape[-1]

        x = tensor.reshape(T, -1, freq)  # (T, K, freq)

        # Time masks
        for _ in range(self.n_time_masks):
            if self.time_mask_param <= 0 or T <= 1:
                break
            t = np.random.randint(0, min(self.time_mask_param, T) + 1)
            if t == 0:
                continue
            t0 = np.random.randint(0, max(T - t + 1, 1))
            x[t0 : t0 + t, :, :] = 0

        # Freq masks
        for _ in range(self.n_freq_masks):
            if self.freq_mask_param <= 0 or freq <= 1:
                break
            f = np.random.randint(0, min(self.freq_mask_param, freq) + 1)
            if f == 0:
                continue
            f0 = np.random.randint(0, max(freq - f + 1, 1))
            x[:, :, f0 : f0 + f] = 0

        return x.reshape_as(tensor)


# -------------------------
# New raw-domain augmentations
# -------------------------

@dataclass
class RandomGain:
    """Multiply signal by a random gain.

    Can apply one gain per channel, per band, or globally depending on dims.

    Args:
        min_gain, max_gain: uniform range.
        per_channel: if True, sample independent gain per channel (last dim).
        per_band: if True and tensor has band dim, sample independent gain per band.
        band_dim: band dimension index (default: 1 as used by ToTensor()).
        channel_dim: channel dimension index (default: -1).
    """
    min_gain: float = 0.8
    max_gain: float = 1.25
    per_channel: bool = True
    per_band: bool = True
    band_dim: int = 1
    channel_dim: int = -1

    def __post_init__(self) -> None:
        assert self.min_gain > 0 and self.max_gain > 0
        assert self.max_gain >= self.min_gain

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor
        device = x.device
        dtype = x.dtype

        # Build broadcastable gain shape.
        gain_shape = [1] * x.ndim
        if self.per_band and x.ndim > self.band_dim:
            gain_shape[self.band_dim] = x.shape[self.band_dim]
        if self.per_channel:
            gain_shape[self.channel_dim] = x.shape[self.channel_dim]

        g = torch.empty(gain_shape, device=device, dtype=dtype).uniform_(
            float(self.min_gain), float(self.max_gain)
        )
        return x * g


@dataclass
class AdditiveGaussianNoise:
    """Add zero-mean Gaussian noise with random sigma.

    Args:
        min_sigma, max_sigma: sigma range (in same units as signal).
        per_channel: if True, sample sigma per channel; else global sigma.
        channel_dim: channel dimension index.
    """
    min_sigma: float = 0.0
    max_sigma: float = 0.02
    per_channel: bool = False
    channel_dim: int = -1

    def __post_init__(self) -> None:
        assert self.min_sigma >= 0 and self.max_sigma >= 0
        assert self.max_sigma >= self.min_sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.max_sigma == 0:
            return tensor

        x = tensor
        device = x.device
        dtype = x.dtype

        if self.per_channel:
            sigma_shape = [1] * x.ndim
            sigma_shape[self.channel_dim] = x.shape[self.channel_dim]
            sigma = torch.empty(sigma_shape, device=device, dtype=dtype).uniform_(
                float(self.min_sigma), float(self.max_sigma)
            )
        else:
            s = float(np.random.uniform(self.min_sigma, self.max_sigma))
            sigma = torch.tensor(s, device=device, dtype=dtype)

        noise = torch.randn_like(x) * sigma
        return x + noise


@dataclass
class RandomBurstNoise:
    """Inject short burst artifacts (additive) to simulate motion/cable noise.

    For each sample, with probability p, add 1..max_bursts bursts. Each burst:
    choose random start time and duration, add Gaussian noise with std in range.

    Args:
        p: probability to apply.
        max_bursts: maximum number of bursts when applied.
        min_duration, max_duration: burst length in timesteps.
        min_sigma, max_sigma: noise sigma range for burst.
    """
    p: float = 0.15
    max_bursts: int = 2
    min_duration: int = 10
    max_duration: int = 80
    min_sigma: float = 0.02
    max_sigma: float = 0.08

    def __post_init__(self) -> None:
        assert 0 <= self.p <= 1
        assert self.max_bursts >= 1
        assert 1 <= self.min_duration <= self.max_duration
        assert self.min_sigma >= 0 and self.max_sigma >= 0
        assert self.max_sigma >= self.min_sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return tensor

        x = tensor.clone()
        T = x.shape[0]
        if T <= 1:
            return x

        num = np.random.randint(1, self.max_bursts + 1)
        for _ in range(num):
            dur = int(np.random.randint(self.min_duration, self.max_duration + 1))
            dur = min(dur, T)
            start = int(np.random.randint(0, max(T - dur + 1, 1)))
            sigma = float(np.random.uniform(self.min_sigma, self.max_sigma))
            x[start : start + dur] = x[start : start + dur] + torch.randn_like(
                x[start : start + dur]
            ) * sigma
        return x


@dataclass
class ChannelDropout:
    """Randomly drop (zero) entire channels.

    Args:
        p: probability per channel to drop.
        channel_dim: channel dimension.
        value: fill value (0 is typical).
    """
    p: float = 0.05
    channel_dim: int = -1
    value: float = 0.0

    def __post_init__(self) -> None:
        assert 0 <= self.p <= 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.p == 0:
            return tensor

        x = tensor.clone()
        C = x.shape[self.channel_dim]
        mask = torch.rand((C,), device=x.device) >= self.p  # keep-mask
        # reshape mask to broadcast
        shape = [1] * x.ndim
        shape[self.channel_dim] = C
        mask = mask.reshape(shape)
        x = x * mask + (1.0 - mask.to(x.dtype)) * float(self.value)
        return x


@dataclass
class TemporalChannelDropout:
    """Randomly create brief flatlines/dropouts in random channels.

    Args:
        p: probability to apply.
        num_drops: number of dropout segments to apply.
        min_duration, max_duration: duration range in timesteps.
        channel_dim: channel dimension.
        value: fill value.
    """
    p: float = 0.2
    num_drops: int = 2
    min_duration: int = 10
    max_duration: int = 120
    channel_dim: int = -1
    value: float = 0.0

    def __post_init__(self) -> None:
        assert 0 <= self.p <= 1
        assert self.num_drops >= 1
        assert 1 <= self.min_duration <= self.max_duration

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return tensor

        x = tensor.clone()
        T = x.shape[0]
        C = x.shape[self.channel_dim]
        if T <= 1 or C <= 0:
            return x

        for _ in range(self.num_drops):
            ch = int(np.random.randint(0, C))
            dur = int(np.random.randint(self.min_duration, self.max_duration + 1))
            dur = min(dur, T)
            start = int(np.random.randint(0, max(T - dur + 1, 1)))

            # Build slicer
            slc = [slice(None)] * x.ndim
            slc[0] = slice(start, start + dur)
            slc[self.channel_dim] = slice(ch, ch + 1)
            x[tuple(slc)] = float(self.value)

        return x


@dataclass
class RandomTimeWarp:
    """Mild time-warp by resampling along the time dimension then cropping/padding.

    This keeps tensor rank and non-time dims unchanged.

    Args:
        p: probability to apply.
        min_rate, max_rate: resampling factor range. <1 slows down, >1 speeds up.
        mode: interpolation mode ('linear' recommended).
    """
    p: float = 0.2
    min_rate: float = 0.95
    max_rate: float = 1.05
    mode: str = "linear"

    def __post_init__(self) -> None:
        assert 0 <= self.p <= 1
        assert self.min_rate > 0 and self.max_rate > 0
        assert self.max_rate >= self.min_rate

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return tensor

        x = tensor
        T = x.shape[0]
        if T <= 2:
            return x

        rate = float(np.random.uniform(self.min_rate, self.max_rate))
        new_T = max(int(round(T * rate)), 2)

        # Interpolate along time: reshape to (N, C, T) for interpolate, with N=prod(other dims)
        y = x.movedim(0, -1)  # (..., T)
        y_flat = y.reshape(-1, 1, T)  # (K, 1, T)

        y_rs = torch.nn.functional.interpolate(
            y_flat,
            size=new_T,
            mode=self.mode,
            align_corners=False if self.mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
        )  # (K,1,new_T)
        y_rs = y_rs.reshape(*y.shape[:-1], new_T)  # (..., new_T)
        y_rs = y_rs.movedim(-1, 0)  # (new_T, ...)

        # Crop/pad back to T
        if new_T > T:
            start = int(np.random.randint(0, new_T - T + 1))
            y_out = y_rs[start : start + T]
        elif new_T < T:
            pad = T - new_T
            # Randomly distribute padding left/right
            left = int(np.random.randint(0, pad + 1))
            right = pad - left
            y_out = torch.nn.functional.pad(
                y_rs.movedim(0, -1),  # (..., new_T)
                (left, right),
                mode="replicate",
            ).movedim(-1, 0)
        else:
            y_out = y_rs

        return y_out


# -------------------------
# New spectrogram-domain augmentation
# -------------------------

@dataclass
class RandomFrequencyEQ:
    """Apply a smooth random frequency response curve to spectrogram features.

    Multiplies the spectrogram (in log domain) by adding a smooth offset over freq:
        logspec' = logspec + curve(freq)

    Args:
        p: probability to apply.
        max_db: maximum absolute offset in log10 domain (approx; treated as additive).
        n_control_points: number of control points to interpolate across frequency.
        freq_dim: frequency dimension index (default: -1 for (.., freq)).
    """
    p: float = 0.3
    max_db: float = 0.15
    n_control_points: int = 6
    freq_dim: int = -1

    def __post_init__(self) -> None:
        assert 0 <= self.p <= 1
        assert self.max_db >= 0
        assert self.n_control_points >= 2

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return tensor

        x = tensor
        F = x.shape[self.freq_dim]
        if F <= 1:
            return x

        device = x.device
        dtype = x.dtype

        # Sample smooth curve at control points then interpolate to F.
        cp = torch.empty((self.n_control_points,), device=device, dtype=dtype).uniform_(
            -float(self.max_db), float(self.max_db)
        )
        cp = cp[None, None, :]  # (1,1,P)
        curve = torch.nn.functional.interpolate(
            cp,
            size=F,
            mode="linear",
            align_corners=True,
        ).reshape(F)  # (F,)

        # Reshape curve for broadcasting
        shape = [1] * x.ndim
        shape[self.freq_dim] = F
        curve = curve.reshape(shape)

        return x + curve