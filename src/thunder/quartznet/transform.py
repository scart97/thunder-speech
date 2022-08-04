"""Functionality to transform the audio input in the same way
that the Quartznet model expects it.
"""

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = [
    "FeatureBatchNormalizer",
    "DitherAudio",
    "PreEmphasisFilter",
    "PowerSpectrum",
    "MelScale",
    "FilterbankFeatures",
]

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torchaudio.functional import melscale_fbanks

from thunder.blocks import (
    Masked,
    MultiSequential,
    convolution_stft,
    lengths_to_mask,
    normalize_tensor,
)
from thunder.quartznet.spec_augment import SpecAugment, SpecCutout


class FeatureBatchNormalizer(nn.Module):
    def __init__(self):
        """Normalize batch at the feature dimension."""
        super().__init__()
        self.div_guard = 1e-5

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, features, time)
            lengths: corresponding length of each element in the input tensor.
        """
        # https://github.com/pytorch/pytorch/issues/45208
        # https://github.com/pytorch/pytorch/issues/44768
        with torch.no_grad():
            mask = lengths_to_mask(lengths, x.shape[-1])
            return (
                normalize_tensor(x, mask.unsqueeze(1), div_guard=self.div_guard),
                lengths,
            )


class DitherAudio(nn.Module):
    def __init__(self, dither: float = 1e-5):
        """Add some dithering to the audio tensor.

        Note:
            From wikipedia: Dither is an intentionally applied
            form of noise used to randomize quantization error.

        Args:
            dither: Amount of dither to add.
        """
        super().__init__()
        self.dither = dither

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time)
        """
        if self.training:
            return x + (self.dither * torch.randn_like(x))
        else:
            return x


class PreEmphasisFilter(nn.Module):
    def __init__(self, preemph: float = 0.97):
        """Applies preemphasis filtering to the audio signal.
        This is a classic signal processing function to emphasise
        the high frequency portion of the content compared to the
        low frequency. It applies a FIR filter of the form:

        `y[n] = y[n] - preemph * y[n-1]`

        Args:
            preemph: Filter control factor.
        """
        super().__init__()
        self.preemph = preemph

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time)
        """
        return torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )


class PowerSpectrum(nn.Module):
    def __init__(
        self,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        n_fft: Optional[int] = None,
    ):
        """Calculates the power spectrum of the audio signal, following the same
        method as used in NEMO.

        Args:
            n_window_size: Number of elements in the window size.
            n_window_stride: Number of elements in the window stride.
            n_fft: Number of fourier features.

        Raises:
            ValueError: Raised when incompatible parameters are passed.
        """
        super().__init__()
        if n_window_size <= 0 or n_window_stride <= 0:
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        window_tensor = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window_tensor)
        # This way so that the torch.stft can be changed to the patched version
        # before scripting. That way it works correctly when the export option
        # doesnt support fft, like mobile or onnx.
        self.stft_func = torch.stft

    def get_sequence_length(self, lengths: torch.Tensor) -> torch.Tensor:
        seq_len = torch.floor(lengths / self.hop_length) + 1
        return seq_len.to(dtype=torch.long)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, time)
        """
        x = self.stft_func(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window.to(dtype=torch.float),
            return_complex=False,
        )

        # torch returns real, imag; so convert to magnitude
        x = torch.sqrt(x.pow(2).sum(-1))
        # get power spectrum
        x = x.pow(2.0)
        return x, self.get_sequence_length(lengths)


class MelScale(nn.Module):
    def __init__(
        self, sample_rate: int, n_fft: int, nfilt: int, log_scale: bool = True
    ):
        """Convert a spectrogram to Mel scale, following the default
        formula of librosa instead of the one used by torchaudio.
        Also converts to log scale.

        Args:
            sample_rate: Sampling rate of the signal
            n_fft: Number of fourier features
            nfilt: Number of output mel filters to use
            log_scale: Controls if the output should also be applied a log scale.
        """
        super().__init__()

        filterbanks = (
            melscale_fbanks(
                int(1 + n_fft // 2),
                n_mels=nfilt,
                sample_rate=sample_rate,
                f_min=0,
                f_max=sample_rate / 2,
                norm="slaney",
                mel_scale="slaney",
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )
        self.register_buffer("fb", filterbanks)
        self.log_scale = log_scale

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, features, time)
        """
        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features
        # We want to avoid taking the log of zero
        if self.log_scale:
            x = torch.log(x + 2**-24)
        return x


def FilterbankFeatures(
    sample_rate: int = 16000,
    n_window_size: int = 320,
    n_window_stride: int = 160,
    n_fft: int = 512,
    preemph: float = 0.97,
    nfilt: int = 64,
    dither: float = 1e-5,
    num_cutout_masks: int = 0,
    num_time_masks: int = 0,
    num_freq_masks: int = 0,
    mask_time_width: int = 50,
    mask_freq_width: int = 20,
) -> nn.Module:
    """Creates the Filterbank features used in the Quartznet model.

    Args:
        sample_rate: Sampling rate of the signal.
        n_window_size: Number of elements in the window size.
        n_window_stride: Number of elements in the window stride.
        n_fft: Number of fourier features.
        preemph: Preemphasis filtering control factor.
        nfilt: Number of output mel filters to use.
        dither: Amount of dither to add.
    Returns:
        Module that computes the features based on raw audio tensor.
    """
    if num_cutout_masks > 0 and (num_freq_masks + num_time_masks > 0):
        raise ValueError("Cutout and SpecAugment can't be used at the same time.")

    base_modules = [
        Masked(DitherAudio(dither=dither), PreEmphasisFilter(preemph=preemph)),
        PowerSpectrum(
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            n_fft=n_fft,
        ),
        Masked(MelScale(sample_rate=sample_rate, n_fft=n_fft, nfilt=nfilt)),
        FeatureBatchNormalizer(),
    ]

    if num_cutout_masks > 0:
        base_modules.append(
            Masked(
                SpecCutout(
                    rect_masks=num_cutout_masks,
                    time_width=mask_time_width,
                    freq_width=mask_freq_width,
                )
            )
        )
    if num_freq_masks + num_time_masks > 0:
        base_modules.append(
            Masked(
                SpecAugment(
                    time_masks=num_time_masks,
                    freq_masks=num_freq_masks,
                    time_width=mask_time_width,
                    freq_width=mask_freq_width,
                )
            )
        )

    return MultiSequential(*base_modules)


def patch_stft(filterbank: nn.Module) -> nn.Module:
    """This function applies a patch to the FilterbankFeatures to use instead a convolution
    layer based stft. That makes possible to export to onnx and use the scripted model
    directly on arm cpu's, inside mobile applications.

    Args:
        filterbank: the FilterbankFeatures layer to be patched

    Returns:
        Layer with the stft operation patched.
    """
    filterbank[1].stft_func = convolution_stft
    return filterbank
