"""Functionality to preprocess the audio input in the same way
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

# Copyright (c) 2021 scart97

import math
from typing import Optional

import torch
from torch import nn

from thunder.librosa_compat import create_fb_matrix


class FeatureBatchNormalizer(nn.Module):
    def __init__(self):
        """Normalize batch at the feature dimension."""
        super().__init__()
        self.div_guard = 1e-5

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, features, time)
        """

        seq_lens = seq_lens.to(dtype=torch.long)

        x_mean = torch.zeros((seq_lens.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_lens.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)

        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, : seq_lens[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_lens[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += self.div_guard
        out = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

        for i in range(x.shape[0]):
            out[i, :, seq_lens[i] :] = 0
        return out


class FinalPad(nn.Module):
    def __init__(self, pad_to: int = 16, pad_value: int = 0):
        """Pad spectrogram with `pad_value` to nearest multiple of `pad_to`

        Args:
            pad_to: input will be padded to a multiple of pad_to
            pad_value: value to pad with, default 0
        """
        super().__init__()
        self.pad_to = pad_to
        self.pad_value = pad_value

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, features, time)
        """
        # Todo: Refactor to be more clear, I just dont want to risk introducing a bug at this point
        pad_amt = x.size(-1) % self.pad_to
        if pad_amt != 0:
            x = nn.functional.pad(x, (0, self.pad_to - pad_amt), value=self.pad_value)
        return x


class DitherAudio(nn.Module):
    def __init__(self, dither: float = 1e-5):
        """Add some dithering to the audio tensor.

        Note:
            From wikipedia: Dither is an intentionally applied
            form of noise used to randomize quantization error.

        Args:
            dither : Amount of dither to add.
        """
        super().__init__()
        self.dither = dither

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, time)
        """
        torch.set_printoptions(threshold=10_000)
        if self.training:
            out = x + self.dither * torch.randn_like(x)
            return out
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
            preemph : Filter control factor.
        """
        super().__init__()
        self.preemph = preemph

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, time)
        """
        out = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        return out


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
            n_window_size : Number of elements in the window size.
            n_window_stride : Number of elements in the window stride.
            n_fft : Number of fourier features.

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

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, time)
        """

        with torch.cuda.amp.autocast(enabled=False):
            x = torch.stft(
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
        return x


class MelScale(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int, nfilt: int, log_scale: bool = True):
        """Convert a spectrogram to Mel scale, following the default
        formula of librosa instead of the one used by torchaudio.
        Also converts to log scale.

        Args:
            sample_rate : Sampling rate of the signal
            n_fft : Number of fourier features
            nfilt : Number of output mel filters to use
            log_scale : Controls if the output should also be applied a log scale.
        """
        super().__init__()
        filterbanks = (
            create_fb_matrix(
                int(1 + n_fft // 2),
                n_mels=nfilt,
                sample_rate=sample_rate,
                f_min=0,
                f_max=sample_rate / 2,
                norm="slaney",
                htk=False,
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
            x : Tensor of shape (batch, features, time)
        """
        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features
        # We want to avoid taking the log of zero
        if self.log_scale:
            x = torch.log(x + 2 ** -24)
        return x


class FilterbankFeatures(nn.Module):
    """Creates the Filterbank features used in the Quartznet model.

    Args:
        sample_rate : Sampling rate of the signal
        n_window_size : Number of elements in the window size.
        n_window_stride : Number of elements in the window stride.
        n_fft : Number of fourier features.
        preemph : Preemphasis filtering control factor.
        nfilt : Number of output mel filters to use
        dither : Amount of dither to add.

    Returns:
        Module that computes the features based on raw audio tensor.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        n_fft: int = 512,
        preemph: float = 0.97,
        nfilt: int = 64,
        dither: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.apply_dither = DitherAudio(dither=dither)
        self.apply_preemph = PreEmphasisFilter(preemph=preemph)
        self.apply_power_spec = PowerSpectrum(
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            n_fft=n_fft,
        )
        self.apply_mel = MelScale(sample_rate=sample_rate, n_fft=n_fft, nfilt=nfilt)
        self.normalize = FeatureBatchNormalizer()

        self.final_pad = FinalPad()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        x = self.apply_dither(x)
        x = self.apply_preemph(x)
        x = self.apply_power_spec(x)
        x = self.apply_mel(x)
        x = self.normalize(x, seq_lens)
        x = self.final_pad(x)
        return x
