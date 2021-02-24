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

import librosa
import torch
import torch.nn as nn


class FeatureBatchNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.div_guard = 1e-5

    def forward(self, x):
        x_mean = torch.zeros((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)

        for i in range(x.shape[0]):
            if x[i, :, :].shape[1] == 1:
                raise ValueError(
                    "normalize_batch received a tensor of length 1. This will result "
                    "in torch.std() returning nan"
                )
            x_mean[i, :] = x[i, :, :].mean(dim=1)
            x_std[i, :] = x[i, :, :].std(dim=1)
        # make sure x_std is not zero
        x_std += self.div_guard
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)


class DitherAudio(nn.Module):
    def __init__(self, dither=1e-5):
        super().__init__()
        self.dither = dither

    @torch.no_grad()
    def forward(self, x):
        if self.training:
            return x + self.dither * torch.randn_like(x)
        else:
            return x


class PreEmphasisFilter(nn.Module):
    def __init__(self, preemph=0.97):
        super().__init__()
        self.preemph = preemph

    @torch.no_grad()
    def forward(self, x):
        return torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )


class PowerSpectrum(nn.Module):
    def __init__(
        self,
        n_window_size=320,
        n_window_stride=160,
        n_fft=None,
        nfilt=64,
    ):
        super().__init__()
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        self.nfilt = nfilt
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        window_tensor = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window_tensor)

    @torch.no_grad()
    def forward(self, x):
        # disable autocast to get full range of stft values
        # with torch.cuda.amp.autocast(enabled=False):
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
    def __init__(self, sample_rate, n_fft, nfilt, log_scale=True):
        super().__init__()
        filterbanks = torch.tensor(
            librosa.filters.mel(
                sample_rate, n_fft, n_mels=nfilt, fmin=0.0, fmax=sample_rate / 2
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)
        self.log_scale = log_scale

    @torch.no_grad()
    def forward(self, x):
        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features
        # We want to avoid taking the log of zero
        if self.log_scale:
            x = torch.log(x + 2 ** -24)
        return x


def FilterbankFeatures(
    sample_rate=16000,
    n_window_size=320,
    n_window_stride=160,
    n_fft=512,
    preemph=0.97,
    nfilt=64,
    dither=1e-5,
    **kwargs,
):
    return nn.Sequential(
        DitherAudio(dither=dither),
        PreEmphasisFilter(preemph=preemph),
        PowerSpectrum(
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            n_fft=n_fft,
            nfilt=nfilt,
        ),
        MelScale(sample_rate=sample_rate, n_fft=n_fft, nfilt=nfilt),
        FeatureBatchNormalizer(),
    )
