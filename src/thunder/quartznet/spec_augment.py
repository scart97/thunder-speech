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

# Original file: https://github.com/NVIDIA/NeMo/blob/54e6f6ee688f09810d3e54661275fd5c8718db00/nemo/collections/asr/parts/spectr_augment.py

import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.functional import mask_along_axis


class SpecAugment(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
    """

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=10,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                for _ in range(self.time_masks):
                    x = mask_along_axis(x, self.time_width, 0.0, 2)

                for _ in range(self.freq_masks):
                    x = mask_along_axis(x, self.freq_width, 0.0, 1)

        return x


def _create_mask(specgram: Tensor, mask_param: int, axis: int):
    # modified from mask_along_axis in torchaudio
    # source: https://github.com/pytorch/audio/blob/54eb0991fae635c6586f7f1d6bf6080128fbff11/torchaudio/functional/functional.py#L870
    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (specgram.size(axis) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    mask = torch.arange(
        0, specgram.shape[axis], device=specgram.device, dtype=specgram.dtype
    )
    mask = (mask >= mask_start) & (mask < mask_end)
    if axis == 1:
        mask = mask.unsqueeze(-1)
    return mask


class SpecCutout(nn.Module):
    """
    Zeroes out(cuts) random rectangles in the spectrogram
    as described in (https://arxiv.org/abs/1708.04552).

    params:
    rect_masks - how many rectangular masks should be cut
    freq_width - maximum size of cut rectangles along the frequency dimension
    time_width - maximum size of cut rectangles along the time dimension
    """

    def __init__(self, rect_masks: int = 0, time_width: int = 5, freq_width: int = 20):
        super().__init__()
        self.rect_masks = rect_masks
        self.time_width = time_width
        self.freq_width = freq_width

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                for _ in range(self.rect_masks):
                    freq_mask = _create_mask(x, self.freq_width, 1)
                    time_mask = _create_mask(x, self.freq_width, 2)
                    x = x.masked_fill(freq_mask & time_mask, 0.0)
        return x
