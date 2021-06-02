# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

import torch
from torch import nn


class Wav2Vec2Preprocess(nn.Module):
    def __init__(self, div_guard: float = 1e-5):
        """Wav2Vec model preprocessing. It only consists of normalizing the audio.

        Args:
            div_guard : Guard value to prevent division by zero.
        """
        super().__init__()
        self.div_guard = div_guard

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Applies the normalization

        Args:
            audio : Audio tensor of shape [batch_size, time]

        Returns:
            Normalized audio tensor with same shape as input
        """
        mean = audio.mean(1, keepdim=True).detach()
        std = (audio.var(1, keepdim=True).detach() + self.div_guard).sqrt()
        return (audio - mean) / std
