"""
Implementation of data preprocessing transform compatible with the huggingface wav2vec2 one
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

from typing import Optional, Tuple

import torch
from torch import nn

from thunder.blocks import lengths_to_mask, normalize_tensor


class Wav2Vec2Preprocess(nn.Module):
    def __init__(
        self,
        div_guard: float = 1e-7,
        mask_input: bool = False,
    ):
        """Wav2Vec model preprocessing. It consists of normalizing the audio and optional mask.

        Args:
            div_guard: Guard value to prevent division by zero.
            mask_input: controls the use of masking in the input tensor.
        """
        super().__init__()
        self.div_guard = div_guard
        self.mask_input = mask_input

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies the normalization

        Args:
            audio: Audio tensor of shape [batch_size, time]
            audio_lengths: corresponding length of each element in the input tensor.

        Returns:
            Normalized audio tensor with same shape as input. Optionally the valid mask
        """
        attention_mask: Optional[torch.Tensor] = None
        if self.mask_input:
            attention_mask = lengths_to_mask(
                audio_lengths, max_length=audio.size(-1)
            ).int()

        return (
            normalize_tensor(audio, attention_mask, div_guard=self.div_guard),
            audio_lengths,
        )
