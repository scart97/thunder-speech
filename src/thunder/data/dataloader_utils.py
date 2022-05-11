"""Helper functions used by the speech dataloaders.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["asr_collate"]

from typing import List, Tuple

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def asr_collate(samples: List[Tuple[Tensor, str]]) -> Tuple[Tensor, Tensor, List[str]]:
    """Function that collect samples and adds padding.

    Args:
        samples: Samples produced by dataloader

    Returns:
        Tuple containing padded audios, audio lengths and the list of corresponding transcriptions in that order.
    """
    samples = sorted(samples, key=lambda sample: sample[0].size(-1), reverse=True)
    padded_audios = pad_sequence([s[0].squeeze() for s in samples], batch_first=True)

    audio_lengths = Tensor([s[0].size(-1) for s in samples])

    texts = [s[1] for s in samples]

    return (padded_audios, audio_lengths, texts)
