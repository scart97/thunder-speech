"""Helper functions used by the speech dataloaders.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from typing import Iterable, List, Tuple

import numpy as np
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler


def asr_collate(samples: List[Tuple[Tensor, str]]) -> Tuple[Tensor, Tensor, List[str]]:
    """Function that collect samples and adds padding.

    Args:
        samples : Samples produced by dataloader

    Returns:
        Tuple containing padded audios, audio lengths (normalized to 0.0 <-> 1.0 range) and the list of corresponding transcriptions in that order.
    """
    samples = sorted(samples, key=lambda sample: sample[0].size(-1), reverse=True)
    padded_audios = pad_sequence([s[0].squeeze() for s in samples], batch_first=True)
    audio_lengths = Tensor([s[0].size(-1) for s in samples])
    audio_lengths = audio_lengths / audio_lengths.max()  # Normalize by max length
    texts = [s[1] for s in samples]

    return (padded_audios, audio_lengths, texts)


class BucketingSampler(Sampler):
    def __init__(self, data_source: Iterable, batch_size: int = 16):
        """Samples batches assuming they are in order of size to batch
        similarly sized samples together

        Args:
            data_source : Source of elements already sorted by length.
            batch_size : Number of elements to batch.
        """
        super().__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        np.random.shuffle(self.bins)
        for ids in self.bins:
            yield ids

    def __len__(self):
        return len(self.bins)
