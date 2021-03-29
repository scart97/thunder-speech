import numpy as np
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler


def asr_collate(samples):
    "Function that collect samples and adds padding."
    samples = sorted(samples, key=lambda sample: sample[0].size(-1), reverse=True)
    padded_audios = pad_sequence([s[0].squeeze() for s in samples], batch_first=True)

    audio_lengths = Tensor([s[0].size(-1) for s in samples])
    audio_lengths = audio_lengths / audio_lengths.max()  # Normalize by max length

    texts = [s[1] for s in samples]

    return (padded_audios, audio_lengths, texts)


class BucketingSampler(Sampler):
    """
    Samples batches assuming they are in order of size to batch
    similarly sized samples together
    """

    def __init__(self, data_source, batch_size=1):
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
