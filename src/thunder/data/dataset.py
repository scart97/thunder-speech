# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from typing import List

import torchaudio
from torch.utils.data import Dataset


class BaseSpeechDataset(Dataset):
    def __init__(self, items: List[str], force_mono: bool = True, sr: int = 16000):
        super().__init__()
        self.items = items
        self.sr = sr
        self.force_mono = force_mono

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.get_item(index)
        audio, sr = self.open_audio(item)
        audio = self.correct_audio(audio, sr)
        text = self.open_text(item)
        return audio, text

    def get_item(self, index):
        return self.items[index]

    def open_audio(self, item):
        return torchaudio.load(item)

    def correct_audio(self, audio, sr):
        if self.force_mono and (audio.shape[0] > 1):
            audio = audio.mean(0, keepdim=True)
        if self.sr != sr:
            tfm = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            audio = tfm(audio)
        return audio

    def open_text(self, item):
        raise NotImplementedError()


class ManifestSpeechDataset(BaseSpeechDataset):
    def open_audio(self, item):
        return torchaudio.load(item["audio_filepath"])

    def open_text(self, item):
        return item["text"]
