# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

"""
Speech recognition datasets
"""

__all__ = ["BaseSpeechDataset", "ManifestSpeechDataset"]

import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import torch
import torchaudio
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchaudio.functional import resample


class AudioFileLoader(nn.Module):
    def __init__(self, force_mono: bool = True, sample_rate: int = 16000):
        """Module containing the data loading and basic preprocessing.
        It's used internally by the datasets, but can be exported so
        that during inference time there's no code dependency.

        Args:
            force_mono: If true, convert all the loaded samples to mono.
            sample_rate: Sample rate used by the dataset. All of the samples that have different rate will be resampled.
        """
        super().__init__()
        self.force_mono = force_mono
        self.sample_rate = sample_rate

    @torch.jit.export
    def open_audio(self, item: str) -> Tuple[Tensor, int]:
        """Uses the data returned by get_item to open the audio

        Args:
            item: Data returned by get_item(index)

        Returns:
            Tuple containing the audio tensor with shape (channels, time), and the corresponding sample rate.
        """
        return torchaudio.load(item)

    @torch.jit.export
    def preprocess_audio(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Apply some base transforms to the audio, that fix silent problems.
        It transforms all the audios to mono (depending on class creation parameter),
        remove the possible DC bias present and then resamples the audios to a common
        sample rate.

        Args:
            audio: Audio tensor
            sample_rate: Sample rate

        Returns:
            Audio tensor after the transforms.
        """
        if self.force_mono and (audio.shape[0] > 1):
            audio = audio.mean(0, keepdim=True)

        # Removing the dc component from the audio
        # It happens when a faulty capture device introduce
        # an offset into the recorded waveform, and this can
        # cause problems with later transforms.
        # https://en.wikipedia.org/wiki/DC_bias
        audio = audio - audio.mean(1)

        if self.sample_rate != sample_rate:
            audio = resample(
                audio, orig_freq=int(sample_rate), new_freq=int(self.sample_rate)
            )
        return audio

    def forward(self, item: str) -> Tensor:
        """Opens audio item and do basic preprocessing

        Args:
            item: Path to the audio to be opened

        Returns:
            Audio tensor after preprocessing
        """
        audio, sample_rate = self.open_audio(item)
        return self.preprocess_audio(audio, sample_rate)


class BaseSpeechDataset(Dataset):
    def __init__(
        self, items: Sequence, force_mono: bool = True, sample_rate: int = 16000
    ):
        """This is the base class that implements the minimal functionality to have a compatible
        speech dataset, in a way that can be easily customized by subclassing.

        Args:
            items: Source of items in the dataset, sorted by audio duration. This can be a list of files, a pandas dataframe or any other iterable structure where you record your data.
            force_mono: If true, convert all the loaded samples to mono.
            sample_rate: Sample rate used by the dataset. All of the samples that have different rate will be resampled.
        """
        super().__init__()
        self.items = items
        self.loader = AudioFileLoader(force_mono, sample_rate)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        item = self.get_item(index)
        # Dealing with input
        audio, sr = self.open_audio(item)
        audio = self.preprocess_audio(audio, sr)
        # Dealing with output
        text = self.open_text(item)
        text = self.preprocess_text(text)

        return audio, text

    def all_outputs(self) -> List[str]:
        """Return a list with just the outputs for the whole dataset.
        Useful when creating the initial vocab tokens, or to train a
        language model.

        Returns:
            All of the outputs of the dataset, with the corresponding preprocessing applied.
        """
        outputs = []
        for index in range(len(self)):
            item = self.get_item(index)
            text = self.open_text(item)
            text = self.preprocess_text(text)
            outputs.append(text)
        return outputs

    def get_item(self, index: int) -> Any:
        """Get the item source specified by the index.

        Args:
            index: Indicates what item it needs to return information about.

        Returns:
            Whatever data necessary to open the audio and text corresponding to this index.
        """
        return self.items[index]

    def open_audio(self, item: Any) -> Tuple[Tensor, int]:
        """Uses the data returned by get_item to open the audio

        Args:
            item: Data returned by get_item(index)

        Returns:
            Tuple containing the audio tensor with shape (channels, time), and the corresponding sample rate.
        """
        return self.loader.open_audio(item)

    def preprocess_audio(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Apply some base transforms to the audio, that fix silent problems.
        It transforms all the audios to mono (depending on class creation parameter),
        remove the possible DC bias present and then resamples the audios to a common
        sample rate.

        Args:
            audio: Audio tensor
            sample_rate: Sample rate

        Returns:
            Audio tensor after the transforms.
        """
        return self.loader.preprocess_audio(audio, sample_rate)

    def open_text(self, item: Any) -> str:
        """Opens the transcription based on the data returned by get_item(index)

        Args:
            item: The data returned by get_item.

        Returns:
            The transcription corresponding to the item.
        """
        raise NotImplementedError()

    def preprocess_text(self, text: str) -> str:
        """Add here preprocessing steps to remove some common problems in the text.

        Args:
            text: Label text

        Returns:
            Label text after processing
        """
        return text


class ManifestSpeechDataset(BaseSpeechDataset):
    def __init__(self, file: Union[str, Path], force_mono: bool, sample_rate: int):
        """Dataset that loads from nemo manifest files.

        Args:
            file: Nemo manifest file.
            force_mono: If true, convert all the loaded samples to mono.
            sample_rate: Sample rate used by the dataset. All of the samples that have different rate will be resampled.
        """
        file = Path(file)
        # Reading from the manifest file
        items = [json.loads(line) for line in file.read_text().strip().splitlines()]
        super().__init__(items, force_mono=force_mono, sample_rate=sample_rate)

    def open_audio(self, item: dict) -> Tuple[Tensor, int]:
        return self.loader.open_audio(item["audio_filepath"])

    def open_text(self, item: dict) -> str:
        return item["text"]
