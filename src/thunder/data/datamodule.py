"""
Implements pytorch lightning's Datamodule for audio datasets.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["BaseDataModule", "ManifestDatamodule"]

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataloader_utils import asr_collate
from .dataset import BaseSpeechDataset, ManifestSpeechDataset


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 10,
        num_workers: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_dataset(self, split: str) -> BaseSpeechDataset:
        """Function to get the corresponding dataset to the specified split.
        This should be implemented by subclasses.

        Args:
            split: One of "train", "valid" or "test".

        Returns:
            The corresponding dataset.
        """
        raise NotImplementedError()

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = self.get_dataset(split="train")
            self.val_dataset = self.get_dataset(split="valid")
        if stage in (None, "test"):
            self.test_dataset = self.get_dataset(split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=asr_collate,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=asr_collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=asr_collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def steps_per_epoch(self) -> int:
        """Number of steps for each training epoch. Used for learning rate scheduling.

        Returns:
            Number of steps
        """
        return len(self.train_dataset) // self.batch_size


class ManifestDatamodule(BaseDataModule):
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        test_manifest: str,
        force_mono: bool = True,
        sample_rate: int = 16000,
        batch_size: int = 10,
        num_workers: int = 8,
    ):
        """Datamodule compatible with the NEMO manifest data format.

        Args:
            train_manifest: Training manifest file
            val_manifest: Validation manifest file
            test_manifest: Test manifest file
            force_mono: Check [`ManifestSpeechDataset`][thunder.data.dataset.ManifestSpeechDataset]
            sample_rate: Check [`ManifestSpeechDataset`][thunder.data.dataset.ManifestSpeechDataset]
            batch_size: Batch size used by dataloader
            num_workers: Number of workers used by dataloader
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.manifest_mapping = {
            "train": train_manifest,
            "valid": val_manifest,
            "test": test_manifest,
        }
        self.force_mono = force_mono
        self.sample_rate = sample_rate

    def get_dataset(self, split: str) -> ManifestSpeechDataset:
        return ManifestSpeechDataset(
            self.manifest_mapping[split], self.force_mono, self.sample_rate
        )
