# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

import json
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataloader_utils import BucketingSampler, asr_collate
from .dataset import ManifestSpeechDataset


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        bs: int = 10,
        num_workers: int = 8,
        force_mono: bool = True,
        sr: int = 16000,
    ):
        super().__init__()
        self.bs = bs
        self.num_workers = num_workers
        self.force_mono = force_mono
        self.sr = sr

    def get_dataset(self, split: str):
        raise NotImplementedError()

    def setup(self, stage):
        self.train_dataset = self.get_dataset(split="train")
        self.sampler = BucketingSampler(self.train_dataset, batch_size=self.bs)
        self.val_dataset = self.get_dataset(split="valid")
        self.test_dataset = self.get_dataset(split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.sampler,
            collate_fn=asr_collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.bs,
            shuffle=False,
            collate_fn=asr_collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.bs,
            shuffle=False,
            collate_fn=asr_collate,
            num_workers=self.num_workers,
        )

    @property
    def steps_per_epoch(self):
        return len(self.sampler)


class ManifestDatamodule(BaseDataModule):
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        test_manifest: str,
        bs: int = 10,
        num_workers: int = 8,
        force_mono: bool = True,
        sr: int = 16000,
    ):
        super().__init__(
            bs=bs,
            num_workers=num_workers,
            force_mono=force_mono,
            sr=sr,
        )
        self.manifest_mapping = {
            "train": train_manifest,
            "valid": val_manifest,
            "test": test_manifest,
        }

    def get_dataset(self, split: str):
        file = Path(self.manifest_mapping[split])
        items = [json.loads(line) for line in file.read_text().strip().splitlines()]
        sorted_items = list(sorted(items, key=lambda x: x["duration"]))
        return ManifestSpeechDataset(sorted_items, self.force_mono, self.sr)
