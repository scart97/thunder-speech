# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path

import pytest

import torch

from thunder.data.datamodule import ManifestDatamodule


@pytest.fixture
def single_process_loader(sample_manifest):
    dm = ManifestDatamodule(
        train_manifest=sample_manifest,
        val_manifest=sample_manifest,
        test_manifest=sample_manifest,
        num_workers=0,
        batch_size=10,
    )
    dm.prepare_data()
    dm.setup(None)
    return dm


@pytest.fixture
def multi_process_loader(sample_manifest):
    dm = ManifestDatamodule(
        train_manifest=sample_manifest,
        val_manifest=sample_manifest,
        test_manifest=sample_manifest,
        num_workers=12,
        batch_size=10,
    )
    dm.prepare_data()
    dm.setup(None)
    return dm


def test_fixture(sample_manifest: Path):
    assert sample_manifest.exists()
    assert len(sample_manifest.read_text()) > 0


def test_all_outputs(single_process_loader):
    dataset = single_process_loader.train_dataset
    outputs = dataset.all_outputs()
    assert len(outputs) == len(dataset)
    assert isinstance(outputs, list)
    assert outputs[0] == dataset[0][1]


@pytest.mark.parametrize("subset", ["train_dataset", "val_dataset", "test_dataset"])
def test_dataset_shape_and_type(single_process_loader, subset):
    dataset = getattr(single_process_loader, subset)
    sample = dataset[0]
    # one element is (audio_tensor, text)
    assert len(sample) == 2
    assert torch.is_tensor(sample[0])
    assert isinstance(sample[1], str)
    # The audio tensor is mono
    assert sample[0].ndim == 2
    assert sample[0].shape[0] == 1


@pytest.mark.parametrize("subset", ["train_dataset", "val_dataset", "test_dataset"])
def test_audio_scale(single_process_loader, subset):
    dataset = getattr(single_process_loader, subset)
    audio = dataset[0][0]
    for audio, _ in dataset:
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0
        assert (audio > 0.0).any()
        assert (audio < 0.0).any()


@pytest.mark.parametrize("subset", ["val_dataset", "test_dataset"])
def test_augmentation_disabled_val_test(single_process_loader, subset):
    dataset = getattr(single_process_loader, subset)
    for i in range(len(dataset)):
        audio1, text1 = dataset[i]
        audio2, text2 = dataset[i]
        assert torch.allclose(audio1, audio2)
        assert text1 == text2


def _can_load_data(loader):
    assert next(iter(loader.train_dataloader()))
    assert next(iter(loader.val_dataloader()))
    assert next(iter(loader.test_dataloader()))


def test_load_single_process(single_process_loader):
    _can_load_data(single_process_loader)


def test_load_multi_process(multi_process_loader):
    _can_load_data(multi_process_loader)
