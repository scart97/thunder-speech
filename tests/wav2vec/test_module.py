# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from string import ascii_lowercase

import pytest

import pytorch_lightning as pl
import torch

from tests.utils import mark_slow, requirescuda
from thunder.data.datamodule import ManifestDatamodule
from thunder.wav2vec.module import Wav2Vec2Module


@mark_slow
@requirescuda
def test_dev_run_train(sample_manifest):
    module = Wav2Vec2Module(list(ascii_lowercase))
    data = ManifestDatamodule(
        train_manifest=sample_manifest,
        val_manifest=sample_manifest,
        test_manifest=sample_manifest,
        num_workers=0,
    )
    trainer = pl.Trainer(
        fast_dev_run=True, logger=None, checkpoint_callback=None, gpus=-1
    )
    trainer.fit(module, datamodule=data)


@mark_slow
def test_predict():
    module = Wav2Vec2Module(list(ascii_lowercase))
    fake_input = torch.randn(1, 16000)
    fake_transcription = module.predict(fake_input)
    assert isinstance(fake_transcription, list)
    assert len(fake_transcription) == 1
    assert isinstance(fake_transcription[0], str)


@pytest.mark.xfail
@mark_slow
def test_script_module():
    module = Wav2Vec2Module(list(ascii_lowercase))
    torch.jit.script(module)
