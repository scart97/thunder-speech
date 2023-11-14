# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from urllib.error import HTTPError
import pytest

import pytorch_lightning as pl
import torch
import torchaudio

from tests.utils import mark_slow, requirescuda
from thunder.data.datamodule import ManifestDatamodule
from thunder.registry import load_pretrained


@mark_slow
def test_expected_prediction_from_pretrained_model(sample_audio):
    # Loading the sample file
    try:
        # Preparing data and model
        module = load_pretrained("QuartzNet5x5LS_En")
        audio, sr = torchaudio.load(sample_audio)
        assert sr == 16000

        output = module.predict(audio)
        expected = "the world needs opportunities for new leaders and new ideas"
        assert output[0].strip() == expected
    except HTTPError:
        return


@mark_slow
@requirescuda
def test_dev_run_train(sample_manifest):
    try:
        module = load_pretrained("QuartzNet5x5LS_En")
    except HTTPError:
        return
    data = ManifestDatamodule(
        train_manifest=sample_manifest,
        val_manifest=sample_manifest,
        test_manifest=sample_manifest,
        num_workers=0,
    )
    trainer = pl.Trainer(
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        accelerator="gpu",
    )
    trainer.fit(module, datamodule=data)


@pytest.mark.xfail
@mark_slow
def test_script_module():
    try:
        module = load_pretrained("QuartzNet5x5LS_En")
    except HTTPError:
        return
    module_script = module.to_torchscript()
    x = torch.randn(10, 1337)
    out1 = module.predict(x)[0]
    out2 = module_script.predict(x)[0]
    assert out1 == out2
