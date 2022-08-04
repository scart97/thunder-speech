# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97


import pickle
from urllib.error import HTTPError

import pytest

import pytorch_lightning as pl
import torch
import torchaudio

from tests.utils import mark_slow, requirescuda
from thunder.data.datamodule import ManifestDatamodule
from thunder.huggingface.compatibility import prepare_scriptable_wav2vec
from thunder.registry import load_pretrained


@pytest.fixture(scope="session")
def wav2vec_base():
    return load_pretrained("facebook/wav2vec2-base-960h")


def _copy_model(model):
    return pickle.loads(pickle.dumps(model))


@mark_slow
@requirescuda
def test_dev_run_train(wav2vec_base, sample_manifest):
    module = _copy_model(wav2vec_base)
    # Lowercase the vocab just to run this test, as labeled text is lowercase
    module.text_transform.vocab.itos = [
        x.lower() for x in module.text_transform.vocab.itos
    ]

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
        devices=-1,
    )
    trainer.fit(module, datamodule=data)


@mark_slow
def test_expected_prediction_from_pretrained_model(wav2vec_base, sample_audio):
    try:
        # Preparing data and model
        audio, sr = torchaudio.load(sample_audio)
        assert sr == 16000

        output = wav2vec_base.predict(audio)
        expected = "THE WORLD NEEDS OPPORTUNITIES FOR NEW LEADERS AND NEW IDEAS"
        assert output[0].strip() == expected
    except HTTPError:
        return


@mark_slow
def test_script_module(wav2vec_base):
    torchaudio_module = _copy_model(wav2vec_base)
    torchaudio_module.eval()
    torchaudio_module = prepare_scriptable_wav2vec(torchaudio_module)
    scripted = torchaudio_module.to_torchscript()

    fake_input = torch.randn(1, 16000)
    fake_transcription = wav2vec_base.predict(fake_input)
    torchaudio_transcription = torchaudio_module.predict(fake_input)
    scripted_transcription = scripted.predict(fake_input)

    assert fake_transcription[0] == torchaudio_transcription[0]
    assert fake_transcription[0] == scripted_transcription[0]


@mark_slow
def test_quantized_script_module(wav2vec_base):
    torchaudio_module = _copy_model(wav2vec_base)
    torchaudio_module = prepare_scriptable_wav2vec(torchaudio_module, quantized=True)
    torchaudio_module.eval()
    scripted = torchaudio_module.to_torchscript()

    fake_input = torch.randn(1, 16000)
    fake_transcription = wav2vec_base.predict(fake_input)
    torchaudio_transcription = scripted.predict(fake_input)

    assert fake_transcription[0] == torchaudio_transcription[0]


@mark_slow
def test_pretrained_without_vocab():
    with pytest.warns(UserWarning, match="missing"):
        load_pretrained("facebook/wav2vec2-base-100k-voxpopuli")
