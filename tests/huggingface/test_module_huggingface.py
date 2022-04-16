# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97


from urllib.error import HTTPError

import pytorch_lightning as pl
import torch
import torchaudio

from tests.utils import mark_slow, requirescuda
from thunder.data.datamodule import ManifestDatamodule
from thunder.huggingface.compatibility import prepare_scriptable_wav2vec
from thunder.registry import load_pretrained


@mark_slow
@requirescuda
def test_dev_run_train(sample_manifest):
    module = load_pretrained("facebook/wav2vec2-base-960h")
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
        fast_dev_run=True, logger=False, enable_checkpointing=False, gpus=-1
    )
    trainer.fit(module, datamodule=data)


@mark_slow
def test_expected_prediction_from_pretrained_model(sample_audio):
    try:
        # Preparing data and model
        module = load_pretrained("facebook/wav2vec2-base-960h")
        audio, sr = torchaudio.load(sample_audio)
        assert sr == 16000

        output = module.predict(audio)
        expected = "THE WORLD NEEDS OPPORTUNITIES FOR NEW LEADERS AND NEW IDEAS"
        assert output[0].strip() == expected
    except HTTPError:
        return


@mark_slow
def test_script_module():
    module = load_pretrained("facebook/wav2vec2-base-960h")
    module.eval()
    torchaudio_module = load_pretrained("facebook/wav2vec2-base-960h")
    torchaudio_module.eval()
    torchaudio_module = prepare_scriptable_wav2vec(torchaudio_module)
    scripted = torch.jit.script(torchaudio_module)

    fake_input = torch.randn(1, 16000)
    fake_transcription = module.predict(fake_input)
    torchaudio_transcription = torchaudio_module.predict(fake_input)
    scripted_transcription = scripted.predict(fake_input)

    assert fake_transcription[0] == torchaudio_transcription[0]
    assert fake_transcription[0] == scripted_transcription[0]


@mark_slow
def test_quantized_script_module():
    module = load_pretrained("facebook/wav2vec2-base-960h")
    module.eval()
    torchaudio_module = load_pretrained("facebook/wav2vec2-base-960h")
    torchaudio_module = prepare_scriptable_wav2vec(torchaudio_module, quantized=True)
    torchaudio_module.eval()
    scripted = torch.jit.script(torchaudio_module)

    fake_input = torch.randn(1, 16000)
    fake_transcription = module.predict(fake_input)
    torchaudio_transcription = scripted.predict(fake_input)

    assert fake_transcription[0] == torchaudio_transcription[0]
