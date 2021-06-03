from urllib.error import HTTPError

import pytest

import pytorch_lightning as pl
import torch
import torchaudio
from torchaudio.datasets.utils import download_url

from tests.utils import mark_slow, requirescuda
from thunder.citrinet.module import CitrinetCheckpoint, CitrinetModule
from thunder.data.datamodule import ManifestDatamodule
from thunder.utils import get_default_cache_folder


@mark_slow
def test_expected_prediction_from_pretrained_model():
    try:
        folder = get_default_cache_folder()
        download_url(
            "https://github.com/fastaudio/10_Speakers_Sample/raw/76f365de2f4d282ec44450d68f5b88de37b8b7ad/train/f0001_us_f0001_00001.wav",
            download_folder=str(folder),
            filename="f0001_us_f0001_00001.wav",
            resume=True,
        )
        module = CitrinetModule.load_from_nemo(
            checkpoint_name=CitrinetCheckpoint.stt_en_citrinet_256
        )

        audio, sr = torchaudio.load(folder / "f0001_us_f0001_00001.wav")
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
        module = CitrinetModule.load_from_nemo(
            checkpoint_name=CitrinetCheckpoint.stt_en_citrinet_256
        )
    except HTTPError:
        return
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


def test_script_module():
    try:
        module = CitrinetModule.load_from_nemo(
            checkpoint_name=CitrinetCheckpoint.stt_en_citrinet_256
        )
    except HTTPError:
        return
    module_script = torch.jit.script(module)
    x = torch.randn(10, 1337)
    out1 = module.predict(x)[0]
    out2 = module_script.predict(x)[0]
    assert out1 == out2


def test_try_to_load_without_parameters_raises_error():
    with pytest.raises(ValueError):
        CitrinetModule.load_from_nemo()


def test_change_vocab():
    try:
        module = CitrinetModule.load_from_nemo(
            checkpoint_name=CitrinetCheckpoint.stt_en_citrinet_256
        )
    except HTTPError:
        return
    module.change_vocab(["a", "b", "c"])
    assert module.hparams.initial_vocab_tokens == ["a", "b", "c"]
    # comparing to 10 to account for the 3 initial tokens plus
    # the few special tokens automatically added.
    assert len(module.text_transform.vocab) < 10
    assert module.decoder.out_channels < 10
