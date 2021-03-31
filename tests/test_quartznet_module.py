from string import ascii_lowercase

import pytest

import pytorch_lightning as pl
import torch
import torchaudio
from torchaudio.datasets.utils import download_url

from tests.utils import mark_slow, requirescuda
from thunder.data.datamodule import ManifestDatamodule
from thunder.quartznet.module import QuartznetModule
from thunder.utils import get_default_cache_folder


@mark_slow
def test_expected_prediction_from_pretrained_model():
    # Loading the sample file
    folder = get_default_cache_folder()
    download_url(
        "https://github.com/fastaudio/10_Speakers_Sample/raw/76f365de2f4d282ec44450d68f5b88de37b8b7ad/train/f0001_us_f0001_00001.wav",
        download_folder=str(folder),
        filename="f0001_us_f0001_00001.wav",
        resume=True,
    )
    # Preparing data and model
    module = QuartznetModule.load_from_nemo(checkpoint_name="QuartzNet5x5LS-En")
    audio, sr = torchaudio.load(folder / "f0001_us_f0001_00001.wav")
    assert sr == 16000

    output = module.predict(audio)
    expected = "the world needs opportunities for new leaders and new ideas"
    assert output[0].strip() == expected


@mark_slow
@requirescuda
def test_dev_run_train(sample_manifest):
    module = QuartznetModule(list(ascii_lowercase))
    data = ManifestDatamodule(
        train_manifest=sample_manifest,
        val_manifest=sample_manifest,
        test_manifest=sample_manifest,
    )
    trainer = pl.Trainer(
        fast_dev_run=True, logger=None, checkpoint_callback=None, gpus=-1
    )
    trainer.fit(module, datamodule=data)


def test_script_module():
    module = QuartznetModule(list(ascii_lowercase))
    module_script = torch.jit.script(module)
    x = torch.randn(10, 1337)
    out1 = module.predict(x)[0]
    out2 = module_script.predict(x)[0]
    assert out1 == out2


def test_try_to_load_without_parameters_raises_error():
    with pytest.raises(ValueError):
        QuartznetModule.load_from_nemo()
