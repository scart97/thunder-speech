from string import ascii_lowercase
from urllib.error import HTTPError

import pytest

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import OneCycleLR

from tests.utils import mark_slow, requirescuda
from thunder.blocks import conv1d_decoder, linear_decoder
from thunder.data.datamodule import ManifestDatamodule
from thunder.finetune import FinetuneCTCModule


@mark_slow
@requirescuda
def test_dev_run_train(sample_manifest):
    try:
        module = FinetuneCTCModule(
            "stt_en_citrinet_256",
            decoder_class=conv1d_decoder,
            tokens=list(ascii_lowercase),
            text_kwargs={"blank_token": "BLANK"},
        )
    except HTTPError:
        return

    # Tokens should be ascii_lowercase + blank
    assert module.text_transform.num_tokens == len(list(ascii_lowercase)) + 1
    assert module.text_transform.vocab.blank_token == "BLANK"

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
@requirescuda
def test_special_scheduler_args(sample_manifest):
    try:
        module = FinetuneCTCModule(
            "stt_en_citrinet_256",
            decoder_class=conv1d_decoder,
            tokens=list(ascii_lowercase),
            text_kwargs={"blank_token": "BLANK"},
            lr_scheduler_class=OneCycleLR,
            lr_scheduler_kwargs={"max_lr": 1e-3, "total_steps_arg": "total_steps"},
        )
    except HTTPError:
        return

    # Tokens should be ascii_lowercase + blank
    assert module.text_transform.num_tokens == len(list(ascii_lowercase)) + 1
    assert module.text_transform.vocab.blank_token == "BLANK"

    data = ManifestDatamodule(
        train_manifest=sample_manifest,
        val_manifest=sample_manifest,
        test_manifest=sample_manifest,
        num_workers=0,
    )
    trainer = pl.Trainer(
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        accelerator="gpu",
        devices=-1,
    )
    trainer.fit(module, datamodule=data)


@mark_slow
def test_missing_parameter():
    with pytest.raises(ValueError, match="tokens"):
        FinetuneCTCModule(
            "stt_en_citrinet_256",
            tokens=list(ascii_lowercase),
        )
    with pytest.raises(ValueError, match="decoder"):
        FinetuneCTCModule(
            "stt_en_citrinet_256",
            decoder_class=conv1d_decoder,
        )


@mark_slow
def test_compatibility_between_decoders():
    # conv model with linear decoder
    module_conv = FinetuneCTCModule(
        "stt_en_citrinet_256",
        decoder_class=linear_decoder,
        tokens=list(ascii_lowercase),
        decoder_kwargs={"decoder_dropout": 0.0},
    )
    # hf model with conv decoder
    module_hf = FinetuneCTCModule(
        "facebook/wav2vec2-base-960h",
        decoder_class=conv1d_decoder,
        tokens=list(ascii_lowercase),
    )
    test_input = torch.rand(1, 1337)
    module_conv.predict(test_input)
    module_hf.predict(test_input)
