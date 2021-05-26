# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError

import torch
from torchaudio.datasets.utils import extract_archive

from tests.utils import mark_slow
from thunder.quartznet.blocks import Quartznet_decoder, Quartznet_encoder
from thunder.quartznet.compatibility import (
    NemoCheckpoint,
    download_checkpoint,
    load_quartznet_weights,
    read_params_from_config,
)
from thunder.quartznet.transform import FilterbankFeatures


@mark_slow
def test_can_load_weights():
    # Quartznet 5x5 is small (25mb), so it can be downloaded while testing.
    try:

        cfg = download_checkpoint(NemoCheckpoint.QuartzNet5x5LS_En)
        with TemporaryDirectory() as extract_path:
            extract_path = Path(extract_path)
            extract_archive(str(cfg), extract_path)
            config_path = extract_path / "model_config.yaml"
            encoder_params, initial_vocab, _ = read_params_from_config(config_path)
            encoder = Quartznet_encoder(64, **encoder_params)
            decoder = Quartznet_decoder(len(initial_vocab) + 1)
            load_quartznet_weights(
                encoder, decoder, extract_path / "model_weights.ckpt"
            )
    except HTTPError:
        return


@mark_slow
def test_create_from_manifest():
    path = Path("tests/nemo_config_samples")
    for cfg in path.glob("*.yaml"):
        encoder_params, initial_vocab, preprocess_params = read_params_from_config(cfg)
        fb = FilterbankFeatures(**preprocess_params)
        encoder = Quartznet_encoder(64, **encoder_params)
        decoder = Quartznet_decoder(len(initial_vocab))

        x = torch.randn(10, 1337)
        feat = fb(x)
        lens = torch.randint(10, 100, (10,))
        out, _ = encoder(feat, lens)
        out2 = decoder(out)
        assert feat.shape[0] == x.shape[0]
        assert feat.shape[1] == 64
        assert out.shape[0] == x.shape[0]
        assert not torch.isnan(out).any()
        assert out2.shape[0] == x.shape[0]
        assert not torch.isnan(out2).any()

        if "Net5x5" in cfg.name:
            encoder2 = Quartznet_encoder(64)
            encoder2.load_state_dict(encoder.state_dict())
        else:
            encoder2 = Quartznet_encoder(64, repeat_blocks=3)
            encoder2.load_state_dict(encoder.state_dict())
