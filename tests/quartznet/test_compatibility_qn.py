# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path
from urllib.error import HTTPError

import torch

from tests.utils import mark_slow
from thunder.blocks import conv1d_decoder
from thunder.quartznet.blocks import QuartznetEncoder
from thunder.quartznet.compatibility import (
    QuartznetCheckpoint,
    load_components_from_quartznet_config,
    load_quartznet_checkpoint,
)


@mark_slow
def test_can_load_weights():
    # Quartznet 5x5 is small (25mb), so it can be downloaded while testing.
    try:
        load_quartznet_checkpoint(QuartznetCheckpoint.QuartzNet5x5LS_En)
    except HTTPError:
        return


@mark_slow
def test_create_from_manifest():
    path = Path("tests/nemo_config_samples")
    for cfg in path.glob("*.yaml"):
        encoder, fb, text_tfm = load_components_from_quartznet_config(cfg)
        decoder = conv1d_decoder(1024, text_tfm.num_tokens)

        x = torch.randn(10, 1337)
        lens = torch.Tensor([1000] * 10)
        feat, feat_lens = fb(x, lens)
        out, _ = encoder(feat, feat_lens)
        out2 = decoder(out)
        assert feat.shape[0] == x.shape[0]
        assert feat.shape[1] == 64
        assert out.shape[0] == x.shape[0]
        assert not torch.isnan(out).any()
        assert out2.shape[0] == x.shape[0]
        assert not torch.isnan(out2).any()

        if "Net5x5" in cfg.name:
            encoder2 = QuartznetEncoder()
            encoder2.load_state_dict(encoder.state_dict())
        else:
            encoder2 = QuartznetEncoder(repeat_blocks=3)
            encoder2.load_state_dict(encoder.state_dict())
