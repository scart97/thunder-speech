from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError

import torch
from torch import nn

from thunder.quartznet.compatibility import get_quartznet, read_config
from thunder.quartznet.model import (
    Quartznet5x5_encoder,
    Quartznet15x5_encoder,
    Quartznet_decoder,
)


def test_can_open_quartznet():
    # Quartznet 5x5 is small (25mb), so it can be downloaded while testing.
    try:
        with TemporaryDirectory() as tmpdir:
            encoder, decoder = get_quartznet("QuartzNet5x5LS-En", tmpdir)
            assert isinstance(encoder, nn.Module)
            assert isinstance(decoder, nn.Module)
    except HTTPError:
        return


def test_create_from_manifest():
    path = Path("tests/nemo_config_samples")
    for cfg in path.glob("*.yaml"):
        encoder, decoder = read_config(cfg)
        assert isinstance(encoder, nn.Module)
        assert isinstance(decoder, nn.Module)
        x = torch.randn(10, 64, 1337)
        out = encoder(x)
        out2 = decoder(out)
        assert out.shape[0] == x.shape[0]
        assert not torch.isnan(out).any()
        assert out2.shape[0] == x.shape[0]
        assert not torch.isnan(out2).any()

        if "Net5x5" in cfg.name:
            encoder2 = Quartznet5x5_encoder(64)
            encoder2.load_state_dict(encoder.state_dict())
        else:
            encoder2 = Quartznet15x5_encoder(64)
            encoder2.load_state_dict(encoder.state_dict())
        decoder2 = Quartznet_decoder(1024, 28)
        decoder2.load_state_dict(decoder[0].state_dict())
