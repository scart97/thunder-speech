from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError

import torch
from torch import nn

from thunder.jasper.model import get_jasper, read_config


def test_can_open_quartznet():
    # Quartznet 5x5 is small (25mb), so it can be downloaded while testing.
    try:
        with TemporaryDirectory() as tmpdir:
            encoder, decoder = get_jasper("QuartzNet5x5LS-En", tmpdir)
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
        x = torch.randn(10, encoder[0].inplanes, 1337)
        out = encoder(x)
        out2 = decoder(out)
        assert out.shape[0] == x.shape[0]
        assert out2.shape[0] == x.shape[0]
