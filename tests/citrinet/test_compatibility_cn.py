# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError

from torchaudio.datasets.utils import extract_archive

from tests.utils import mark_slow
from thunder.citrinet.blocks import Citrinet_encoder
from thunder.citrinet.compatibility import (
    CitrinetCheckpoint,
    read_params_from_config_citrinet,
)
from thunder.quartznet.blocks import Quartznet_decoder
from thunder.quartznet.compatibility import download_checkpoint, load_quartznet_weights


@mark_slow
def test_can_load_weights():
    # Download small citrinet while testing
    try:

        cfg = download_checkpoint(CitrinetCheckpoint.stt_en_citrinet_256)
        with TemporaryDirectory() as extract_path:
            extract_path = Path(extract_path)
            extract_archive(str(cfg), extract_path)
            config_path = extract_path / "model_config.yaml"
            encoder_params, initial_vocab, _ = read_params_from_config_citrinet(
                config_path
            )
            encoder = Citrinet_encoder(80, **encoder_params)
            decoder = Quartznet_decoder(len(initial_vocab) + 1, 640)
            load_quartznet_weights(
                encoder, decoder, extract_path / "model_weights.ckpt"
            )
    except HTTPError:
        return
