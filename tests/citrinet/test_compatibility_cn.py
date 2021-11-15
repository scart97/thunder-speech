# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from urllib.error import HTTPError

from tests.utils import mark_slow
from thunder.citrinet.compatibility import CitrinetCheckpoint, load_citrinet_checkpoint


@mark_slow
def test_can_load_weights():
    # Download small citrinet while testing
    try:
        load_citrinet_checkpoint(CitrinetCheckpoint.stt_en_citrinet_256)
    except HTTPError:
        return
