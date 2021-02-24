# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97


import pytest

import torch

from thunder.librosa_compat import create_fb_matrix


@pytest.mark.parametrize("N_FFT", [256, 512, 1024])
@pytest.mark.parametrize("SR", [8000, 16000, 22000, 44100])
@pytest.mark.parametrize("n_mels", [32, 40, 64, 80, 96, 128])
@pytest.mark.parametrize("htk", [True, False])
@pytest.mark.filterwarnings("ignore")
def test_mel_filters_compatibility(N_FFT, SR, n_mels, htk):
    librosa = pytest.importorskip("librosa")

    lib = librosa.filters.mel(SR, N_FFT, n_mels=n_mels, htk=htk)
    lib = torch.tensor(lib).T
    tor = create_fb_matrix(
        int(1 + N_FFT // 2),
        n_mels=n_mels,
        sample_rate=SR,
        f_min=0,
        f_max=SR / 2,
        norm="slaney",
        htk=htk,
    )
    assert torch.allclose(tor, lib, atol=1e-6)


def test_create_fb_matrix_norm_error():
    with pytest.raises(ValueError):
        create_fb_matrix(
            10,
            n_mels=10,
            sample_rate=10,
            f_min=0,
            f_max=10,
            norm="this_should_cause_error",
            htk=True,
        )


def test_create_fb_matrix_warning():
    with pytest.warns(UserWarning):
        create_fb_matrix(
            int(1 + 128 // 2),
            n_mels=128,
            sample_rate=16000,
            f_min=0,
            f_max=16000 / 2,
            norm="slaney",
            htk=True,
        )
