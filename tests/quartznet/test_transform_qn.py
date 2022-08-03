# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97


import pytest

import torch
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, none, one_of

from tests.utils import (
    _test_batch_independence,
    _test_device_move,
    mark_slow,
    requirescuda,
)
from thunder.quartznet.spec_augment import SpecAugment
from thunder.quartznet.transform import (
    DitherAudio,
    FeatureBatchNormalizer,
    FilterbankFeatures,
    MelScale,
    PowerSpectrum,
    PreEmphasisFilter,
    patch_stft,
)


def test_normalize_preserve_shape():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    lens = torch.Tensor([1000] * 10)
    out, out_lens = norm(x, lens)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == x.shape[1]
    assert out.shape[2] == x.shape[2]
    assert torch.allclose(lens, out_lens)


def test_normalize_has_correct_mean_std():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    lens = torch.Tensor([1337] * 10)
    out, _ = norm(x, lens)
    # Testing for each element in the batch
    for xb in out:
        assert torch.allclose(xb[:, :].mean(), torch.zeros(1), atol=0.1)
        assert torch.allclose(xb[:, :].std(), torch.ones(1), atol=0.1)


@requirescuda
def test_normalize_device_move():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    lens = torch.Tensor([1000] * 10)
    _test_device_move(norm, (x, lens))


def test_normalize_script():
    norm = FeatureBatchNormalizer()
    norm.eval()
    x = torch.randn(10, 40, 1337)
    lens = torch.Tensor([1000] * 10)
    norm_script = torch.jit.script(norm)

    out1, lens1 = norm(x, lens)
    out2, lens2 = norm_script(x, lens)
    assert torch.allclose(out1, out2)
    assert torch.allclose(lens1, lens2)


@given(floats())
def test_dither_retains_shape(dither_magnitude):
    dither = DitherAudio(dither_magnitude)
    x = torch.randn(10, 1337)
    out = dither(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == x.shape[1]


@given(floats())
def test_dither_eval_mode_retain_input(dither_magnitude):
    dither = DitherAudio(dither_magnitude)
    dither.eval()
    x = torch.randn(10, 1337)
    out = dither(x)
    assert torch.allclose(out, x)


@given(floats(min_value=1e-6))
def test_dither_changes_input(dither_magnitude):
    dither = DitherAudio(dither_magnitude)
    x = torch.randn(10, 1337)
    out = dither(x)
    assert not torch.allclose(out, x)


@given(floats())
def test_dither_batch_independence(dither_magnitude):
    dither = DitherAudio(dither_magnitude)
    x = torch.randn(10, 1337)
    _test_batch_independence(dither, x)


@requirescuda
@given(floats())
def test_dither_device_move(dither_magnitude):
    dither = DitherAudio(dither_magnitude)
    x = torch.randn(10, 1337)
    _test_device_move(dither, x)


@given(floats())
def test_dither_script(dither_magnitude):
    dither = DitherAudio(dither_magnitude)
    dither.eval()
    x = torch.randn(10, 1337)
    dither_script = torch.jit.script(dither)

    assert torch.allclose(dither(x), dither_script(x))


preemph_params = given(floats(min_value=0.000001, max_value=0.999999999))


@preemph_params
def test_preemph_filter_retain_shape(preemph):
    filt = PreEmphasisFilter(preemph)
    x = torch.randn(10, 1337)
    out = filt(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == x.shape[1]


def test_preemph_filter_zero_gain():
    # With zero gain the output should be the input
    filt = PreEmphasisFilter(0.0)
    x = torch.randn(10, 1337)
    out = filt(x)
    assert torch.allclose(out, x)


@preemph_params
def test_preemph_filter_changes_input(preemph):
    filt = PreEmphasisFilter(preemph)
    x = torch.randn(10, 1337)
    out = filt(x)
    assert not torch.allclose(out, x)


@requirescuda
@preemph_params
def test_preemph_filter_device_move(preemph):
    filt = PreEmphasisFilter(preemph)
    x = torch.randn(10, 1337)
    _test_device_move(filt, x)


@preemph_params
def test_preemph_filter_script(preemph):
    filt = PreEmphasisFilter(preemph)
    filt.eval()
    x = torch.randn(10, 1337)
    filt_script = torch.jit.script(filt)

    assert torch.allclose(filt(x), filt_script(x), atol=1e-6)


powerspec_params = given(
    n_window_size=integers(min_value=16, max_value=128),
    n_window_stride=integers(min_value=8, max_value=64),
    n_fft=one_of(none(), integers(min_value=128, max_value=256)),
)


@powerspec_params
def test_powerspectrum_shape(**kwargs):
    spec = PowerSpectrum(**kwargs)
    x = torch.randn(10, 1337)
    lens = torch.Tensor([1337] * 10)
    out, out_lens = spec(x, lens)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == int(1 + spec.n_fft // 2)
    assert out.shape[2] == int(1 + x.shape[1] // spec.hop_length)
    assert (out_lens == out.shape[2]).all()
    assert len(out.shape) == 3


@given(n_window_size=integers(max_value=0), n_window_stride=integers(max_value=0))
def test_powerspec_raises(**kwargs):
    with pytest.raises(ValueError):
        PowerSpectrum(**kwargs)


@requirescuda
@powerspec_params
def test_powerspectrum_device_move(**kwargs):
    spec = PowerSpectrum(**kwargs)
    x = torch.randn(10, 1337)
    lens = torch.Tensor([1337] * 10)
    _test_device_move(spec, (x, lens))


@powerspec_params
def test_powerspectrum_script(**kwargs):
    spec = PowerSpectrum(**kwargs)
    spec.eval()
    x = torch.randn(10, 1337)
    lens = torch.Tensor([1337] * 10)
    spec_script = torch.jit.script(spec)

    out1, lens1 = spec(x, lens)
    out2, lens2 = spec_script(x, lens)
    assert torch.allclose(out1, out2)
    assert torch.allclose(lens1, lens2)


melscale_params = given(
    sample_rate=integers(min_value=8000, max_value=9000),
    n_fft=integers(min_value=500, max_value=512),
    nfilt=integers(min_value=60, max_value=64),
)


@melscale_params
def test_melscale_shape(**kwargs):
    mel = MelScale(**kwargs)
    x = torch.randn(10, int(1 + kwargs["n_fft"] // 2), 137).abs()
    out = mel(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == kwargs["nfilt"]
    assert out.shape[2] == x.shape[2]
    assert not torch.isnan(out).any()


@melscale_params
def test_melscale_zero_guard(**kwargs):
    mel = MelScale(**kwargs)
    x = torch.zeros((10, int(1 + kwargs["n_fft"] // 2), 137))
    out = mel(x)
    assert not torch.isnan(out).any()
    assert torch.isfinite(out).all()


@requirescuda
@melscale_params
def test_melscale_device_move(**kwargs):
    mel = MelScale(**kwargs)
    x = torch.randn(10, int(1 + kwargs["n_fft"] // 2), 137).abs()
    _test_device_move(mel, x)


@melscale_params
def test_melscale_script(**kwargs):
    mel = MelScale(**kwargs)
    mel.eval()
    x = torch.randn(10, int(1 + kwargs["n_fft"] // 2), 137).abs()
    mel_script = torch.jit.script(mel)

    assert torch.allclose(mel(x), mel_script(x))


filterbank_params = given(
    sample_rate=integers(8000, 9000),
    n_window_size=integers(min_value=16, max_value=128),
    n_window_stride=integers(min_value=8, max_value=64),
    n_fft=integers(min_value=500, max_value=512),
    preemph=floats(min_value=0.9, max_value=0.99),
    nfilt=integers(min_value=60, max_value=64),
    dither=floats(min_value=-1000.0, max_value=1000.0),
)


@filterbank_params
def test_filterbank_shape(**kwargs):
    fb = FilterbankFeatures(**kwargs)
    x = torch.randn(10, 1337)
    lens = torch.Tensor([1000] * 10)
    out, _ = fb(x, lens)
    assert out.shape[0] == x.shape[0]


def test_patch_stft_similar_output():
    fb = FilterbankFeatures()
    fb.eval()
    x = torch.randn(10, 1000)
    lens = torch.Tensor([1000] * 10)
    out1, _ = fb(x, lens)
    fb = patch_stft(fb)
    scripted = torch.jit.script(fb)
    out2, _ = scripted(x, lens)
    assert torch.allclose(out1, out2, atol=1e-3)


@requirescuda
@filterbank_params
@settings(deadline=None, max_examples=10)
def test_filterbank_device_move(**kwargs):
    fb = FilterbankFeatures(**kwargs)
    x = torch.randn(10, 1337)
    lens = torch.Tensor([1000] * 10)
    # Relaxed tolerance because of log operation
    # inside the Melscale
    _test_device_move(fb, (x, lens), atol=1e-3)


@mark_slow
@filterbank_params
@settings(deadline=None, max_examples=10)
def test_filterbank_script(**kwargs):
    fb = FilterbankFeatures(**kwargs)
    fb.eval()
    x = torch.randn(10, 1337)
    lens = torch.Tensor([1000] * 10)
    fb_script = torch.jit.script(fb)

    out1, lens1 = fb(x, lens)
    out2, lens2 = fb_script(x, lens)
    assert torch.allclose(out1, out2, atol=1e-3)
    assert torch.allclose(lens1, lens2)


def test_specaugment_freq():
    sa = SpecAugment(freq_masks=5)
    x = torch.ones((1, 100, 100))
    out = sa(x)
    idx = torch.argwhere(out[0] == 0.0)
    # All of the masks are complete
    assert len(idx) % 100 == 0
    # Have the full sequence 0-99 for every mask
    assert idx[:, 1].sum() % 4950 == 0


def test_specaugment_time():
    sa = SpecAugment(time_masks=5)
    x = torch.ones((1, 100, 100))
    out = sa(x)
    idx = torch.argwhere(out[0] == 0.0)
    # All of the masks are complete
    assert len(idx) % 100 == 0
    # Have the full sequence 0-99 for every mask
    assert idx[:, 0].sum() % 4950 == 0


specaugment_params = given(
    time_masks=integers(0, 50),
    freq_masks=integers(0, 50),
    time_width=integers(0, 50),
    freq_width=integers(0, 50),
)


@specaugment_params
def test_specaugment_shape(**kwargs):
    sa = SpecAugment(**kwargs)
    x = torch.ones((1, 100, 100))
    out = sa(x)
    assert out.shape[0] == x.shape[0]


@requirescuda
@specaugment_params
@settings(deadline=None, max_examples=10)
def test_specaugment_device_move(**kwargs):
    sa = SpecAugment(**kwargs)
    x = torch.ones((1, 100, 100))
    _test_device_move(sa, x)


@specaugment_params
@settings(deadline=None, max_examples=10)
def test_specaugment_script(**kwargs):
    sa = SpecAugment(**kwargs)
    sa.eval()
    x = torch.ones((1, 100, 100))

    sa_script = torch.jit.script(sa)

    out1 = sa(x)
    out2 = sa_script(x)
    assert torch.allclose(out1, out2, atol=1e-3)
