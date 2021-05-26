# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from math import ceil, floor
from tempfile import TemporaryDirectory

import pytest

import torch
from hypothesis import given, settings
from hypothesis.strategies import booleans, integers, lists
from torch import nn

from tests.utils import (
    _test_batch_independence,
    _test_device_move,
    _test_parameters_update,
    mark_slow,
    requirescuda,
)
from thunder.quartznet.blocks import (
    InitMode,
    MaskedConv1d,
    QuartznetBlock,
    get_same_padding,
    init_weights,
)


def test_init_linear_weights():
    linear_layer = nn.Linear(128, 10)
    original_std = linear_layer.weight.std()
    original_mean = linear_layer.weight.mean()

    for init in InitMode:
        init_weights(linear_layer, init)
        assert linear_layer.weight.std() != original_std
        assert linear_layer.weight.mean() != original_mean

    with pytest.raises(ValueError):
        init_weights(linear_layer, "unknown_init")


def test_init_batchnorm1d():
    bn_layer = nn.BatchNorm1d(128, 10)

    for init in InitMode:
        init_weights(bn_layer, init)
        assert (bn_layer.running_mean == 0).all()
        assert (bn_layer.running_var == 1).all()
        assert bn_layer.num_batches_tracked == 0
        assert (bn_layer.weight == 1).all()
        assert (bn_layer.bias == 0).all()


def test_init_masked_conv():
    conv_layer = MaskedConv1d(128, 10, 11)
    original_std = conv_layer.conv.weight.std()
    original_mean = conv_layer.conv.weight.mean()

    for init in InitMode:
        init_weights(conv_layer, init)
        assert conv_layer.conv.weight.std() != original_std
        assert conv_layer.conv.weight.mean() != original_mean

    with pytest.raises(ValueError):
        init_weights(conv_layer, "unknown_init")


def conv_outsize(i: int, p: int, k: int, s: int, d: int) -> int:
    """Calculates conv output size based on input size and params
        Based on https://arxiv.org/abs/1603.07285 Section 2.4 - Relationship 6
        and also the Conv1d pytorch docs - https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    Args:
        i: input size
        p: padding
        k: kernel size
        s: stride
        d: dilation

    Returns:
        output size
    """
    numerator = i + 2 * p - d * (k - 1) - 1
    return floor(numerator / s + 1)


@given(
    i=integers(10, 10_000),
    stride=integers(1, 4),
    ks=integers(3, 100).filter(lambda x: x % 2 == 1),
)
def test_same_padding_unit_dilation(i, stride, ks):
    p = get_same_padding(ks, stride, 1)
    assert conv_outsize(i, p, ks, stride, 1) == ceil(i / stride)


@given(
    i=integers(100, 10_000),
    dilation=integers(2, 5),
    ks=integers(3, 100).filter(lambda x: x % 2 == 1),
)
def test_same_padding_unit_stride(i, dilation, ks):
    p = get_same_padding(ks, 1, dilation)
    assert conv_outsize(i, p, ks, 1, dilation) == i


@given(
    i=integers(10, 10_000),
    stride=integers(2, 10),
    ks=integers(3, 100).filter(lambda x: x % 2 == 1),
    dilation=integers(2, 10),
)
def test_same_padding_ValueError(i, stride, ks, dilation):
    with pytest.raises(ValueError):
        get_same_padding(ks, stride, dilation)


@given(
    in_channels=integers(1, 128),
    out_channels=integers(1, 128),
    kernel_size=integers(1, 128),
    stride=integers(1, 64),
    padding=integers(1, 64),
)
def test_maskconv_init(in_channels, out_channels, kernel_size, stride, padding):
    maskconv = MaskedConv1d(in_channels, out_channels, kernel_size, stride, padding)
    assert maskconv.conv.in_channels == in_channels
    assert maskconv.conv.out_channels == out_channels
    assert maskconv.conv.kernel_size[0] == kernel_size
    assert maskconv.conv.stride[0] == stride
    assert maskconv.conv.padding[0] == padding


def test_mask_fill():
    x = torch.randn(10, 128, 1337)
    lens = torch.Tensor([1000] * 10)
    conv = MaskedConv1d(128, 10, 3)
    x_mask = conv.mask_fill(x, lens)
    lens_mask = conv.get_seq_len(lens)

    assert lens_mask[0] == conv_outsize(1000, 0, 3, 1, 1)
    assert (x_mask[:, :, 1000:] == 0).all()


def test_maskconv_script():
    x = torch.randn(10, 128, 1337)
    lens = torch.randint(10, 1337, (10,))
    conv = MaskedConv1d(128, 10, 3)

    conv_script = torch.jit.script(conv)
    x_old, lens_old = conv(x, lens)
    x_new, lens_new = conv_script(x, lens)
    assert torch.allclose(x_old, x_new)
    assert torch.allclose(lens_old, lens_new)


def test_maskconv_onnx():
    x = torch.randn(10, 128, 1337)
    lens = torch.randint(10, 1337, (10,))
    conv = MaskedConv1d(128, 10, 3)
    with TemporaryDirectory() as export_path:
        torch.onnx.export(conv, (x, lens), f"{export_path}/maskconv.onnx")


quartznet_parameters = given(
    in_channels=integers(16, 32),
    out_channels=integers(16, 32),
    repeat=integers(1, 4),
    kernel_size=lists(
        integers(11, 33).filter(lambda x: x % 2 == 1), min_size=1, max_size=1
    ),
    stride=lists(integers(1, 3), min_size=1, max_size=1),
    dilation=lists(integers(1, 2), min_size=1, max_size=1),
    residual=booleans(),
    separable=booleans(),
)


@quartznet_parameters
def test_QuartznetBlock_combinations(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))

    out, _ = block(x, lens)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == kwargs["out_channels"]


@mark_slow
@quartznet_parameters
def test_QuartznetBlock_update(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    _test_parameters_update(block, (x, lens))


@mark_slow
@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_independence(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    _test_batch_independence(block, (x, lens))


@mark_slow
@requirescuda
@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_device_move(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    _test_device_move(block, (x, lens))


@mark_slow
@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_trace(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
        block.eval()
    except ValueError:
        return

    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    block_trace = torch.jit.trace(block, (x, lens))

    out1, lens1 = block(x, lens)
    out2, lens2 = block_trace(x, lens)
    assert torch.allclose(out1, out2)
    assert torch.allclose(lens1, lens2)


@mark_slow
@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_script(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
        block.eval()
    except ValueError:
        return

    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    block_script = torch.jit.script(block)

    out1, lens1 = block(x, lens)
    out2, lens2 = block_script(x, lens)
    assert torch.allclose(out1, out2)
    assert torch.allclose(lens1, lens2)


@mark_slow
@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_onnx(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    with TemporaryDirectory() as export_path:
        torch.onnx.export(
            block,
            (x, lens),
            f"{export_path}/QuartznetBlock.onnx",
            verbose=True,
            opset_version=11,
        )
