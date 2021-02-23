# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from math import ceil, floor
from tempfile import TemporaryDirectory
from typing import List

import pytest

import torch
from hypothesis import given, settings
from hypothesis.strategies import booleans, integers, lists
from pytorch_lightning import seed_everything
from torch import nn

from thunder.quartznet.blocks import (
    InitMode,
    QuartznetBlock,
    get_same_padding,
    init_weights,
)

# ##############################################
# Awesome resource on deep learning testing.
# Inspired various tests here.
# https://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html
# ##############################################

requirescuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test requires cuda."
)


def _test_parameters_update(model: nn.Module, x: List[torch.Tensor]):
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    outputs = model(x)
    if isinstance(outputs, list):
        outputs = outputs[-1]
    loss = outputs.mean()
    loss.backward()
    optim.step()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert (torch.sum(param.grad ** 2) != 0.0).all()


def _test_device_move(model: nn.Module, x: torch.Tensor):
    model.eval()
    seed_everything(42)
    outputs_cpu = model(x)

    seed_everything(42)
    model = model.cuda()
    outputs_gpu = model(x.cuda())

    seed_everything(42)
    model = model.cpu()
    outputs_back_on_cpu = model(x.cpu())
    model.train()
    assert torch.allclose(outputs_cpu, outputs_gpu.cpu(), atol=1e-4)
    assert torch.allclose(outputs_cpu, outputs_back_on_cpu)


def _test_batch_independence(model: nn.Module, x: torch.Tensor):
    x.requires_grad_(True)

    # Compute forward pass in eval mode to deactivate batch norm
    model.eval()
    outputs = model(x)
    model.train()

    # Mask loss for certain samples in batch
    batch_size = x.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(outputs)
    mask[mask_idx] = 0
    outputs = outputs * mask

    # Compute backward pass
    loss = outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(x.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)


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


quartznet_parameters = given(
    inplanes=integers(16, 32),
    planes=integers(16, 32),
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
    x = torch.randn(10, kwargs["inplanes"], 1337)
    out = block(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == kwargs["planes"]


@quartznet_parameters
def test_QuartznetBlock_update(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["inplanes"], 1337)
    _test_parameters_update(block, x)


@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_independence(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["inplanes"], 1337)
    _test_batch_independence(block, x)


@requirescuda
@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_device_move(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["inplanes"], 1337)
    _test_device_move(block, x)


@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_trace(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
        block.eval()
    except ValueError:
        return

    x = torch.randn(10, kwargs["inplanes"], 1337)
    block_trace = torch.jit.trace(block, (x))

    assert torch.allclose(block(x), block_trace(x))


@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_script(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
        block.eval()
    except ValueError:
        return

    x = torch.randn(10, kwargs["inplanes"], 1337)
    block_script = torch.jit.script(block)
    assert torch.allclose(block(x), block_script(x))


@quartznet_parameters
@settings(deadline=None)
def test_QuartznetBlock_onnx(**kwargs):
    try:
        block = QuartznetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["inplanes"], 1337)
    with TemporaryDirectory() as export_path:
        torch.onnx.export(
            block,
            (x),
            f"{export_path}/QuartznetBlock.onnx",
            verbose=True,
            opset_version=11,
        )
