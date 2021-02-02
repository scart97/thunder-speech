from math import ceil, floor
from tempfile import TemporaryDirectory
from typing import List

import pytest

import torch
from hypothesis import given
from hypothesis.strategies import integers
from pytorch_lightning import seed_everything
from torch import nn

from thunder.jasper.blocks import (
    GroupShuffle,
    InitMode,
    InterpolationMode,
    MaskedConv1d,
    SqueezeExcite,
    compute_new_kernel_size,
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
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    outputs = model(*x)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = outputs.mean()
    loss.backward()
    optim.step()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert (torch.sum(param.grad ** 2) != 0.0).all()


def _test_simple_device_move(model: nn.Module, x: torch.Tensor):
    seed_everything(42)
    outputs_cpu = model(x)

    seed_everything(42)
    model = model.cuda()
    outputs_gpu = model(x.cuda())

    seed_everything(42)
    model = model.cpu()
    outputs_back_on_cpu = model(x.cpu())

    assert torch.allclose(outputs_cpu, outputs_gpu.cpu())
    assert torch.allclose(outputs_cpu, outputs_back_on_cpu)


def _test_simple_batch_independence(model: nn.Module, x: torch.Tensor):
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


def test_init_batchnorm1d():
    bn_layer = nn.BatchNorm1d(128, 10)

    for init in InitMode:
        init_weights(bn_layer, init)
        assert (bn_layer.running_mean == 0).all()
        assert (bn_layer.running_var == 1).all()
        assert bn_layer.num_batches_tracked == 0
        assert (bn_layer.weight == 1).all()
        assert (bn_layer.bias == 0).all()


def test_compute_kernel_size():
    for kernel_size in range(1, 90):
        for kernel_width in [0.1 * i for i in range(30)]:
            result = compute_new_kernel_size(kernel_size, kernel_width)
            # New kernel should be odd
            assert result % 2 == 1
            # And it should be positive
            assert result > 0
            # The kernel width should correctly control the
            # expansion or contraction in size.
            if kernel_width < 1.0:
                assert result <= kernel_size
            else:
                assert result >= kernel_size


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


def test_maskconv_error():
    with pytest.raises(ValueError):
        MaskedConv1d(128, 64, 3, heads=4)


def test_maskconv_parameters_update():
    x = torch.randn(10, 128, 1337)
    lens = torch.Tensor([1000] * 10)
    conv = MaskedConv1d(128, 10, 3)
    _test_parameters_update(conv, [x, lens])


@requirescuda
def test_maskconv_device_move():
    conv = MaskedConv1d(128, 10, 3)
    x = torch.randn(10, 128, 1337)
    lens = torch.Tensor([1000] * 10)

    seed_everything(42)
    outputs_cpu, lens_cpu = conv(x, lens)

    seed_everything(42)
    conv = conv.cuda()
    outputs_gpu, lens_gpu = conv(x.cuda(), lens.cuda())

    seed_everything(42)
    conv = conv.cpu()
    outputs_back_on_cpu, lens_back_on_cpu = conv(x.cpu(), lens.cpu())

    assert torch.allclose(lens_cpu, lens_gpu.cpu())
    assert torch.allclose(lens_cpu, lens_back_on_cpu)

    assert torch.allclose(outputs_cpu, outputs_gpu.cpu(), atol=1e-6)
    assert torch.allclose(outputs_cpu, outputs_back_on_cpu)


def test_maskconv_batch_independence():
    conv = MaskedConv1d(128, 10, 3)
    x = torch.randn(10, 128, 1337)
    lens = torch.Tensor([1000] * 10)
    x.requires_grad_(True)

    # Compute forward pass in eval mode to deactivate batch norm
    conv.eval()
    outputs, _ = conv(x, lens)
    conv.train()

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
        torch.onnx.export(conv, (x, lens), f"{export_path}/maskconv.onnx", verbose=True)


def test_maskconv_heads_trace():
    x = torch.randn(10, 128, 1337)
    lens = torch.randint(10, 1337, (10,))

    conv = MaskedConv1d(128, 10, 3, heads=2, groups=128)
    conv_trace = torch.jit.trace(conv, (x, lens))
    x_old, lens_old = conv(x, lens)
    x_new, lens_new = conv_trace(x, lens)
    assert torch.allclose(x_old, x_new)
    assert torch.allclose(lens_old, lens_new)


@pytest.mark.xfail
def test_maskconv_heads_script():
    # Only torch.jit.trace works with einops
    conv = MaskedConv1d(128, 10, 3, heads=2, groups=128)
    torch.jit.script(conv)


def test_group_shuffle():
    gs = GroupShuffle(4, 128)
    x = torch.randn(10, 128, 1337)
    out = gs(x)
    assert out.shape == x.shape
    assert not torch.allclose(x, out)


@requirescuda
def test_group_shuffle_device_move():
    gs = GroupShuffle(4, 128)
    x = torch.randn(10, 128, 1337)
    _test_simple_device_move(gs, x)


def test_groupshuffle_batch_independence():
    gs = GroupShuffle(4, 128)
    x = torch.randn(10, 128, 1337)
    _test_simple_batch_independence(gs, x)


def test_group_shuffle_trace():
    x_traced = torch.randn(5, 128, 137)
    gs = GroupShuffle(4, 128)
    gs_trace = torch.jit.trace(gs, x_traced)
    # using a different shape than the traced one
    x = torch.randn(10, 128, 1337)
    assert torch.allclose(gs_trace(x), gs(x))

    with TemporaryDirectory() as save_dir:
        save_file = f"{save_dir}/shuffle.pth"
        torch.jit.save(gs_trace, save_file)
        gs_loaded = torch.jit.load(save_file)
        assert torch.allclose(gs_loaded(x), gs(x))


def test_group_shuffle_onnx():
    x = torch.randn(10, 128, 1337)
    gs = GroupShuffle(4, 128)
    with TemporaryDirectory() as export_path:
        torch.onnx.export(gs, x, f"{export_path}/shuffle.onnx", verbose=True)


@pytest.mark.xfail
def test_group_shuffle_script():
    # Only torch.jit.trace works with einops
    gs = GroupShuffle(4, 128)
    gs_script = torch.jit.script(gs)
    x = torch.randn(10, 128, 1337)
    assert torch.allclose(gs_script(x), gs(x))


def test_squeezeexcite_retains_shape():
    x = torch.randn(10, 128, 1337)
    se = SqueezeExcite(128, 4)
    assert x.shape == se(x).shape


def test_squeezeexcite_interpolation():
    x = torch.randn(10, 128, 1337)
    for interp in InterpolationMode:
        se = SqueezeExcite(128, 4, context_window=2, interpolation_mode=interp)
        assert x.shape == se(x).shape


@requirescuda
def test_squeezeexcite_device_move():
    se = SqueezeExcite(128, 4)
    x = torch.randn(10, 128, 1337)
    _test_simple_device_move(se, x)


@requirescuda
def test_squeezeexcite_interp_device_move():
    for interp in InterpolationMode:
        se = SqueezeExcite(128, 4, context_window=2, interpolation_mode=interp)
        x = torch.randn(10, 128, 1337)
        _test_simple_device_move(se, x)


def test_squeezeexcite_batch_independence():
    se = SqueezeExcite(128, 4)
    x = torch.randn(10, 128, 1337)
    _test_simple_batch_independence(se, x)


def test_squeezeexcite_interp_batch_independence():
    for interp in InterpolationMode:
        se = SqueezeExcite(128, 4, context_window=2, interpolation_mode=interp)
        x = torch.randn(10, 128, 1337)
        _test_simple_batch_independence(se, x)


def test_se_parameters_updated():
    se = SqueezeExcite(128, 4)
    x = torch.randn(10, 128, 1337)
    _test_parameters_update(se, [x])


def test_se_interp_parameters_updated():
    for interp in InterpolationMode:
        se = SqueezeExcite(128, 4, context_window=2, interpolation_mode=interp)
        x = torch.randn(10, 128, 1337)
        _test_parameters_update(se, [x])


def test_squeezeexcite_trace():
    x_traced = torch.randn(5, 128, 137)
    se = SqueezeExcite(128, 4)
    se_trace = torch.jit.trace(se, x_traced)
    # using a different shape than the traced one
    x = torch.randn(10, 128, 1337)
    assert torch.allclose(se_trace(x), se(x))

    with TemporaryDirectory() as save_dir:
        save_file = f"{save_dir}/shuffle.pth"
        torch.jit.save(se_trace, save_file)
        gs_loaded = torch.jit.load(save_file)
        assert torch.allclose(gs_loaded(x), se(x))


def test_squeezeexcite_onnx():
    x = torch.randn(10, 128, 1337)
    se = SqueezeExcite(128, 4)
    with TemporaryDirectory() as export_path:
        torch.onnx.export(se, x, f"{export_path}/squeeze.onnx", verbose=True)


@pytest.mark.xfail
def test_squeezeexcite_script():
    # Only torch.jit.trace works with einops
    se = SqueezeExcite(128, 4)
    se_script = torch.jit.script(se)
    x = torch.randn(10, 128, 1337)
    assert torch.allclose(se_script(x), se(x))
