# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from tempfile import TemporaryDirectory

import torch
from hypothesis import given, settings
from hypothesis.strategies import booleans, integers, lists

from tests.utils import (
    _test_batch_independence,
    _test_device_move,
    _test_parameters_update,
    mark_slow,
    requirescuda,
)
from thunder.citrinet.blocks import CitrinetBlock, SqueezeExcite


def test_squeezeexcite_retains_shape():
    x = torch.randn(10, 128, 1337)
    se = SqueezeExcite(128, 4)
    assert x.shape == se(x).shape


@requirescuda
def test_squeezeexcite_device_move():
    se = SqueezeExcite(128, 4)
    x = torch.randn(10, 128, 1337)
    _test_device_move(se, x)


def test_squeezeexcite_batch_independence():
    se = SqueezeExcite(128, 4)
    x = torch.randn(10, 128, 1337)
    _test_batch_independence(se, x)


def test_squeezeexcite_parameters_updated():
    se = SqueezeExcite(128, 4)
    x = torch.randn(10, 128, 1337)
    _test_parameters_update(se, x)


def test_squeezeexcite_trace():
    x_traced = torch.randn(5, 128, 137)
    se = SqueezeExcite(128, 4)
    # using a different shape than the traced one
    x = torch.randn(10, 128, 1337)
    se_trace = torch.jit.trace(se, x_traced, check_inputs=[x])

    with TemporaryDirectory() as save_dir:
        save_file = f"{save_dir}/squeeze.pth"
        torch.jit.save(se_trace, save_file)
        gs_loaded = torch.jit.load(save_file)
        assert torch.allclose(gs_loaded(x), se(x))


def test_squeezeexcite_onnx():
    x = torch.randn(10, 128, 1337)
    se = SqueezeExcite(128, 4)
    with TemporaryDirectory() as export_path:
        torch.onnx.export(se, x, f"{export_path}/squeeze.onnx", verbose=True)


def test_squeezeexcite_script():
    se = SqueezeExcite(128, 4)
    se_script = torch.jit.script(se)
    x = torch.randn(10, 128, 1337)
    assert torch.allclose(se_script(x), se(x))


citrinet_parameters = given(
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


@citrinet_parameters
def test_CitrinetBlock_combinations(**kwargs):
    try:
        block = CitrinetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))

    out, _ = block(x, lens)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == kwargs["out_channels"]


@mark_slow
@citrinet_parameters
def test_CitrinetBlock_update(**kwargs):
    try:
        block = CitrinetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    _test_parameters_update(block, (x, lens))


@mark_slow
@citrinet_parameters
@settings(deadline=None)
def test_CitrinetBlock_independence(**kwargs):
    try:
        block = CitrinetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    _test_batch_independence(block, (x, lens))


@mark_slow
@requirescuda
@citrinet_parameters
@settings(deadline=None)
def test_CitrinetBlock_device_move(**kwargs):
    try:
        block = CitrinetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    _test_device_move(block, (x, lens))


@mark_slow
@citrinet_parameters
@settings(deadline=None)
def test_CitrinetBlock_trace(**kwargs):
    try:
        block = CitrinetBlock(**kwargs)
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
@citrinet_parameters
@settings(deadline=None)
def test_CitrinetBlock_script(**kwargs):
    try:
        block = CitrinetBlock(**kwargs)
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
@citrinet_parameters
@settings(deadline=None)
def test_CitrinetBlock_onnx(**kwargs):
    try:
        block = CitrinetBlock(**kwargs)
    except ValueError:
        return
    x = torch.randn(10, kwargs["in_channels"], 1337)
    lens = torch.randint(10, 1337, (10,))
    with TemporaryDirectory() as export_path:
        torch.onnx.export(
            block,
            (x, lens),
            f"{export_path}/CitrinetBlock.onnx",
            verbose=True,
            opset_version=11,
        )
