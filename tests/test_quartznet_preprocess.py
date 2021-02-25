# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from tempfile import TemporaryDirectory

import torch

from tests.utils import _test_batch_independence, _test_device_move, requirescuda
from thunder.quartznet.preprocess import FeatureBatchNormalizer


def test_normalize_preserve_shape():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    out = norm(x)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == x.shape[1]
    assert out.shape[2] == x.shape[2]


def test_normalize_has_correct_mean_std():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    out = norm(x)
    # Testing for each element in the batch
    for xb in out:
        assert torch.allclose(xb[:, :].mean(), torch.zeros(1), atol=0.1)
        assert torch.allclose(xb[:, :].std(), torch.ones(1), atol=0.1)


def test_normalize_batch_independence():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    _test_batch_independence(norm, x)


@requirescuda
def test_normalize_device_move():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    _test_device_move(norm, x)


def test_normalize_trace():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    norm_trace = torch.jit.trace(norm, (x))

    assert torch.allclose(norm(x), norm_trace(x))


def test_normalize_script():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)
    norm_script = torch.jit.script(norm)

    assert torch.allclose(norm(x), norm_script(x))


def test_normalize_onnx():
    norm = FeatureBatchNormalizer()
    x = torch.randn(10, 40, 1337)

    with TemporaryDirectory() as export_path:
        torch.onnx.export(
            norm,
            (x),
            f"{export_path}/Normalize.onnx",
            verbose=True,
            opset_version=11,
        )
