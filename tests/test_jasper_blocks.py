import pytest

from torch import nn

from thunder.jasper.blocks import (
    InitMode,
    MaskedConv1d,
    compute_new_kernel_size,
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
