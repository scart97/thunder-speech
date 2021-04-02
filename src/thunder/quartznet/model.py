"""Functionality to quickly create a new quartznet model.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97


from typing import List

from torch import nn

from thunder.quartznet.blocks import body, init_weights, stem


def Quartznet5(
    feat_in: int,
    filters: List[int] = [256, 256, 512, 512, 512],
    kernel_sizes: List[int] = [33, 39, 51, 63, 75],
    repeat_blocks: int = 1,
) -> nn.Module:
    """Basic Quartznet encoder setup.
    Can be used to build either Quartznet5x5 (repeat_blocks=1) or Quartznet15x5 (repeat_blocks=3)

    Args:
        feat_in : Number of input features to the model.
        repeat_blocks : Number of repetitions of each block.

    Returns:
        Pytorch model corresponding to the encoder.
    """
    return nn.Sequential(
        stem(feat_in),
        *body(filters, kernel_sizes, repeat_blocks),
    )


def Quartznet5x5_encoder(feat_in: int = 64) -> nn.Module:
    """Build encoder corresponding to the Quartznet5x5 model.

    Args:
        feat_in : Number of input features to the model.

    Returns:
        Pytorch model of the encoder
    """
    return Quartznet5(feat_in)


def Quartznet15x5_encoder(feat_in: int = 64) -> nn.Module:
    """Build encoder corresponding to the Quartznet15x5 model.

    Args:
        feat_in : Number of input features to the model.

    Returns:
        Pytorch model of the encoder
    """
    return Quartznet5(feat_in, repeat_blocks=3)


def Quartznet_decoder(num_classes: int, input_channels: int = 1024) -> nn.Module:
    """Build the Quartznet decoder.

    Args:
        num_classes : Number of output classes of the model. It's the size of the vocabulary, excluding the blank symbol.

    Returns:
        Pytorch model of the decoder
    """
    decoder = nn.Conv1d(
        input_channels,
        num_classes,
        kernel_size=1,
        bias=True,
    )
    decoder.apply(init_weights)
    return decoder
