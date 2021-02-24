"""All of the stuff to load the quartznet checkpoint
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97


from torch import nn

from thunder.quartznet.blocks import body, init_weights, pre_head, stem


def Quartznet5(feat_in: int, repeat_blocks: int = 1) -> nn.Module:
    """Basic Quartznet encoder setup. Can be used to build either Quartznet5x5 or Quartznet15x5,

    Args:
        feat_in : Number of input features to the model.
        repeat_blocks : Number of repetitions of each block.

    Returns:
        Pytorch model corresponding to the encoder.
    """
    filters = [256, 256, 512, 512, 512]
    kernel_sizes = [33, 39, 51, 63, 75]
    return nn.Sequential(
        stem(feat_in),
        *body(filters, kernel_sizes, repeat_blocks),
        *pre_head(),
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


def Quartznet_decoder(num_classes: int) -> nn.Module:
    """Build the Quartznet decoder.

    Args:
        num_classes : Number of output classes of the model. It's the size of the vocabulary, excluding the blank symbol.

    Returns:
        Pytorch model of the decoder
    """
    decoder = nn.Conv1d(
        1024,
        num_classes + 1,
        kernel_size=1,
        bias=True,
    )
    decoder.apply(init_weights)
    return decoder
