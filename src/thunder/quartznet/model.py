"""All of the stuff to load the quartznet checkpoint
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

import torch
from torch import nn

from thunder.quartznet.blocks import body, init_weights, pre_head, stem


def Quartznet5(feat_in: int, repeat_blocks: int = 1):
    filters = [256, 256, 512, 512, 512]
    kernel_sizes = [33, 39, 51, 63, 75]
    return nn.Sequential(
        stem(feat_in),
        *body(filters, kernel_sizes, repeat_blocks),
        *pre_head(),
    )


def Quartznet5x5_encoder(feat_in: int = 64):
    return Quartznet5(feat_in)


def Quartznet15x5_encoder(feat_in: int = 64):
    return Quartznet5(feat_in, repeat_blocks=3)


def Quartznet_decoder(feat_in: int, num_classes: int) -> nn.Module:
    decoder = torch.nn.Conv1d(
        feat_in,
        num_classes + 1,
        kernel_size=1,
        bias=True,
    )
    decoder.apply(init_weights)
    return decoder
