"""Basic building blocks to create the Quartznet model
"""

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

# Original file: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/quartznet.py

__all__ = [
    "InitMode",
    "init_weights",
    "_get_act_dropout_layer",
    "_get_conv_bn_layer",
    "QuartznetBlock",
    "stem",
    "body",
    "Quartznet_encoder",
]

from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
from torch import nn
from torch.nn.common_types import _size_1_t

from thunder.blocks import get_same_padding
from thunder.utils import default_list


class InitMode(str, Enum):
    """Weight init methods. Used by [`init_weights`][thunder.quartznet.blocks.init_weights].

    Note:
        Possible values are `xavier_uniform`,`xavier_normal`,`kaiming_uniform` and `kaiming_normal`
    """

    xavier_uniform = "xavier_uniform"
    xavier_normal = "xavier_normal"
    kaiming_uniform = "kaiming_uniform"
    kaiming_normal = "kaiming_normal"


def init_weights(m: nn.Module, mode: InitMode = InitMode.xavier_uniform):
    """Initialize Linear, Conv1d or BatchNorm1d weights.
    There's no return, the operation occurs inplace.

    Args:
        m: The layer to be initialized
        mode: Weight initialization mode. Only applicable to linear and conv layers.

    Raises:
        ValueError: Raised when the initial mode is not one of the possible options.
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode == InitMode.xavier_uniform:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == InitMode.xavier_normal:
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == InitMode.kaiming_uniform:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == InitMode.kaiming_normal:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.layer = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = x == 0.0
            result = self.layer(x)
            return torch.masked_fill(result, mask, 0.0)
        else:
            return self.layer(x)


def _get_conv_bn_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: _size_1_t = 11,
    separable: bool = False,
    **conv_kwargs,
):

    if separable:
        layers = [
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                **conv_kwargs,
            ),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
                bias=conv_kwargs.get("bias", False),
            ),
        ]
    else:
        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                **conv_kwargs,
            )
        ]

    layers.append(MaskedBatchNorm1d(out_channels))

    return layers


def _get_act_dropout_layer(drop_prob: float = 0.2):
    return [nn.ReLU(True), nn.Dropout(p=drop_prob)]


class QuartznetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeat: int = 5,
        kernel_size: _size_1_t = (11,),
        stride: _size_1_t = (1,),
        dilation: _size_1_t = (1,),
        dropout: float = 0.0,
        residual: bool = True,
        separable: bool = False,
    ):
        """Quartznet block. This is a refactoring of the Jasperblock present on the NeMo toolkit,
        but simplified to only support the new quartznet model. Biggest change is that
        dense residual used on Jasper is not supported here, and masked convolutions were also removed.

        Args:
            in_channels : Number of input channels
            out_channels : Number of output channels
            repeat : Repetitions inside block.
            kernel_size : Kernel size.
            stride : Stride of each repetition.
            dilation : Dilation of each repetition.
            dropout : Dropout used before each activation.
            residual : Controls the use of residual connection.
            separable : Controls the use of separable convolutions.
        """
        super().__init__()

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])

        inplanes_loop = in_channels
        conv = []

        for _ in range(repeat - 1):

            conv.extend(
                _get_conv_bn_layer(
                    inplanes_loop,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_val,
                    separable=separable,
                    bias=False,
                )
            )

            conv.extend(_get_act_dropout_layer(drop_prob=dropout))

            inplanes_loop = out_channels

        conv.extend(
            _get_conv_bn_layer(
                inplanes_loop,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                separable=separable,
                bias=False,
            )
        )

        self.mconv = nn.Sequential(*conv)

        if residual:
            stride_residual = stride if stride[0] == 1 else stride[0] ** repeat

            self.res = nn.Sequential(
                *_get_conv_bn_layer(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride_residual,
                    bias=False,
                )
            )
        else:
            self.res = None

        self.mout = nn.Sequential(*_get_act_dropout_layer(drop_prob=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, features, time) where #features == inplanes

        Returns:
            Result of applying the block on the input, and corresponding output lengths
        """

        # compute forward convolutions
        out = self.mconv(x)

        # compute the residuals
        if self.res is not None:
            res_out = self.res(x)
            out = out + res_out

        # compute the output
        return self.mout(out)


def stem(feat_in: int) -> QuartznetBlock:
    """Creates the Quartznet stem. That is the first block of the model, that process the input directly.

    Args:
        feat_in : Number of input features

    Returns:
        Quartznet stem block
    """
    return QuartznetBlock(
        feat_in,
        256,
        repeat=1,
        stride=(2,),
        kernel_size=(33,),
        residual=False,
        separable=True,
    )


def body(
    filters: List[int], kernel_size: List[int], repeat_blocks: int = 1
) -> List[QuartznetBlock]:
    """Creates the body of the Quartznet model. That is the middle part.

    Args:
        filters : List of filters inside each block in the body.
        kernel_size : Corresponding list of kernel sizes for each block. Should have the same length as the first argument.
        repeat_blocks : Number of repetitions of each block inside the body.

    Returns:
        List of layers that form the body of the network.
    """
    layers = []
    f_in = 256
    for f, k in zip(filters, kernel_size):
        for _ in range(repeat_blocks):
            layers.append(QuartznetBlock(f_in, f, kernel_size=(k,), separable=True))
            f_in = f
    layers.extend(
        [
            QuartznetBlock(
                f_in,
                512,
                repeat=1,
                dilation=(2,),
                kernel_size=(87,),
                residual=False,
                separable=True,
            ),
            QuartznetBlock(
                512, 1024, repeat=1, kernel_size=(1,), residual=False, separable=False
            ),
        ]
    )
    return layers


@dataclass
class EncoderConfig:
    """Configuration to create [`Quartznet_encoder`][thunder.quartznet.blocks.Quartznet_encoder]

    Attributes:
        feat_in: Number of input features to the model. defaults to 64.
        filters: List of filter sizes used to create the encoder blocks. defaults to [256, 256, 512, 512, 512].
        kernel_sizes: List of kernel sizes corresponding to each filter size. defaults to [33, 39, 51, 63, 75].
        repeat_blocks: Number of repetitions of each block. defaults to 1.
    """

    feat_in: int = 64
    filters: List[int] = default_list([256, 256, 512, 512, 512])
    kernel_sizes: List[int] = default_list([33, 39, 51, 63, 75])
    repeat_blocks: int = 1


def Quartznet_encoder(cfg: EncoderConfig = EncoderConfig()) -> nn.Module:
    """Basic Quartznet encoder setup.
    Can be used to build either Quartznet5x5 (repeat_blocks=1) or Quartznet15x5 (repeat_blocks=3)

    Args:
        cfg: required config to create instance
    Returns:
        Pytorch model corresponding to the encoder.
    """
    return nn.Sequential(
        stem(cfg.feat_in),
        *body(cfg.filters, cfg.kernel_sizes, cfg.repeat_blocks),
    )
