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

from enum import Enum
from typing import List

import torch
import torch.nn as nn


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
    """Initialize Linear, MaskedConv1d/Conv1d or BatchNorm1d weights.
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


def get_same_padding(kernel_size: int, stride: int, dilation: int) -> int:
    """Calculates the padding size to obtain same padding.
        Same padding means that the output will have the
        shape input_shape / stride. That means, for
        stride = 1 the output shape is the same as the input,
        and stride = 2 gives an output that is half of the
        input shape.

    Args:
        kernel_size : convolution kernel size. Only tested to be correct with odd values.
        stride : convolution stride
        dilation : convolution dilation

    Raises:
        ValueError: Only stride or dilation may be greater than 1

    Returns:
        padding value to obtain same padding.
    """
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * (kernel_size - 1) + 1) // 2
    return kernel_size // 2


class QuartznetBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        repeat: int = 5,
        kernel_size: List[int] = [11],
        stride: List[int] = [1],
        dilation: List[int] = [1],
        dropout: float = 0.0,
        residual: bool = True,
        separable: bool = False,
    ):
        """Quartznet block. This is a refactoring of the Jasperblock present on the NeMo toolkit,
        but simplified to only support the new quartznet model. Biggest change is that
        dense residual used on Jasper is not supported here, and masked convolutions were also removed.

        Args:
            inplanes : Number of input planes
            planes : Number of output planes
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

        inplanes_loop = inplanes
        conv = []

        for _ in range(repeat - 1):

            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_val,
                    separable=separable,
                    bias=False,
                )
            )

            conv.extend(self._get_act_dropout_layer(drop_prob=dropout))

            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
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
                *self._get_conv_bn_layer(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride_residual,
                    bias=False,
                )
            )
        else:
            self.res = None

        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout))

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        separable=False,
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

        layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))

        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2):
        return [nn.ReLU(True), nn.Dropout(p=drop_prob)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, features, time) where #features == inplanes

        Returns:
            Result of applying the block on the input
        """

        # compute forward convolutions
        out = self.mconv(x)

        # compute the residuals
        if self.res is not None:
            res_out = self.res(x)
            out = out + res_out

        # compute the output
        out = self.mout(out)
        return out


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
        stride=[2],
        kernel_size=[33],
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
            layers.append(QuartznetBlock(f_in, f, kernel_size=[k], separable=True))
            f_in = f
    layers.extend(
        [
            QuartznetBlock(
                f_in,
                512,
                repeat=1,
                dilation=[2],
                kernel_size=[87],
                residual=False,
                separable=True,
            ),
            QuartznetBlock(
                512, 1024, repeat=1, kernel_size=[1], residual=False, separable=False
            ),
        ]
    )
    return layers
