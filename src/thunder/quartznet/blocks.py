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
    "get_same_padding",
    "MaskedConv1d",
    "MultiSequential",
    "Masked",
    "QuartznetBlock",
    "stem",
    "body",
    "Quartznet_encoder",
    "Quartznet_decoder",
]

from enum import Enum
from typing import List, Tuple

import torch
from torch import nn


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
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
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


class MaskedConv1d(nn.Module):
    __constants__ = ["use_mask", "padding", "dilation", "kernel_size", "stride"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_mask: bool = True,
    ):
        """Masked Convolution.
        This module correspond to a 1d convolution with input masking. Arguments to create are the
        same as nn.Conv1d, but with the addition of use_mask for special behaviour.
        Args:
            in_channels : Same as nn.Conv1d
            out_channels : Same as nn.Conv1d
            kernel_size : Same as nn.Conv1d
            stride : Same as nn.Conv1d
            padding : Same as nn.Conv1d
            dilation : Same as nn.Conv1d
            groups : Same as nn.Conv1d
            bias : Same as nn.Conv1d
            use_mask : Controls the masking of input before the convolution during the forward.
        """
        super(MaskedConv1d, self).__init__()

        self.use_mask = use_mask

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.padding = self.conv.padding[0]
        self.dilation = self.conv.dilation[0]
        self.kernel_size = self.conv.kernel_size[0]
        self.stride = self.conv.stride[0]

    def get_seq_len(self, lens: torch.Tensor) -> torch.Tensor:
        """Get the lengths of the inputs after the convolution operation is applied.
        Args:
            lens : Original lengths of the inputs
        Returns:
            Resulting lengths after the convolution
        """
        return (
            lens + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1

    def mask_fill(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """Mask the input based on it's respective lengths.
        Args:
            x : Signal to be processed, of shape (batch, features, time)
            lens : Lenghts of each element in the batch of x, with shape (batch)
        Returns:
            The masked signal
        """
        lens = lens.to(dtype=torch.long)
        max_len = x.size(2)
        mask = torch.arange(max_len, device=lens.device).expand(
            lens.shape[0], max_len
        ) >= lens.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        return x

    def forward(
        self, x: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method
        Args:
            x : Signal to be processed, of shape (batch, features, time)
            lens : Lenghts of each element in the batch of x, with shape (batch)
        Returns:
            Both the signal processed by the convolution and the resulting lengths
        """
        if self.use_mask:
            x = self.mask_fill(x, lens)
        out = self.conv(x)
        return out, self.get_seq_len(lens)


class MultiSequential(nn.Sequential):
    def forward(self, x1, x2):
        for module in self.children():
            x1, x2 = module(x1, x2)
        return x1, x2


class Masked(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        return self.layer(x), lens


def _get_conv_bn_layer(
    in_channels,
    out_channels,
    kernel_size=11,
    separable=False,
    **conv_kwargs,
):

    if separable:
        layers = [
            MaskedConv1d(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                **conv_kwargs,
            ),
            MaskedConv1d(
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
            MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                **conv_kwargs,
            )
        ]

    layers.append(Masked(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)))

    return layers


def _get_act_dropout_layer(drop_prob=0.2):
    return [Masked(nn.ReLU(True)), Masked(nn.Dropout(p=drop_prob))]


class QuartznetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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

        self.mconv = MultiSequential(*conv)

        if residual:
            stride_residual = stride if stride[0] == 1 else stride[0] ** repeat

            self.res = MultiSequential(
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

        self.mout = MultiSequential(*_get_act_dropout_layer(drop_prob=dropout))

    def forward(
        self, x: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : Tensor of shape (batch, features, time) where #features == inplanes

        Returns:
            Result of applying the block on the input
        """

        # compute forward convolutions
        out, lens_out = self.mconv(x, lens)

        # compute the residuals
        if self.res is not None:
            res_out, _ = self.res(x, lens)
            out = out + res_out

        # compute the output
        out, lens_out = self.mout(out, lens_out)
        return out, lens_out


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


def Quartznet_encoder(
    feat_in: int,
    filters: List[int] = [256, 256, 512, 512, 512],
    kernel_sizes: List[int] = [33, 39, 51, 63, 75],
    repeat_blocks: int = 1,
) -> nn.Module:
    """Basic Quartznet encoder setup.
    Can be used to build either Quartznet5x5 (repeat_blocks=1) or Quartznet15x5 (repeat_blocks=3)

    Args:
        feat_in : Number of input features to the model.
        filters: List of filter sizes used to create the encoder blocks
        kernel_sizes: List of kernel sizes corresponding to each filter size
        repeat_blocks : Number of repetitions of each block.

    Returns:
        Pytorch model corresponding to the encoder.
    """
    return MultiSequential(
        stem(feat_in),
        *body(filters, kernel_sizes, repeat_blocks),
    )


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
