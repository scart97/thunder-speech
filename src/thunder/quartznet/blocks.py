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
        self.count = 0
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

        layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))

        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2):
        return [nn.ReLU(False), nn.Dropout(p=drop_prob)]

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, features, time) where #features == inplanes

        Returns:
            Result of applying the block on the input
        """

        # compute forward convolutions
        out = x
        lens_orig = seq_lens.clone()
        for i, layer in enumerate(self.mconv):
            if isinstance(layer, MaskedConv1d):
                out, seq_lens = layer(out, seq_lens)
            else:
                self.count += 1
                out = layer(out)

        # compute the residuals
        if self.res is not None:
            res_out = x
            for i, layer in enumerate(self.res):
                if isinstance(layer, MaskedConv1d):
                    res_out, _ = layer(res_out, lens_orig)
                else:
                    res_out = layer(res_out)
            out = out + res_out

        # compute the output
        out = self.mout(out)
        return out, seq_lens


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
    filters: List[int],
    kernel_size: List[int],
    repeat_blocks: int = 1,
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
            QuartznetBlock(512, 1024, repeat=1, kernel_size=[1], residual=False, separable=False),
        ]
    )
    return layers


class MaskedConv1d(nn.Module):
    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
        quantize=False,
    ):
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        # preserve original padding
        self._padding = padding

        # if padding is a tuple/list, it is considered as asymmetric padding
        if type(padding) in (tuple, list):
            self.pad_layer = nn.ConstantPad1d(padding, value=0.0)
            # reset padding for conv since pad_layer will handle this
            padding = 0
        else:
            self.pad_layer = None

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
        self.use_mask = use_mask
        self.heads = heads

        # Calculations for "same" padding cache
        self.same_padding = (self.conv.stride[0] == 1) and (
            2 * self.conv.padding[0] == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
        )
        if self.pad_layer is None:
            self.same_padding_asymmetric = False
        else:
            self.same_padding_asymmetric = (self.conv.stride[0] == 1) and (
                sum(self._padding) == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
            )

        # `self.lens` caches consecutive integers from 0 to `self.max_len` that are used to compute the mask for a
        # batch. Recomputed to bigger size as needed. Stored on a device of the latest batch lens.
        if self.use_mask:
            self.max_len = 0
            self.lens = None

    def get_seq_len(self, lens):
        if self.same_padding or self.same_padding_asymmetric:
            return lens

        if self.pad_layer is None:
            return (
                lens
                + 2 * self.conv.padding[0]
                - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                - 1
            ) // self.conv.stride[0] + 1
        else:
            return (
                lens
                + sum(self._padding)
                - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                - 1
            ) // self.conv.stride[0] + 1

    def forward(self, x, lens):
        if self.use_mask:
            max_len = x.size(2)
            if max_len > self.max_len:
                self.lens = torch.arange(max_len)
                self.max_len = max_len

            self.lens = self.lens.to(lens.device)
            mask = self.lens[:max_len].unsqueeze(0) < lens.unsqueeze(1)
            x = x * mask.unsqueeze(1).to(device=x.device)
            lens = self.get_seq_len(lens)

        # asymmtric pad if necessary
        if self.pad_layer is not None:
            x = self.pad_layer(x)

        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)
        return out, lens
