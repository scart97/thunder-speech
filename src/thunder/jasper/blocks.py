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

# Original file: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/jasper.py

from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


class InitMode(str, Enum):
    """Weight init methods. Used by [`init_weights`][thunder.jasper.blocks.init_weights].

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


def compute_new_kernel_size(kernel_size: int, kernel_width: float) -> int:
    """Calculates the new convolutional kernel size given the factor kernel_width.

    Args:
        kernel_size: Original kernel size
        kernel_width: Contraction or expansion factor

    Returns:
        First positive odd number that is equal or greater than kernel_size * kernel_width
    """
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


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


def Conv1dWithHeads(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    heads: int = -1,
    bias: bool = False,
):
    if not (heads == -1 or groups == in_channels):
        raise ValueError("Only use heads for depthwise convolutions")
    if not (heads == -1 or in_channels % heads == 0):
        raise ValueError(
            f"The number of input channels {in_channels} cannot be evenly distributed into {heads} heads"
        )

    if heads != -1:
        in_channels = heads
        out_channels = heads
        groups = heads

    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )
    if heads != -1:
        conv = nn.Sequential(
            Rearrange("b (f heads) t -> (b f) heads t", heads=heads),
            conv,
            Rearrange(
                "(b f) heads t -> b (f heads) t",
                heads=heads,
                f=out_channels // heads,
            ),
        )
    return conv


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
        heads: int = -1,
        bias: bool = False,
        use_mask: bool = True,
    ):
        """Masked Convolution.
        This module correspond to a 1d convolution with input masking. Arguments to create are the
        same as nn.Conv1d, but with the addition of heads and use_mask for special behaviour.


        Args:
            in_channels : Same as nn.Conv1d
            out_channels : Same as nn.Conv1d
            kernel_size : Same as nn.Conv1d
            stride : Same as nn.Conv1d
            padding : Same as nn.Conv1d
            dilation : Same as nn.Conv1d
            groups : Same as nn.Conv1d
            bias : Same as nn.Conv1d
            heads : Number of heads to be used. Only applicable for depthwise convolutions.
            use_mask : Controls the masking of input before the convolution during the forward.

        Raises:
            ValueError: Selecting heads != -1 without doing a depthwise convolution will raise this error.
        """
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.use_mask = use_mask
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if heads != -1:
            self.conv = nn.Sequential(
                Rearrange("b (f heads) t -> (b f) heads t", heads=heads),
                conv,
                Rearrange(
                    "(b f) heads t -> b (f heads) t",
                    heads=heads,
                    f=out_channels // heads,
                ),
            )
        else:
            self.conv = conv

        self.padding = conv.padding[0]
        self.dilation = conv.dilation[0]
        self.kernel_size = conv.kernel_size[0]
        self.stride = conv.stride[0]

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
        """
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


def GroupShuffle(groups: int, channels: int) -> nn.Module:
    """Group shuffle operator from shufflenet.
    Original paper: https://arxiv.org/abs/1707.01083

    Args:
        groups : Number of groups to be used
        channels : Number of channels of the input.Deprecated, only keep for compatibility with old checkpoints.

    Returns:
        Group shuffle layer
    """
    return Rearrange("b (c1 c2) t -> b (c2 c1) t", c1=groups)


class InterpolationMode(str, Enum):
    """Interpolation mode. Used by [`SqueezeExcite`][thunder.jasper.blocks.SqueezeExcite] block.

    Note:
        Possible values are `nearest`,`linear` and `area`
    """

    nearest = "nearest"
    linear = "linear"
    area = "area"


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: InterpolationMode = InterpolationMode.nearest,
        activation: nn.Module = nn.ReLU(inplace=True),
    ):
        """
        Squeeze-and-Excitation sub-module.

        Args:
            channels: Input number of channels.
            reduction_ratio: Reduction ratio for "squeeze" layer.
            context_window: Integer number of timesteps that the context
                should be computed over, using stride 1 average pooling.
                If value < 1, then global context is computed.
            interpolation_mode: Interpolation mode of timestep dimension.
                Used only if context window is > 1.
            activation: Intermediate activation function used.
        """
        super(SqueezeExcite, self).__init__()
        self.context_window = int(context_window)
        self.interpolation_mode = interpolation_mode

        if self.context_window <= 0:
            self.pool = nn.AdaptiveAvgPool1d(1)  # context window = T
        else:
            self.pool = nn.AvgPool1d(self.context_window, stride=1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, channels, time)

        Returns:
            Tensor of shape (batch, channels, time)
        """
        batch, channels, timesteps = x.size()[:3]
        y = self.pool(x)  # [B, C, T - context_window + 1]
        y = rearrange(y, "b c t_pool -> b t_pool c")
        y = self.fc(y)  # [B, T - context_window + 1, C]
        y = rearrange(y, "b t_pool c -> b c t_pool")

        if self.context_window > 0:
            y = torch.nn.functional.interpolate(
                y, size=timesteps, mode=self.interpolation_mode
            )

        y = torch.sigmoid(y)

        return x * y


class NormalizationType(str, Enum):
    """Normalization type. Used by [`get_normalization`][thunder.jasper.blocks.get_normalization].

    Note:
        Possible values are `group`,`instance`, `layer` and `batch`
    """

    group = "group"
    instance = "instance"
    layer = "layer"
    batch = "batch"


def get_normalization(
    normalization: NormalizationType,
    norm_groups: int,
    out_channels: int,
):
    if normalization == NormalizationType.group:
        return nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
    elif normalization == NormalizationType.instance:
        return nn.InstanceNorm1d(out_channels)
    elif normalization == NormalizationType.layer:
        return nn.GroupNorm(num_groups=1, num_channels=out_channels)
    elif normalization == NormalizationType.batch:
        return nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
    else:
        raise ValueError(
            f"Normalization method ({normalization}) does not match"
            f" one of [batch, layer, group, instance]."
        )


class JasperBlock(nn.Module):
    __constants__ = ["residual_mode", "res", "mconv"]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        repeat: int = 3,
        kernel_size: List[int] = [11],
        kernel_size_factor: float = 1.0,
        stride: List[int] = [1],
        dilation: List[int] = [1],
        dropout: float = 0.2,
        activation=nn.Hardtanh(min_val=0.0, max_val=20.0),
        residual: bool = True,
        groups: int = 1,
        separable: bool = False,
        heads: int = -1,
        normalization: str = "batch",
        norm_groups: int = 1,
        residual_mode: str = "add",
        residual_panes=[],
        se: bool = False,
        se_reduction_ratio: int = 16,
        se_context_window=None,
        se_interpolation_mode: str = "nearest",
        stride_last: bool = False,
    ):
        super().__init__()
        if separable and heads != -1:
            raise ValueError(
                "Separable convolutions are not compatible with multiple heads"
            )

        kernel_size_factor = float(kernel_size_factor)
        kernel_size = [
            compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size
        ]
        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])

        self.residual_mode = residual_mode

        inplanes_loop = inplanes
        conv = []

        for _ in range(repeat - 1):
            # Stride last means only the last convolution in block will have stride
            if stride_last:
                stride_val = [1]
            else:
                stride_val = stride

            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride_val,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    bias=False,
                )
            )

            conv.extend(
                self._get_act_dropout_layer(drop_prob=dropout, activation=activation)
            )

            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups,
                bias=False,
            )
        )

        if se:
            conv.append(
                SqueezeExcite(
                    planes,
                    reduction_ratio=se_reduction_ratio,
                    context_window=se_context_window,
                    interpolation_mode=se_interpolation_mode,
                    activation=activation,
                )
            )

        self.mconv = nn.Sequential(*conv)

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()

            stride_residual = (
                stride if stride[0] == 1 or stride_last else stride[0] ** repeat
            )
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res = nn.Sequential(
                    *self._get_conv_bn_layer(
                        ip,
                        planes,
                        kernel_size=1,
                        normalization=normalization,
                        norm_groups=norm_groups,
                        stride=stride_residual,
                        bias=False,
                    )
                )

                res_list.append(res)

            self.res = res_list
        else:
            self.res = None

        self.mout = nn.Sequential(
            *self._get_act_dropout_layer(drop_prob=dropout, activation=activation)
        )

    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        heads=-1,
        **conv_kwargs,
    ):

        return Conv1dWithHeads(
            in_channels,
            out_channels,
            kernel_size,
            heads=heads,
            **conv_kwargs,
        )

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        groups=1,
        heads=-1,
        separable=False,
        normalization="batch",
        norm_groups=1,
        **conv_kwargs,
    ):

        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    groups=in_channels,
                    heads=heads,
                    **conv_kwargs,
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=conv_kwargs.get("bias", False),
                    groups=groups,
                ),
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    groups=groups,
                    **conv_kwargs,
                )
            ]

        norm_groups = out_channels if norm_groups == -1 else norm_groups
        layers.append(get_normalization(normalization, norm_groups, out_channels))

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, activation, drop_prob=0.2):
        return [activation, nn.Dropout(p=drop_prob)]

    def forward(self, xs: List[torch.Tensor]):
        # compute forward convolutions
        out = xs[-1]
        out = self.mconv(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = layer(xs[i])
                if self.residual_mode == "add" or self.residual_mode == "stride_add":
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)

        # compute the output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out]

        return [out]
