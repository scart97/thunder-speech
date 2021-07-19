"""Building blocks that can be shared across all models.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["get_same_padding", "conv1d_decoder"]

from torch import Tensor, nn


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


def conv1d_decoder(decoder_input_channels: int, num_classes: int) -> nn.Module:
    """Decoder that uses one conv1d layer

    Args:
        num_classes : Number of output classes of the model. It's the size of the vocabulary, excluding the blank symbol.
        decoder_input_channels : Number of input channels of the decoder. That is the number of channels of the features created by the encoder.

    Returns:
        Pytorch model of the decoder
    """
    decoder = nn.Conv1d(
        decoder_input_channels,
        num_classes,
        kernel_size=1,
        bias=True,
    )
    nn.init.xavier_uniform_(decoder.weight, gain=1.0)
    return decoder


class SwapLastDimension(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(-1, -2)


def linear_decoder(
    decoder_input_channels: int, num_classes: int, decoder_dropout: float
) -> nn.Module:
    """Decoder that uses a linear layer with dropout

    Args:
        decoder_dropout: Amount of dropout to be used in the decoder
        decoder_input_channels : Number of input channels of the decoder. That is the number of channels of the features created by the encoder.
        num_classes : Number of output classes of the model. It's the size of the vocabulary, excluding the blank symbol.

    Returns:
        Module that represents the decoder.
    """

    # SwapLastDimension is necessary to
    # change from (batch, time, #vocab) to (batch, #vocab, time)
    # that is expected by the rest of the library
    return nn.Sequential(
        nn.Dropout(decoder_dropout),
        nn.Linear(decoder_input_channels, num_classes),
        SwapLastDimension(),
    )
