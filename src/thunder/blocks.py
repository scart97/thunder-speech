"""Building blocks that can be shared across all models.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["convolution_stft", "get_same_padding", "conv1d_decoder"]

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def convolution_stft(
    input_data,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    window: torch.Tensor = torch.hann_window(1024, periodic=False),
    center: bool = True,
    return_complex: bool = False,
):
    """Implements the stft operation using the convolution method. This is one alternative
    to make possible to export code using this operation to onnx and arm based environments.
    The signature shuld follow the same as torch.stft, making it possible to just swap the two.
    The code is based on https://github.com/pseeth/torch-stft
    """
    assert n_fft >= win_length
    pad_amount = int(n_fft / 2)

    fourier_basis = torch.fft.fft(torch.eye(n_fft))

    cutoff = int((n_fft / 2 + 1))
    fourier_basis = torch.vstack(
        [torch.real(fourier_basis[:cutoff, :]), torch.imag(fourier_basis[:cutoff, :])]
    )
    forward_basis = fourier_basis[:, None, :].float()

    window_pad = (n_fft - win_length) // 2
    window_pad2 = n_fft - (window_pad + win_length)
    fft_window = torch.nn.functional.pad(window, [window_pad, window_pad2])
    # window the bases
    forward_basis *= fft_window
    forward_basis = forward_basis.float()

    num_batches = input_data.shape[0]
    num_samples = input_data.shape[-1]

    # similar to librosa, reflect-pad the input
    input_data = input_data.view(num_batches, 1, num_samples)

    input_data = F.pad(
        input_data.unsqueeze(1),
        (pad_amount, pad_amount, 0, 0),
        mode="reflect",
    )
    input_data = input_data.squeeze(1)

    forward_transform = F.conv1d(
        input_data, forward_basis, stride=hop_length, padding=0
    )

    cutoff = int((n_fft / 2) + 1)
    real_part = forward_transform[:, :cutoff, :]
    imag_part = forward_transform[:, cutoff:, :]
    return torch.stack((real_part, imag_part), dim=-1)


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
    """Layer that swap the last two dimensions of the data."""

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
