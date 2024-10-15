"""Building blocks that can be shared across all models.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = [
    "MultiSequential",
    "Masked",
    "normalize_tensor",
    "lengths_to_mask",
    "get_same_padding",
    "conv1d_decoder",
    "SwapLastDimension",
    "linear_decoder",
]

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class MultiSequential(nn.Sequential):
    """nn.Sequential equivalent with 2 inputs/outputs"""

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for module in self.children():
            audio, audio_lengths = module(audio, audio_lengths)
        return audio, audio_lengths


class Masked(nn.Module):
    """Wrapper to mix normal modules with others that take 2 inputs"""

    def __init__(self, *layers):
        super().__init__()
        self.layer = nn.Sequential(*layers)

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.layer(audio), audio_lengths


def normalize_tensor(
    input_values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    div_guard: float = 1e-7,
    dim: int = -1,
) -> torch.Tensor:
    """Normalize tensor values, optionally using some mask to define the valid region.

    Args:
        input_values: input tensor to be normalized
        mask: Optional mask describing the valid elements.
        div_guard: value used to prevent division by zero when normalizing.
        dim: dimension used to calculate the mean and variance.

    Returns:
        Normalized tensor
    """
    # Vectorized implementation of (x - x.mean()) / x.std() considering only the valid mask
    if mask is not None:
        # Making sure the elements outside the mask are zero, to have the correct mean/std
        input_values = torch.masked_fill(input_values, ~mask.type(torch.bool), 0.0)
        # Number of valid elements
        num_elements = mask.sum(dim=dim, keepdim=True).detach()
        # Mean is sum over number of elements
        x_mean = input_values.sum(dim=dim, keepdim=True).detach() / num_elements
        # std numerator: sum of squared differences to the mean
        numerator = (input_values - x_mean).pow(2).sum(dim=dim, keepdim=True).detach()
        x_std = (numerator / num_elements).sqrt()
        # using the div_guard to prevent division by zero
        normalized = (input_values - x_mean) / (x_std + div_guard)
        # Cleaning elements outside of valid mask
        return torch.masked_fill(normalized, ~mask.type(torch.bool), 0.0)

    mean = input_values.mean(dim=dim, keepdim=True).detach()
    std = (input_values.var(dim=dim, keepdim=True).detach() + div_guard).sqrt()
    return (input_values - mean) / std


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Convert from integer lengths of each element to mask representation

    Args:
        lengths: lengths of each element in the batch
        max_length: maximum length expected. Can be greater than lengths.max()

    Returns:
        Corresponding boolean mask indicating the valid region of the tensor.
    """
    lengths = lengths.type(torch.long)
    mask = torch.arange(max_length, device=lengths.device).expand(
        lengths.shape[0], max_length
    ) < lengths.unsqueeze(1)
    return mask


def get_same_padding(kernel_size: int, stride: int, dilation: int) -> int:
    """Calculates the padding size to obtain same padding.
        Same padding means that the output will have the
        shape input_shape / stride. That means, for
        stride = 1 the output shape is the same as the input,
        and stride = 2 gives an output that is half of the
        input shape.

    Args:
        kernel_size: convolution kernel size. Only tested to be correct with odd values.
        stride: convolution stride
        dilation: convolution dilation

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
        num_classes: Number of output classes of the model. It's the size of the vocabulary, excluding the blank symbol.
        decoder_input_channels: Number of input channels of the decoder. That is the number of channels of the features created by the encoder.

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
        decoder_input_channels: Number of input channels of the decoder. That is the number of channels of the features created by the encoder.
        num_classes: Number of output classes of the model. It's the size of the vocabulary, excluding the blank symbol.

    Returns:
        Module that represents the decoder.
    """

    # SwapLastDimension is necessary to
    # change from (batch, time, #vocab) to (batch, #vocab, time)
    # that is expected by the rest of the library
    return nn.Sequential(
        SwapLastDimension(),
        nn.Dropout(decoder_dropout),
        nn.Linear(decoder_input_channels, num_classes),
        SwapLastDimension(),
    )
