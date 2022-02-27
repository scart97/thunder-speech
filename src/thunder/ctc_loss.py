"""Functionality to calculate the ctc loss.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["calculate_ctc"]

from torch import Tensor
from torch.nn.functional import ctc_loss, log_softmax


def calculate_ctc(
    probabilities: Tensor,
    y: Tensor,
    prob_lengths: Tensor,
    y_lengths: Tensor,
    blank_idx: int,
) -> Tensor:
    """Calculates the ctc loss based on model probabilities (also called emissions) and
    labels.

    Args:
        probabilities: Output of the model, before any softmax operation. Shape [batch, #vocab, time]
        y: Tensor containing the corresponding labels. Shape [batch]
        prob_lengths: Lengths of each element in the input. Shape [batch]
        y_lengths: Lenghts of each element in the output. Should NOT be normalized.
        blank_idx: Index of the blank token in the vocab.

    Returns:
        Loss tensor that can be backpropagated.
    """
    # Change from (batch, #vocab, time) to (time, batch, #vocab)
    probabilities = probabilities.permute(2, 0, 1)
    logprobs = log_softmax(probabilities, dim=2)

    return ctc_loss(
        logprobs,
        y,
        prob_lengths.long(),
        y_lengths,
        blank=blank_idx,
        reduction="mean",
        zero_infinity=True,
    )
