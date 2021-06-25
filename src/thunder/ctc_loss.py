"""Functionality to calculate the ctc loss.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["calculate_ctc"]

from torch import Tensor
from torch.nn.functional import ctc_loss, log_softmax


def calculate_ctc(
    probabilities: Tensor, y: Tensor, prob_lens: Tensor, y_lens: Tensor, blank_idx: int
) -> Tensor:
    """Calculates the ctc loss based on model probabilities (also called emissions) and
    labels.

    Args:
        probabilities : Output of the model, before any softmax operation. Shape [batch, #vocab, time]
        y : Tensor containing the corresponding labels. Shape [batch]
        prob_lens : Lengths of each element in the input, normalized so that the max length is 1.0. Shape [batch]
        y_lens : Lenghts of each element in the output. Should NOT be normalized.
        blank_idx : Index of the blank token in the vocab.

    Returns:
        Loss tensor that can be backpropagated.
    """
    # Change from (batch, #vocab, time) to (time, batch, #vocab)
    probabilities = probabilities.permute(2, 0, 1)
    logprobs = log_softmax(probabilities, dim=2)
    # Calculate the logprobs correct length based on the
    # normalized original lengths
    prob_lens = (prob_lens * logprobs.shape[0]).long()

    return ctc_loss(
        logprobs,
        y,
        prob_lens,
        y_lens,
        blank=blank_idx,
        reduction="mean",
        zero_infinity=True,
    )
