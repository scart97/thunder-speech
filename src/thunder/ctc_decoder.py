from typing import List

import torch
from torch import nn


class CTCGreedyDecoder(nn.Module):
    def __init__(self, vocab: List[str], blank: str):
        """This layer applies a greedy ctc decoding operation
        over the outputs of a model, returning the corresponding
        string representation of the output.

        Args:
            vocab : List of strings that form the vocab.
            blank : String that correspond to the blank symbol inside that vocab
        """
        super().__init__()
        self.vocab = vocab
        self.blank = blank

    def forward(self, predictions: torch.Tensor) -> List[str]:
        """
        Args:
            predictions : Tensor of shape (batch, vocab_len, time)

        Returns:
            A list of decoded strings, one for each element in the batch.
        """
        decoded = predictions.argmax(1)
        out_list: List[str] = []

        for element in decoded:
            # Remove consecutive repeated elements
            element = torch.unique_consecutive(element)
            # Map back to string
            out = [self.vocab[d] for d in element]
            # Join prediction into one string
            out = "".join(out)
            # Remove the blank token from output
            out = out.replace(self.blank, "")
            out_list.append(out)

        return out_list
