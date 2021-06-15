# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["BatchTextTransformer", "Vocab"]

from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from thunder.text_processing.tokenizer import char_tokenizer
from thunder.text_processing.vocab import Vocab


class BatchTextTransformer(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        tokenizer=char_tokenizer,
    ):
        """That class is the glue code that uses all of the text processing
        stuff to encode an entire batch of text at once.

        Args:
            vocab : Vocabulary to be used
            tokenize_func : Function that will perform the tokenization of each individual text sample. Defaults to char_tokenizer.
        """
        super().__init__()
        self.vocab = vocab
        self.tokenizer = tokenizer

    def encode(self, items: List[str], return_length: bool = True, device=None):
        tokenized = [self.tokenizer(x) for x in items]
        expanded_tokenized = [self.vocab.add_special_tokens(x) for x in tokenized]
        encoded = [
            self.vocab.numericalize(x).to(device=device) for x in expanded_tokenized
        ]

        encoded_batched = pad_sequence(
            encoded, batch_first=True, padding_value=self.vocab.pad_idx
        )
        if return_length:
            lengths = torch.LongTensor([len(it) for it in encoded]).to(device=device)
            return encoded_batched, lengths
        else:
            return encoded_batched

    @torch.jit.export
    def decode_prediction(self, predictions: torch.Tensor) -> List[str]:
        """
        Args:
            predictions : Tensor of shape (batch, time)

        Returns:
            A list of decoded strings, one for each element in the batch.
        """
        out_list: List[str] = []

        for element in predictions:
            # Remove consecutive repeated elements
            element = torch.unique_consecutive(element)
            # Map back to string
            out = self.vocab.decode_into_text(element)
            # Join prediction into one string
            out = "".join(out)
            # _ is a special char only present on sentencepiece
            out = out.replace("▁", " ")
            out = self.vocab.remove_special_tokens(out)
            out_list.append(out)

        return out_list
