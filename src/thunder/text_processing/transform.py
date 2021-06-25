# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["BatchTextTransformer", "Vocab"]

from typing import List, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from thunder.text_processing.tokenizer import BPETokenizer, char_tokenizer
from thunder.text_processing.vocab import SimpleVocab, Vocab


class BatchTextTransformer(nn.Module):
    def __init__(
        self,
        initial_vocab_tokens: List[str],
        simple_vocab: bool,
        sentencepiece_model: Optional[str] = None,
    ):
        """That class is the glue code that uses all of the text processing
        functions to encode/decode an entire batch of text at once.

        Args:
            initial_vocab_tokens : List of tokens to create the vocabulary, special tokens should not be included here.
            simple_vocab : Controls if the used vocabulary will only have the blank token or more additional special tokens.
            sentencepiece_model: Path to sentencepiece .model file, if applicable
        """
        super().__init__()
        self.vocab = (
            SimpleVocab(initial_vocab_tokens)
            if simple_vocab
            else Vocab(initial_vocab_tokens)
        )
        self.tokenizer = (
            BPETokenizer(sentencepiece_model) if sentencepiece_model else char_tokenizer
        )

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
    def decode_prediction(
        self, predictions: torch.Tensor, remove_repeated: bool = True
    ) -> List[str]:
        """
        Args:
            predictions : Tensor of shape (batch, time)
            remove_repeated: controls if repeated elements without a blank between them will be removed while decoding

        Returns:
            A list of decoded strings, one for each element in the batch.
        """
        out_list: List[str] = []

        for element in predictions:
            # Remove consecutive repeated elements
            if remove_repeated:
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
