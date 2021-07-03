# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["BatchTextTransformer", "TextTransformConfig"]

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from thunder.text_processing.tokenizer import BPETokenizer, char_tokenizer
from thunder.text_processing.vocab import SimpleVocab, Vocab


@dataclass
class TextTransformConfig:
    """Configuration to create [`BatchTextTransformer`][thunder.text_processing.transform.BatchTextTransformer]

    Attributes:
        initial_vocab_tokens: List of tokens to create the vocabulary, special tokens should not be included here. required.
        simple_vocab: Controls if the used vocabulary will only have the blank token or more additional special tokens. defaults to `False`.
        sentencepiece_model: Path to sentencepiece .model file, if applicable.
    """

    initial_vocab_tokens: List[str]
    simple_vocab: bool = False
    sentencepiece_model: Optional[str] = None

    @classmethod
    def from_sentencepiece(cls, output_dir: str) -> "TextTransformConfig":
        """Load the data from a folder that contains the `tokenizer.vocab`
        and `tokenizer.model` outputs from sentencepiece.

        Args:
            output_dir : Output directory of the sentencepiece training, that contains the required files.

        Returns:
            Instance of `TextTransformConfig` with the corresponding data loaded.
        """
        special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
        vocab = []

        with open(f"{output_dir}/tokenizer.vocab", "r") as f:
            # Read tokens from each line and parse for vocab
            for line in f:
                piece = line.split("\t")[0]
                if piece in special_tokens:
                    # skip special tokens
                    continue
                vocab.append(piece)

        return cls(
            initial_vocab_tokens=vocab,
            sentencepiece_model=f"{output_dir}/tokenizer.model",
        )


class BatchTextTransformer(nn.Module):
    def __init__(self, cfg: TextTransformConfig):
        """That class is the glue code that uses all of the text processing
        functions to encode/decode an entire batch of text at once.

        Args:
            cfg: required config to create instance
        """
        super().__init__()
        self.vocab = (
            SimpleVocab(cfg.initial_vocab_tokens)
            if cfg.simple_vocab
            else Vocab(cfg.initial_vocab_tokens)
        )
        self.tokenizer = (
            BPETokenizer(cfg.sentencepiece_model)
            if cfg.sentencepiece_model
            else char_tokenizer
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
