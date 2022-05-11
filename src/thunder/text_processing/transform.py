"""
Process batched text
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["BatchTextTransformer"]

from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from thunder.text_processing.tokenizer import BPETokenizer, char_tokenizer
from thunder.text_processing.vocab import Vocabulary


class BatchTextTransformer(nn.Module):
    def __init__(
        self,
        tokens: List[str],
        blank_token: str = "<blank>",
        pad_token: str = None,
        unknown_token: str = None,
        start_token: str = None,
        end_token: str = None,
        sentencepiece_model: Optional[str] = None,
        custom_tokenizer_function: Callable[[str], List[str]] = None,
    ):
        """That class is the glue code that uses all of the text processing
        functions to encode/decode an entire batch of text at once.


        Args:
            tokens: Basic list of tokens that will be part of the vocabulary.
            blank_token: Check [`Vocabulary`][thunder.text_processing.vocab.Vocabulary]
            pad_token: Check [`Vocabulary`][thunder.text_processing.vocab.Vocabulary]
            unknown_token: Check [`Vocabulary`][thunder.text_processing.vocab.Vocabulary]
            start_token: Check [`Vocabulary`][thunder.text_processing.vocab.Vocabulary]
            end_token: Check [`Vocabulary`][thunder.text_processing.vocab.Vocabulary]
            sentencepiece_model: Path to sentencepiece .model file, if applicable.
            custom_tokenizer_function: Allows the use of a custom function to tokenize the input.
        """
        super().__init__()
        self.vocab = Vocabulary(
            tokens,
            blank_token,
            pad_token,
            unknown_token,
            start_token,
            end_token,
        )

        if custom_tokenizer_function:
            self.tokenizer = custom_tokenizer_function
        elif sentencepiece_model:
            self.tokenizer = BPETokenizer(sentencepiece_model)
        else:
            self.tokenizer = char_tokenizer

    def encode(
        self, items: List[str], return_length: bool = True, device=None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Encode a list of texts to a padded pytorch tensor

        Args:
            items: List of texts to be processed
            return_length: optionally also return the length of each element in the encoded tensor
            device: optional device to create the tensors with

        Returns:
            Either the encoded tensor, or a tuple with tensor and lengths.
        """
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
            predictions: Tensor of shape (batch, time)
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
            # | is a special char used by huggingface as space
            out = out.replace("|", " ")
            out = self.vocab.remove_special_tokens(out)
            out_list.append(out)

        return out_list

    @classmethod
    def from_sentencepiece(cls, output_dir: str) -> "BatchTextTransformer":
        """Load the data from a folder that contains the `tokenizer.vocab`
        and `tokenizer.model` outputs from sentencepiece.

        Args:
            output_dir: Output directory of the sentencepiece training, that contains the required files.

        Returns:
            Instance of `BatchTextTransformer` with the corresponding data loaded.
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
            tokens=vocab,
            sentencepiece_model=f"{output_dir}/tokenizer.model",
        )

    @property
    def num_tokens(self):
        return len(self.vocab.itos)
