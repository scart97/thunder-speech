"""
Classes that represent the vocabulary used by the model.
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = ["Vocabulary"]

from typing import List, Optional

import torch
from torch import nn


class Vocabulary(nn.Module):
    def __init__(
        self,
        tokens: List[str],
        blank_token: str = "<blank>",
        pad_token: Optional[str] = None,
        unknown_token: Optional[str] = None,
        start_token: Optional[str] = None,
        end_token: Optional[str] = None,
    ):
        """Class that represents a vocabulary, with the related methods
        to numericalize a sequence of tokens into numbers, and do the
        reverse mapping of numbers back to tokens.

        Args:
            tokens: Basic list of tokens that will be part of the vocabulary. Check [`docs`](https://scart97.github.io/thunder-speech/quick%20reference%20guide/#how-to-get-the-tokens-from-my-dataset)
            blank_token: Token that will represent the ctc blank.
            pad_token: Token that will represent padding, might also act as the ctc blank.
            unknown_token: Token that will represent unknown elements. Notice that this is different than the blank used by ctc.
            start_token: Token that will represent the beginning of the sequence.
            end_token: Token that will represent the end of the sequence.
        """
        super().__init__()
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token
        self.blank_token = blank_token
        self.pad_token = pad_token or blank_token

        self.itos = tokens
        self._maybe_add_token(blank_token)
        self._maybe_add_token(pad_token)
        self._maybe_add_token(unknown_token)
        self._maybe_add_token(start_token)
        self._maybe_add_token(end_token)

        self.stoi = {token: i for i, token in enumerate(self.itos)}

        self.blank_idx = self.itos.index(self.blank_token)
        self.pad_idx = self.itos.index(self.pad_token)
        self._unk_idx = -1
        if self.unknown_token is not None:
            self._unk_idx = self.itos.index(self.unknown_token)

    def _maybe_add_token(self, token: Optional[str]):
        # Only adds tokens if they are not optional
        # and are not included in the vocabulary already
        if token and (token not in self.itos):
            self.itos = self.itos + [token]

    def numericalize(self, tokens: List[str]) -> torch.Tensor:
        """Function to transform a list of tokens into the corresponding numeric representation.

        Args:
            tokens: A single list of tokens to be transformed

        Returns:
            The corresponding numeric representation
        """
        if self.unknown_token is None:
            # When in there's no unknown token
            # we filter out all of the tokens not in the vocab
            tokens = [t for t in tokens if t in self.itos]
        return torch.tensor(
            [self.stoi.get(it, self._unk_idx) for it in tokens], dtype=torch.long
        )

    @torch.jit.export
    def decode_into_text(self, indices: torch.Tensor) -> List[str]:
        """Function to transform back a list of numbers into the corresponding
        tokens.

        Args:
            indices: Numeric representation. Usually is the result of the model, after a greedy decoding

        Returns:
            Corresponding tokens
        """
        return [self.itos[it] for it in indices]

    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Function to add the special start and end tokens to some
        tokenized text.

        Args:
            tokens: Tokenized text

        Returns:
            Text with the special tokens added.
        """
        if self.start_token is not None:
            tokens = [self.start_token] + tokens
        if self.end_token is not None:
            tokens = tokens + [self.end_token]
        return tokens

    @torch.jit.export
    def remove_special_tokens(self, text: str) -> str:
        """Function to remove the special tokens from the prediction.

        Args:
            text: Decoded text

        Returns:
            Text with the special tokens removed.
        """
        text = text.replace(self.blank_token, "")
        text = text.replace(self.pad_token, "")
        if self.start_token is not None:
            text = text.replace(self.start_token, "")
        if self.end_token is not None:
            text = text.replace(self.end_token, "")
        return text
