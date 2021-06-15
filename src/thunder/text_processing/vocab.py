# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["Vocab"]

from typing import List

import torch
from torch import nn


class SimpleVocab(nn.Module):
    def __init__(
        self,
        initial_vocab_tokens: List[str],
        blank_token: str = "<blank>",
    ):
        """Class that represents a vocabulary, with the related methods
        to numericalize a sequence of tokens into numbers, and do the
        reverse mapping of numbers back to tokens.
        This is a simplified vocabulary that only has the ctc blank as
        special token.

        Args:
            initial_vocab_tokens : Basic list of tokens that will be part of the vocabulary. DO NOT INCLUDE SPECIAL TOKENS THERE. Even the blank is automatically added by the class. Check [`docs`](https://scart97.github.io/thunder-speech/quick%20reference%20guide/#how-to-get-the-initial_vocab_tokens-from-my-dataset)
            blank_token : Token that will represent the special ctc blank.
        """
        super().__init__()
        # There's no problem if the blank_idx == pad_idx
        self.blank_token = blank_token

        self.itos = initial_vocab_tokens + [blank_token]
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.blank_idx = self.itos.index(self.blank_token)
        self.pad_idx = self.blank_idx

    def __len__(self):
        return len(self.itos)

    def numericalize(self, tokens: List[str]) -> torch.Tensor:
        """Function to transform a list of tokens into the corresponding numeric representation.

        Args:
            tokens : A single list of tokens to be transformed

        Returns:
            The corresponding numeric representation
        """
        # When in nemo style vocab, there's no unknown token
        # So we filter out all of the tokens not in the vocab
        tokens = filter(lambda x: x in self.itos, tokens)
        return torch.tensor([self.stoi.get(it) for it in tokens], dtype=torch.long)

    @torch.jit.export
    def decode_into_text(self, indices: torch.Tensor) -> List[str]:
        """Function to transform back a list of numbers into the corresponding
        tokens.

        Args:
            indices : Numeric representation. Usually is the result of the model, after a greedy decoding

        Returns:
            Corresponding tokens
        """
        return [self.itos[it] for it in indices]

    @torch.jit.export
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Function to add the special start and end tokens to some
        tokenized text.

        Args:
            tokens : Tokenized text

        Returns:
            Text with the special tokens added.
        """
        return tokens

    @torch.jit.export
    def remove_special_tokens(self, text: str) -> str:
        """Function to remove the special tokens from the prediction.

        Args:
            text : Decoded text

        Returns:
            Text with the special tokens removed.
        """
        return text.replace(self.blank_token, "")


class Vocab(nn.Module):
    def __init__(
        self,
        initial_vocab_tokens: List[str],
        pad_token: str = "<pad>",
        unknown_token: str = "<unk>",
        start_token: str = "<bos>",
        end_token: str = "<eos>",
    ):
        """Class that represents a vocabulary, with the related methods
        to numericalize a sequence of tokens into numbers, and do the
        reverse mapping of numbers back to tokens.

        Args:
            initial_vocab_tokens : Basic list of tokens that will be part of the vocabulary. DO NOT INCLUDE SPECIAL TOKENS THERE. Even the blank is automatically added by the class. Check [`docs`](https://scart97.github.io/thunder-speech/quick%20reference%20guide/#how-to-get-the-initial_vocab_tokens-from-my-dataset)
            pad_token : Token that will represent padding, and also act as the ctc blank.
            unknown_token : Token that will represent unknown elements. Notice that this is different than the blank used by ctc.
            start_token : Token that will represent the beginning of the sequence.
            end_token : Token that will represent the end of the sequence.
        """
        super().__init__()
        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token
        # There's no problem if the blank_idx == pad_idx
        self.blank_token = self.pad_token

        self.itos = initial_vocab_tokens + [
            pad_token,
            unknown_token,
            start_token,
            end_token,
        ]
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.update_special_idx()

    def update_special_idx(self):
        self.pad_idx = self.itos.index(self.pad_token)
        self.unknown_idx = self.itos.index(self.unknown_token)
        self.start_idx = self.itos.index(self.start_token)
        self.end_idx = self.itos.index(self.end_token)
        self.blank_idx = self.itos.index(self.blank_token)

    def __len__(self):
        return len(self.itos)

    def numericalize(self, tokens: List[str]) -> torch.Tensor:
        """Function to transform a list of tokens into the corresponding numeric representation.

        Args:
            tokens : A single list of tokens to be transformed

        Returns:
            The corresponding numeric representation
        """
        return torch.tensor(
            [self.stoi.get(it, self.unknown_idx) for it in tokens], dtype=torch.long
        )

    @torch.jit.export
    def decode_into_text(self, indices: torch.Tensor) -> List[str]:
        """Function to transform back a list of numbers into the corresponding
        tokens.

        Args:
            indices : Numeric representation. Usually is the result of the model, after a greedy decoding

        Returns:
            Corresponding tokens
        """
        return [self.itos[it] for it in indices]

    @torch.jit.export
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Function to add the special start and end tokens to some
        tokenized text.

        Args:
            tokens : Tokenized text

        Returns:
            Text with the special tokens added.
        """
        return [self.start_token] + tokens + [self.end_token]

    @torch.jit.export
    def remove_special_tokens(self, text: str) -> str:
        """Function to remove the special tokens from the prediction.

        Args:
            text : Decoded text

        Returns:
            Text with the special tokens removed.
        """
        text = text.replace(self.blank_token, "")
        text = text.replace(self.pad_token, "")
        text = text.replace(self.start_token, "")
        text = text.replace(self.end_token, "")
        return text
