from typing import List

from thunder.text_processing.tokenizer import Token


class Vocab:
    def __init__(
        self,
        initial_vocab_tokens: List[Token],
        pad_token: Token = "<pad>",
        unknown_token: Token = "<unk>",
        start_token: Token = "<bos>",
        end_token: Token = "<eos>",
    ):
        """Class that represents a vocabulary, with the related methods
        to numericalize a sequence of tokens into numbers, and do the
        reverse mapping of numbers back to tokens.

        Args:
            initial_vocab_tokens : Basic list of tokens that will be part of the vocabulary. DO NOT INCLUDE SPECIAL TOKENS THERE. Even the blank is automatically added by the class.
            pad_token : Token that will represent padding.
            unknown_token : Token that will represent unknown elements. Notice that this is different than the blank used by ctc.
            start_token : Token that will represent the beginning of the sequence.
            end_token : Token that will represent the end of the sequence.
        """
        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token

        self.itos = [
            pad_token,
            unknown_token,
            start_token,
            end_token,
        ] + initial_vocab_tokens
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def numericalize(self, tokens: List[Token]) -> List[int]:
        """Function to transform a list of tokens into the corresponding numeric representation.

        Args:
            tokens : A single list of tokens to be transformed

        Returns:
            The corresponding numeric representation
        """
        return [self.stoi.get(it, self.unknown_idx) for it in tokens]

    def decode_into_text(self, indices: List[int]) -> List[Token]:
        """Function to transform back a list of numbers into the corresponding
        tokens.

        Args:
            indices : Numeric representation. Usually is the result of the model, after a greedy decoding

        Returns:
            Corresponding tokens
        """
        return [self.itos[it] for it in indices]

    def add_special_tokens(self, tokens: List[Token]) -> List[Token]:
        """Function to add the special start and end tokens to some
        tokenized text.

        Args:
            tokens : Tokenized text

        Returns:
            Text with the special tokens added.
        """
        return [self.start_token] + tokens + [self.end_token]

    @property
    def unknown_idx(self):
        return self.itos.index(self.unknown_token)

    @property
    def start_idx(self):
        return self.itos.index(self.start_token)

    @property
    def end_idx(self):
        return self.itos.index(self.end_token)

    @property
    def pad_idx(self):
        return self.itos.index(self.pad_token)

    @property
    def blank_token(self):
        return self.pad_token

    @property
    def blank_idx(self):
        return self.pad_idx
