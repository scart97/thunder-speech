from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from thunder.text_processing.preprocess import lower_text, normalize_text
from thunder.text_processing.tokenizer import char_tokenizer
from thunder.text_processing.vocab import Vocab
from thunder.utils import chain_calls


class BatchTextTransformer:
    def __init__(
        self,
        vocab: Vocab,
        tokenize_func=char_tokenizer,
        preprocessing_transforms=chain_calls(lower_text, normalize_text),
        after_tokenize=None,
        after_numericalize=None,
    ):
        """That class is the glue code that uses all of the text processing
        stuff to encode an entire batch of text at once.

        Args:
            vocab : Vocabulary to be used
            tokenize_func : Function that will perform the tokenization of each individual text sample. Defaults to char_tokenizer.
            preprocessing_transforms : Functions that will be applied before tokenization, as the first step. Defaults to chain_calls(lower_text, normalize_text).
            after_tokenize : Functions to be applied after the tokenization but before numericalization. Defaults to None.
            after_numericalize : Functions to be applied at the end of the pipeline. Defaults to torch.LongTensor.
        """
        self.vocab = vocab
        self.tokenize_func = tokenize_func
        self.preprocessing_transforms = preprocessing_transforms
        self.after_tokenize = after_tokenize
        self.after_numericalize = after_numericalize

    def encode(self, items: List[str], return_length: bool = True, device=None):
        if self.preprocessing_transforms is not None:
            items = [self.preprocessing_transforms(x) for x in items]

        tokenized = [self.tokenize_func(x) for x in items]

        if self.after_tokenize is not None:
            tokenized = [self.after_tokenize(x) for x in tokenized]

        expanded_tokenized = [self.vocab.add_special_tokens(x) for x in tokenized]
        encoded = [self.vocab.numericalize(it) for it in expanded_tokenized]
        encoded = [torch.LongTensor(it).to(device=device) for it in encoded]
        if self.after_numericalize is not None:
            encoded = [self.after_numericalize(x) for x in encoded]

        encoded = pad_sequence(
            encoded, batch_first=True, padding_value=self.vocab.pad_idx
        )
        if return_length:
            lengths = torch.LongTensor([len(it) for it in expanded_tokenized]).to(
                device=device
            )
            return encoded, lengths
        else:
            return encoded
