# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = [
    "BPETokenizer",
    "SentencepieceModelFile",
    "train_sentencepiece_model",
    "word_tokenizer",
    "char_tokenizer",
    "get_most_frequent_tokens",
]

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
from warnings import warn

import sentencepiece


class BPETokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(model_path)

    def __call__(self, text: str) -> List[str]:
        return self.tokenizer.encode_as_pieces(text)


@dataclass
class SentencepieceModelFile:
    """Dataclass that represents the data from the trained sentencepiece model,
    processed so that it can be integrated with the models.
    """

    model_path: str
    vocabulary_tokens: List[str]

    @classmethod
    def from_folder(cls, output_dir: str) -> "SentencepieceModelFile":
        """Load the data from a folder that contains the `tokenizer.vocab`
        and `tokenizer.model` outputs from sentencepiece.

        Args:
            output_dir : Output directory of the sentencepiece training, that contains the required files.

        Returns:
            Instance of `SentencepieceModelFile` with the corresponding data loaded.
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

        return cls(f"{output_dir}/tokenizer.model", vocab)


def train_sentencepiece_model(
    data_file: str,
    vocab_size: int,
    output_dir: str,
    sample_size: int = -1,
    do_lower_case: bool = True,
    tokenizer_type: str = "unigram",
    character_coverage: float = 1.0,
    train_extremely_large_corpus: bool = False,
    max_sentencepiece_length: int = -1,
) -> SentencepieceModelFile:
    """
    Creates sentence piece tokenizer model from data file.
    This is a direct port of `create_spt_model` present on the NEMO
    toolkit (nemo/collections/common/tokenizers/sentencepiece_tokenizer.py)
    Args:
        data_file: text file containing the sentences that will be used to train the model
        vocab_size: maximum vocabulary size
        output_dir: folder to save created tokenizer model and vocab
        sample_size: maximum number of sentences the trainer loads. -1 means to use all the data.
        do_lower_case: if text should be lower cased before tokenizer model is created
        character_coverage: float value between 0 and 1 (as a percentage). For languages with a vast charset,
            can be < 1.0, but for all other languages, it should be set as 1.0
        train_extremely_large_corpus: If training on huge datasets, pass this flag to allow SentencePiece
            to build the tokenizer.
        max_sentencepiece_length: Limits the maximum length of the SentencePiece subword that can be constructed.
            By default, no limit is placed.
    """
    data_file = Path(data_file)
    if not data_file or not data_file.exists():
        raise ValueError(f"data_file must be valid file path, but got {data_file}")

    output_dir = Path(output_dir)
    if (output_dir / "tokenizer.model").exists():
        warn(
            "There's already a trained sentencepiece model at the output directory. Skipping train."
        )
        return SentencepieceModelFile.from_folder(output_dir)

    output_dir.mkdir(exist_ok=True)

    cmd = (
        f"--input={data_file} --model_prefix={output_dir}/tokenizer "
        f"--vocab_size={vocab_size} "
        f"--shuffle_input_sentence=true --hard_vocab_limit=false "
        f"--model_type={tokenizer_type} "
        f"--character_coverage={character_coverage}"
    )

    if do_lower_case:
        cmd += " --normalization_rule_name=nmt_nfkc_cf"

    if sample_size > 0:
        cmd += f" --input_sentence_size={sample_size}"

    if train_extremely_large_corpus:
        cmd += " --train_extremely_large_corpus=true"

    if max_sentencepiece_length >= 0:
        cmd += f" --max_sentencepiece_length={max_sentencepiece_length}"

    sentencepiece.SentencePieceTrainer.Train(cmd)

    return SentencepieceModelFile.from_folder(output_dir)


def word_tokenizer(text: str) -> List[str]:
    """Tokenize input text splitting into words

    Args:
        text : Input text

    Returns:
        Tokenized text
    """
    return text.split()


def char_tokenizer(text: str) -> List[str]:
    """Tokenize input text splitting into characters

    Args:
        text : Input text

    Returns:
        Tokenized text
    """
    return list(text)


def get_most_frequent_tokens(
    corpus: str,
    tokenize_function: Callable,
    minimum_frequency: int = 1,
    max_number_of_tokens: Optional[int] = None,
) -> List[str]:
    """Helper function to get the most frequent tokens from a text corpus.

    Args:
        corpus : Text corpus to be used, this is a long string containing all of your text
        tokenize_function : Same tokenizer function that will be used during training
        minimum_frequency : Remove any token with frequency less than that. Defaults to 1.
        max_number_of_tokens : Optionally limit to the K most frequent tokens. Defaults to None.

    Returns:
        All of the unique, most frequent tokens, ordered by frequency.
    """

    tokenized = tokenize_function(corpus)
    token_counter = Counter(tokenized)
    output_tokens = []
    for token, count in token_counter.most_common(max_number_of_tokens):
        if count >= minimum_frequency:
            output_tokens.append(token)
    return output_tokens
