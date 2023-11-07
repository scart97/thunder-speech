import pytest

import torch
from hypothesis import assume, given
from hypothesis.strategies import text
from transformers import Wav2Vec2CTCTokenizer

from tests.utils import mark_slow
from thunder.huggingface.compatibility import _tok_to_transform


@pytest.fixture(scope="session")
def hf_tokenizer():
    return Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")


@pytest.fixture(scope="session")
def thunder_tokenizer(hf_tokenizer):
    return _tok_to_transform(hf_tokenizer)


@mark_slow
@given(inp=text(min_size=1))
def test_tok(inp: str, hf_tokenizer, thunder_tokenizer):
    assume("|" not in inp)
    out1 = hf_tokenizer([inp], return_tensors="pt").input_ids
    out2 = thunder_tokenizer.encode([inp], return_length=False)
    assert torch.allclose(out1, out2)

@pytest.mark.xfail
@mark_slow
def test_pretrained_problematic_tokens():
    tok = Wav2Vec2CTCTokenizer.from_pretrained(
        "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2"
    )
    text_transform = _tok_to_transform(tok)
    assert text_transform.num_tokens == 44
    assert text_transform.vocab.start_token is None
