from string import ascii_lowercase
from tempfile import TemporaryDirectory

import pytest

import torch

from thunder.ctc_decoder import CTCGreedyDecoder


@pytest.fixture
def simple_vocab():
    return list(ascii_lowercase) + ["$"]


@pytest.fixture
def blank_input(simple_vocab):
    # A tensor full of blank probabilities
    x = torch.zeros(1, len(simple_vocab), 100)
    x[:, -1, :] = 1
    return x


def test_decoder_remove_blanks(simple_vocab, blank_input):
    decoder = CTCGreedyDecoder(simple_vocab, blank="$")
    out = decoder(blank_input)

    assert len(out) == 1
    assert isinstance(out, list)
    assert type(out[0]) is str
    assert out[0] == ""


def test_decoder_simple_sequence(simple_vocab, blank_input):
    decoder = CTCGreedyDecoder(simple_vocab, blank="$")

    blank_input[:, 0, :10] = 2  # a
    blank_input[:, 1, 15:20] = 2  # b

    out = decoder(blank_input)

    assert len(out) == 1
    assert isinstance(out, list)
    assert type(out[0]) is str
    assert out[0] == "ab"


def test_decoder_repeat_same_element(simple_vocab, blank_input):

    decoder = CTCGreedyDecoder(simple_vocab, blank="$")

    blank_input[:, 0, :10] = 2  # a
    blank_input[:, 0, 15:20] = 2  # a
    out = decoder(blank_input)

    assert len(out) == 1
    assert isinstance(out, list)
    assert type(out[0]) is str
    assert out[0] == "aa"


@pytest.mark.xfail
def test_decoder_trace(simple_vocab, blank_input):
    decoder = CTCGreedyDecoder(simple_vocab, blank="$")
    decoder.eval()
    decoder_trace = torch.jit.trace(decoder, (blank_input))
    assert torch.allclose(decoder(blank_input), decoder_trace(blank_input))


def test_decoder_script(simple_vocab, blank_input):
    decoder = CTCGreedyDecoder(simple_vocab, blank="$")
    decoder.eval()
    decoder_script = torch.jit.script(decoder)
    blank_input[:, 0, :10] = 2
    assert decoder(blank_input)[0] == decoder_script(blank_input)[0]


@pytest.mark.xfail
def test_decoder_onnx(simple_vocab, blank_input):
    # Only torch script is supported, onnx might work or not
    # This is because this layer mixes tensors with strings directly
    decoder = CTCGreedyDecoder(simple_vocab, blank="$")
    decoder.eval()
    with TemporaryDirectory() as export_path:
        torch.onnx.export(
            decoder,
            (blank_input),
            f"{export_path}/decoder.onnx",
            verbose=True,
        )
