import torch
from pytorch_lightning import seed_everything

from tests.utils import requirescuda
from thunder.blocks import _fourier_matrix, convolution_stft, lengths_to_mask


def test_fourier_transform_matrix():
    for n in [64, 128, 256, 512, 1024]:
        mat1 = torch.fft.fft(torch.eye(n))
        mat2 = _fourier_matrix(n, "cpu")
        assert torch.allclose(mat1, mat2, atol=1e-3)


def test_convolution_stft():
    x = torch.randn(10, 1000)
    window_tensor = torch.hann_window(256, periodic=False)

    stft = convolution_stft(
        x, n_fft=1024, hop_length=512, win_length=256, window=window_tensor
    )
    out_original = torch.stft(
        x,
        n_fft=1024,
        hop_length=512,
        win_length=256,
        window=window_tensor,
        return_complex=False,
    )
    assert torch.allclose(stft, out_original, atol=1e-2)


@requirescuda
def test_convolution_stft_device_move():
    x = torch.randn(10, 1000)
    window_tensor = torch.hann_window(256, periodic=False)

    def apply_op(inp):
        return convolution_stft(
            inp,
            n_fft=1024,
            hop_length=512,
            win_length=256,
            window=window_tensor,
        )

    seed_everything(42)
    outputs_cpu = apply_op(x)

    seed_everything(42)
    outputs_gpu = apply_op(x.cuda())

    assert torch.allclose(outputs_cpu, outputs_gpu.cpu(), atol=1e-3)


def test_lengths_to_mask():
    lengths = torch.tensor([8, 5, 4, 3, 1])
    mask = lengths_to_mask(lengths, max_length=8)
    expected = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
        ]
    )
    assert torch.allclose(mask, expected)
