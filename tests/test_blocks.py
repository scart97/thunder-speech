import torch

from thunder.blocks import convolution_stft


def test_convolution_stft():
    x = torch.randn(10, 1000)
    window_tensor = torch.hann_window(256, periodic=False)

    stft = convolution_stft(
        x, n_fft=1024, hop_length=512, win_length=256, window=window_tensor
    )
    out_original = torch.stft(
        x, n_fft=1024, hop_length=512, win_length=256, window=window_tensor
    )
    assert torch.allclose(stft, out_original, atol=1e-3)
