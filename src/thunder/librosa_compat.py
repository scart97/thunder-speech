__all__ = ["hz_to_mel", "mel_to_hz", "mel_frequencies", "create_fb_matrix"]

import math
import warnings
from typing import Optional

import torch
from torch import Tensor


def hz_to_mel(frequencies: int, htk: bool = False) -> int:
    """This is a direct port of librosa.core.conver.hz_to_mel to work with torchaudio.

    Args:
        frequencies : Frequencies to convert
        htk : Use htk formula for conversion or not. Defaults to False.

    Returns:
        Frequencies in mel scale.
    """
    if htk:
        return 2595.0 * math.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + math.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels: Tensor, htk: bool = False) -> Tensor:
    """This is a direct port of librosa.core.conver.mel_to_hz to work with torchaudio.

    Args:
        mels : Frequencies to convert, in mel scale
        htk : Use htk formula for conversion or not. Defaults to False.

    Returns:
        Frequencies in hertz.
    """
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * (logstep * (mels[log_t] - min_log_mel)).exp()

    return freqs


def mel_frequencies(f_min: int, f_max: int, n_mels: int, htk: bool) -> Tensor:
    """Calculates the frequencies to create the mel scale filterbanks.
    This is a direct port of librosa.filters.mel_frequencies to work with pytorch.

    Args:
        f_min : Minimum frequency
        f_max : Maximum frequency
        n_mels : Number of mels
        htk : Use htk formula for mel scale or not

    Returns:
        Tensor containing the corresponding frequencies.
    """
    m_min = hz_to_mel(f_min, htk=htk)
    m_max = hz_to_mel(f_max, htk=htk)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, htk=htk)

    return f_pts


def create_fb_matrix(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Optional[str] = None,
    htk: bool = True,
) -> Tensor:
    """Create a frequency bin conversion matrix.
    This is a direct modification of torchaudio.functional.create_fb_matrix
    to also create the frequencies using the same formula as librosa

    Args:
        n_freqs : Number of frequencies to highlight/apply
        f_min : Minimum frequency (Hz)
        f_max : Maximum frequency (Hz)
        n_mels : Number of mel filterbanks
        sample_rate : Sample rate of the audio waveform
        norm : If 'slaney', divide the triangular mel weights by the width of the mel band (area normalization).
        htk : Use htk formula for mel scale or not.

    Returns:
        Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * create_fb_matrix(A.size(-1), ...)``.
    """

    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    f_pts = mel_frequencies(f_min, f_max, n_mels, htk=htk)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb
