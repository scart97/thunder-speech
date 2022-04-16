# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for thunder.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import json
import os

import pytest

from hypothesis import settings
from torch.hub import download_url_to_file
from torchaudio.datasets.utils import extract_archive

from thunder.text_processing.preprocess import normalize_text
from thunder.utils import audio_len, get_default_cache_folder, get_files

# Increase deadline on CI, where the machine might be slower
# 3 seconds
settings.register_profile("ci", deadline=3000)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(scope="session")
def sample_data():
    path = get_default_cache_folder()
    out_path = path / "lapsbm-backup-lapsbm-ci"
    if not out_path.exists():
        download_url_to_file(
            "https://github.com/scart97/lapsbm-backup/archive/refs/tags/lapsbm-ci.tar.gz",
            str(path / "lapsbm-backup-lapsbm-ci.tar.gz"),
        )
        extract_archive(path / "lapsbm-backup-lapsbm-ci.tar.gz", path)
    return out_path


@pytest.fixture(scope="session")
def sample_audio():
    path = get_default_cache_folder()
    out_path = path / "f0001_us_f0001_00001.wav"
    if not out_path.exists():
        download_url_to_file(
            "https://github.com/fastaudio/10_Speakers_Sample/raw/76f365de2f4d282ec44450d68f5b88de37b8b7ad/train/f0001_us_f0001_00001.wav",
            str(out_path),
        )
    return out_path


@pytest.fixture(scope="session")
def sample_manifest(sample_data):
    audio_files = get_files(sample_data / "LapsBM-F004", ".wav")

    manifest = sample_data / "test_example_manifest.json"
    with open(manifest, "w", encoding="utf8") as f:
        for fil in audio_files:
            data = {
                "audio_filepath": str(fil.resolve()),
                "duration": audio_len(fil),
                "text": normalize_text(fil.with_suffix(".txt").read_text().strip()),
            }
            json.dump(data, f)
            f.write("\n")
    return manifest
