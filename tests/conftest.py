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
from torchaudio.datasets.utils import download_url, extract_archive

from thunder.text_processing.preprocess import normalize_text
from thunder.utils import audio_len, get_default_cache_folder, get_files

# Increase deadline on CI, where the machine might be slower
# 3 seconds
settings.register_profile("ci", deadline=3000)
settings.load_profile(os.getenv(u"HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(scope="session")
def sample_data():
    path = get_default_cache_folder()
    download_url(
        "https://github.com/scart97/lapsbm-backup/archive/refs/tags/lapsbm-ci.tar.gz",
        download_folder=path,
        resume=True,
    )
    extract_archive(path / "lapsbm-backup-lapsbm-ci.tar.gz", path)
    return path / "lapsbm-backup-lapsbm-ci"


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
