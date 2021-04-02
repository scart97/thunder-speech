# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for thunder.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import json

import pytest

from torchaudio.datasets.utils import download_url, extract_archive

from thunder.text_processing.preprocess import normalize_text
from thunder.utils import audio_len, get_default_cache_folder, get_files


@pytest.fixture(scope="session")
def sample_data():
    path = get_default_cache_folder()
    download_url(
        "http://www02.smt.ufrj.br/~igor.quintanilha/lapsbm-test.tar.gz",
        download_folder=path,
        resume=True,
    )
    out_path = path / "lapsbm-test"
    extract_archive(path / "lapsbm-test.tar.gz", out_path)
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
