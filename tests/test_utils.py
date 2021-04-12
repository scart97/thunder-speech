# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

from pathlib import Path

from thunder.utils import audio_len, chain_calls, get_default_cache_folder, get_files


def test_audio_len(sample_data):
    audio_files = get_files(sample_data, ".wav")
    audio_length = audio_len(audio_files[0])
    assert audio_length > 0.0
    assert isinstance(audio_length, float)


def test_get_default_cache_folder():
    path = get_default_cache_folder()
    assert path.exists()


def test_get_files_exist():
    path = Path("tests/nemo_config_samples")
    manifest_files = get_files(path, ".yaml")

    assert len(manifest_files) == 3
    assert isinstance(manifest_files[0], Path)
    assert manifest_files[0].exists()


def test_get_files_work_with_string_input():
    path = "tests/nemo_config_samples"
    manifest_files = get_files(path, ".yaml")

    assert len(manifest_files) == 3
    assert isinstance(manifest_files[0], Path)
    assert manifest_files[0].exists()


def test_get_files_dont_exist():
    path = Path("tests/nemo_config_samples")
    manifest_files = get_files(path, ".mp3")

    assert len(manifest_files) == 0


def test_chain_calls():
    f1 = lambda x: 2 * x  # noqa: E731
    f2 = lambda x: 3 * x  # noqa: E731
    f3 = lambda x: 4 * x  # noqa: E731
    g = chain_calls(f1, f2, f3)
    assert g(1) == 24


def test_chain_calls_dont_modify_input_alone():
    g = chain_calls()
    assert g([1, 2, 3]) == [1, 2, 3]
    assert g(1) == 1
    assert g("abc") == "abc"
    assert g((1, 2, 3)) == (1, 2, 3)


def test_chain_calls_respect_order():
    f1 = lambda x: x + "a"  # noqa: E731
    f2 = lambda x: x + "b"  # noqa: E731
    f3 = lambda x: x + "c"  # noqa: E731
    g = chain_calls(f1, f2, f3)
    assert g("") == "abc"
    g2 = chain_calls(f2, f1, f3)
    assert g2("") == "bac"
