# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 scart97

__all__ = ["audio_len", "get_default_cache_folder", "get_files", "chain_calls"]

import functools
import os
from pathlib import Path
from typing import Callable, List, Union

import torchaudio


def audio_len(item: Union[Path, str]) -> float:
    """Returns the length of the audio file

    Args:
        item : Audio path

    Returns:
        Lenght in seconds of the audio
    """
    metadata = torchaudio.info(item)
    return metadata.num_frames / metadata.sample_rate


def get_default_cache_folder() -> Path:
    """Get the default folder where the cached stuff will be saved.

    Returns:
        Path of the cache folder.
    """
    folder = Path.home() / ".thunder"
    folder.mkdir(exist_ok=True)
    return folder


def get_files(directory: Union[str, Path], extension: str) -> List[Path]:
    """Find all files in directory with extension.

    Args:
        directory : Directory to recursively find the files
        extension : File extension to search for

    Returns:
        List of all the files that match the extension
    """
    files_found = []

    for root, _, files in os.walk(directory, followlinks=True):
        files_found += [Path(root) / f for f in files if f.endswith(extension)]
    return files_found


def chain_calls(*funcs: List[Callable]) -> Callable:
    """Chain multiple functions that take only one argument, producing a new function that is the result
    of calling the individual functions in sequence.

    Example:
    ```python
    f1 = lambda x: 2 * x
    f2 = lambda x: 3 * x
    f3 = lambda x: 4 * x
    g = chain_calls(f1, f2, f3)
    assert g(1) == 24
    ```

    Returns:
        Single chained function
    """

    def call(x, f):
        return f(x)

    def _inner(arg):
        return functools.reduce(call, funcs, arg)

    return _inner
