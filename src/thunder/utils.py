"""
Utility functions
"""

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = [
    "audio_len",
    "get_default_cache_folder",
    "get_files",
    "chain_calls",
    "BaseCheckpoint",
    "download_checkpoint",
    "SchedulerBuilderType",
]

import functools
import os
from enum import Enum
from pathlib import Path
from typing import Callable, List, Type, Union

import torchaudio
import wget
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


def audio_len(item: Union[Path, str]) -> float:
    """Returns the length of the audio file

    Args:
        item: Audio path

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
        directory: Directory to recursively find the files
        extension: File extension to search for

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


class BaseCheckpoint(str, Enum):
    """Base class that represents a pretrained model checkpoint."""

    @classmethod
    def from_string(cls, name: str) -> "BaseCheckpoint":
        """Creates enum value from string. Helper to use with argparse/hydra

        Args:
            name: Name of the checkpoint

        Raises:
            ValueError: Name provided is not a valid checkpoint

        Returns:
            Enum value corresponding to the name
        """
        try:
            return cls[name]
        except KeyError as option_does_not_exist:
            raise ValueError(
                "Name provided is not a valid checkpoint"
            ) from option_does_not_exist


def download_checkpoint(name: BaseCheckpoint, checkpoint_folder: str = None) -> Path:
    """Download checkpoint by identifier.

    Args:
        name: Model identifier. Check checkpoint_archives.keys()
        checkpoint_folder: Folder where the checkpoint will be saved to.

    Returns:
        Path to the saved checkpoint file.
    """
    if checkpoint_folder is None:
        checkpoint_folder = get_default_cache_folder()

    url = name.value
    filename = url.split("/")[-1]
    checkpoint_path = Path(checkpoint_folder) / filename
    if not checkpoint_path.exists():
        wget.download(url, out=str(checkpoint_path))

    return checkpoint_path


# Reference to learning rate scheduler class
_SchedulerClassType = Union[
    Type[_LRScheduler],
    Type[ReduceLROnPlateau],
]
# Arbitrary function that returns a learning rate scheduler
_SchedulerFuncType = Callable[..., Union[_LRScheduler, ReduceLROnPlateau]]
# Two valid options to build a learning rate scheduler
SchedulerBuilderType = Union[_SchedulerClassType, _SchedulerFuncType]

# Reference to optimizer class
_OptimizerClassType = Type[Optimizer]
# Arbitrary function that returns an optimizer
_OptimizerFuncType = Callable[..., Optimizer]
# Two valid options to build an optimizer
OptimizerBuilderType = Union[_OptimizerClassType, _OptimizerFuncType]

# Reference to nn.Module
_ModuleClassType = Type[nn.Module]
# Arbitrary function that returns a nn.Module
_ModuleFuncType = Callable[..., nn.Module]
# Two valid options to build a nn.Module
ModuleBuilderType = Union[_ModuleClassType, _ModuleFuncType]
