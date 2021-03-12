import functools
import os
from pathlib import Path
from typing import Callable, List, Union


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
