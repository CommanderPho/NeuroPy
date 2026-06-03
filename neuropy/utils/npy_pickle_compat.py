"""Load pickled object arrays from .npy files written under older pandas/numpy versions."""
from __future__ import annotations

import contextlib
import pathlib
import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np


def _is_pandas_pickle_error(exc: BaseException) -> bool:
    return isinstance(exc, (ModuleNotFoundError, AttributeError)) and 'pandas' in str(exc).lower()


def _np_load_pickled_item(f: Union[str, Path]) -> Any:
    return np.load(f, allow_pickle=True).item()


@contextlib.contextmanager
def _pandas_compat_pickle_load_context():
    from pandas.compat.pickle_compat import Unpickler as PandasUnpickler
    orig_load = pickle.load

    def compat_load(file, *args, **kwargs):
        return PandasUnpickler(file, *args, **kwargs).load()

    pickle.load = compat_load
    try:
        yield
    finally:
        pickle.load = orig_load


def _load_with_posix_path_hack(f: Union[str, Path]) -> Any:
    print(f"Issue with pickled POSIX_PATH on windows for path {f}, falling back to non-pickled version...")
    orig_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.PurePosixPath
    try:
        return _np_load_pickled_item(f)
    finally:
        pathlib.PosixPath = orig_posix_path


def load_npy_pickled_item(f: Union[str, Path]) -> Any:
    """Load the Python object stored in a pickled 0-d object .npy file (e.g. session ``*.position.npy``).

    Uses standard ``np.load`` first, then falls back to pandas' compat unpickler for legacy pickles.
    """
    f = Path(f) if not isinstance(f, Path) else f
    try:
        return _np_load_pickled_item(f)
    except NotImplementedError:
        return _load_with_posix_path_hack(f)
    except Exception as exc:
        if not _is_pandas_pickle_error(exc):
            raise
        with _pandas_compat_pickle_load_context():
            try:
                return _np_load_pickled_item(f)
            except NotImplementedError:
                return _load_with_posix_path_hack(f)
