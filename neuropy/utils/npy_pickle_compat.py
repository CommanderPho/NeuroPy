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


def _is_numpy_core_pickle_error(exc: BaseException) -> bool:
    return isinstance(exc, ModuleNotFoundError) and 'numpy._core' in str(exc)


_NUMPY2_CORE_SUBMODULES = (
    'multiarray', 'umath', '_multiarray_umath', 'numeric', '_internal', '_dtype_ctypes', 'arrayprint',
    'records', 'memmap', 'defchararray', 'fromnumeric', 'shape_base', 'stride_tricks', 'einsumfunc',
    'overrides', 'scalar', 'cast', '_type_aliases', 'function_base', 'getlimits', 'machar',
)


def _np_load_pickled_item(f: Union[str, Path]) -> Any:
    return np.load(f, allow_pickle=True).item()


@contextlib.contextmanager
def _numpy2_pickle_shim_context():
    """Map ``numpy._core`` to ``numpy.core`` so NumPy-2 pickles load on NumPy-1."""
    import sys
    if int(np.__version__.split('.')[0]) >= 2:
        yield
        return
    import numpy.core as numpy_core
    aliases: dict[str, object] = {'numpy._core': numpy_core}
    for sub in _NUMPY2_CORE_SUBMODULES:
        try:
            aliases[f'numpy._core.{sub}'] = __import__(f'numpy.core.{sub}', fromlist=[sub])
        except ImportError:
            pass
    saved = {k: sys.modules.get(k) for k in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = prev


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


def _load_with_fallbacks(f: Path) -> Any:
    try:
        return _np_load_pickled_item(f)
    except NotImplementedError:
        return _load_with_posix_path_hack(f)


def load_npy_pickled_item(f: Union[str, Path]) -> Any:
    """Load the Python object stored in a pickled 0-d object .npy file (e.g. session ``*.position.npy``).

    Uses standard ``np.load`` first, then NumPy-2 / legacy-pandas compat shims when needed.
    """
    f = Path(f) if not isinstance(f, Path) else f
    try:
        return _load_with_fallbacks(f)
    except Exception as exc:
        if not (_is_numpy_core_pickle_error(exc) or _is_pandas_pickle_error(exc)):
            raise
        with _numpy2_pickle_shim_context(), _pandas_compat_pickle_load_context():
            return _load_with_fallbacks(f)
