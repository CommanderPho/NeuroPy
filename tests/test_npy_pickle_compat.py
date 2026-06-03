import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

tests_folder = Path(__file__).resolve().parent
root_project_folder = tests_folder.parent
if str(root_project_folder) not in sys.path:
    sys.path.insert(0, str(root_project_folder))

from neuropy.utils.npy_pickle_compat import load_npy_pickled_item


def test_load_npy_pickled_item_roundtrip(tmp_path):
    data = {'df': pd.DataFrame({'s': [3.3, 4.4]}, index=pd.Index([2, 4])), 'metadata': None}
    f = tmp_path / 'test.position.npy'
    np.save(f, data)
    loaded = load_npy_pickled_item(f)
    assert isinstance(loaded, dict)
    pd.testing.assert_frame_equal(loaded['df'], data['df'])


@pytest.mark.parametrize('legacy_path', [
    Path(r'W:\Data\Bapun\RatS\Day1Openfield\RatS-Day1Openfield.position.npy'),
])
def test_load_npy_pickled_item_real_legacy_pandas_session_file(legacy_path):
    """Load a pandas-1.x session .npy that fails under plain np.load on pandas 2."""
    if not legacy_path.is_file():
        pytest.skip(f'legacy fixture not available: {legacy_path}')
    with pytest.raises(ModuleNotFoundError, match='pandas.core.indexes.numeric'):
        np.load(legacy_path, allow_pickle=True).item()
    loaded = load_npy_pickled_item(legacy_path)
    assert isinstance(loaded, dict)
    assert 'df' in loaded
    assert isinstance(loaded['df'], pd.DataFrame)
    assert len(loaded['df']) > 0


@pytest.mark.parametrize('neurons_path', [
    Path(r'W:\Data\Bapun\RatS\Day1Openfield\RatS-Day1Openfield.neurons.npy'),
])
def test_load_npy_pickled_item_numpy2_neurons_on_numpy1(neurons_path):
    """Load a NumPy-2-pickled neurons .npy that fails under plain np.load on NumPy 1."""
    if not neurons_path.is_file():
        pytest.skip(f'fixture not available: {neurons_path}')
    if int(np.__version__.split('.')[0]) >= 2:
        pytest.skip('numpy._core shim only applies when running on NumPy 1.x')
    with pytest.raises(ModuleNotFoundError, match='numpy._core'):
        np.load(neurons_path, allow_pickle=True).item()
    loaded = load_npy_pickled_item(neurons_path)
    assert isinstance(loaded, dict)
    assert 'spiketrains' in loaded
