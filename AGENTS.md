# AGENTS.md

## Cursor Cloud specific instructions

NeuroPy is a **pure Python library** for electrophysiology / hippocampal data analysis. There is no server, GUI, or long-running service to start — you "run" it by importing `neuropy` and calling into it (e.g. the README raster-plot example), running `pytest`, or opening the notebooks in `examples/`.

### Environment
- Dependencies are pinned for **Python 3.9–3.11** (`requirements.txt` mirrors the CI install). The system default `python3` is 3.12, which cannot build the pinned deps (e.g. `numpy==1.23.5`, `numba==0.56.4`), so a **Python 3.10 virtualenv at `.venv`** is used (managed by `uv`). Always run project commands with `.venv/bin/python` (or `source .venv/bin/activate`).
- The startup update script installs `requirements.txt` and then applies two dependency fixups that are **not** in `requirements.txt` but are required for the library to import:
  - `nptyping` — imported across `neuropy` but missing from `requirements.txt`.
  - `matplotlib==3.8.4` (pin overrides `requirements.txt`'s `3.9.3`) — the code imports `matplotlib.collections.BrokenBarHCollection`, which was **removed in matplotlib 3.9**. Keep matplotlib `<3.9`. If you ever re-run `uv pip install -r requirements.txt` manually, it will pull matplotlib 3.9.3 back in — re-apply the `matplotlib==3.8.4` pin afterward.

### Lint (matches CI `.github/workflows/python-package.yml`)
- `.venv/bin/flake8 . --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv` — critical errors. This currently **exits non-zero** due to pre-existing violations in the repo (e.g. `E999 TabError` in `neuropy/utils/signal_process.py`, several `F821` undefined names). These are pre-existing code issues, not environment problems.
- `.venv/bin/flake8 . --exit-zero --max-complexity=10 --max-line-length=127 --exclude=.venv` — style pass (never fails).

### Tests
- Run with `.venv/bin/python -m pytest`.
- Data-backed tests (`tests/test_spikes.py`, `test_placefields.py`, `test_epochs.py`, `test_position.py`) require fixtures pulled via **DVC** from a Google Drive remote (`dvc pull`, needs credentials) and are expected to fail/skip without that data. `tests/test_position.py` also fails to collect because it imports `unittesting_extensions` (needs `tests/` on `sys.path`).
- Self-contained modules that run without any external data: `tests/test_utils.py`, `tests/test_contexts.py`, `tests/test_indexing_helpers.py`, `tests/test_subsettable_dict.py`, `tests/test_nwb_data_session_format.py`. Run them directly, e.g.:
  `.venv/bin/python -m pytest tests/test_utils.py tests/test_contexts.py tests/test_indexing_helpers.py tests/test_subsettable_dict.py tests/test_nwb_data_session_format.py`
  One pre-existing failure here, `test_contexts.py::TestIdentifyingContext::test_query_case_insensitivity`, is a code logic bug (not an environment issue).

### Smoke test / "run the app"
The quickest end-to-end check that core functionality works (README example):
```python
import matplotlib; matplotlib.use("Agg")
import numpy as np
from neuropy.core import Neurons
from neuropy import plotting
spiketrains = np.array([np.sort(np.random.rand(_)*1000.0) for _ in range(100,200)], dtype=object)
plotting.plot_raster(Neurons(spiketrains, t_stop=1000), color="jet")
```
