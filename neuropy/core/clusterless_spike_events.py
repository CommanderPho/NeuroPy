"""Sparse clusterless spike-event container and portable NPZ persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from neuropy.core.datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol


CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION: int = 1


class ClusterlessSpikeEvents(StartStopTimesMixin, TimeSlicableObjectProtocol, DataWriter):
    """Sparse clusterless spike events for portable transfer, one row per detected spike."""

    def __init__(self, spike_times_sec: np.ndarray, electrode_indices: np.ndarray, marks: np.ndarray, sampling_frequency_hz: float = 1000.0, electrode_mode: str = "channel", n_mark_dims: Optional[int] = None, t_start: float = 0.0, t_stop: Optional[float] = None, t_end: Optional[float] = None, source_phy_path: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        super().__init__(metadata=metadata)
        self.spike_times_sec = np.asarray(spike_times_sec).reshape(-1)
        self.electrode_indices = np.asarray(electrode_indices).reshape(-1)
        self.marks = np.asarray(marks)
        self._validate_parallel_arrays()
        if n_mark_dims is None:
            n_mark_dims = int(self.marks.shape[1])
        elif int(n_mark_dims) != int(self.marks.shape[1]):
            raise ValueError(f"n_mark_dims ({n_mark_dims}) must match marks.shape[1] ({self.marks.shape[1]}).")
        self.sampling_frequency_hz = float(sampling_frequency_hz)
        self.electrode_mode = str(electrode_mode)
        self.n_mark_dims = int(n_mark_dims)
        self.t_start = float(t_start)
        if t_stop is None:
            t_stop = t_end
        if t_stop is None:
            t_stop = float(np.max(self.spike_times_sec)) if len(self.spike_times_sec) > 0 else 0.0
        self.t_stop = float(t_stop)
        self.source_phy_path = source_phy_path



    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _validate_parallel_arrays(self) -> None:
        if self.marks.ndim != 2:
            raise ValueError(f"marks must be a 2-D array with shape (n_spikes, n_mark_dims); got shape {self.marks.shape}.")
        n_spikes = len(self.spike_times_sec)
        if len(self.electrode_indices) != n_spikes or self.marks.shape[0] != n_spikes:
            raise ValueError("spike_times_sec, electrode_indices, and marks must have the same number of rows.")


    @property
    def t_end(self) -> float:
        return self.t_stop


    @t_end.setter
    def t_end(self, value: float) -> None:
        self.t_stop = float(value)


    @property
    def time(self) -> np.ndarray:
        return self.spike_times_sec


    @property
    def time_variable_name(self) -> str:
        return "spike_times_sec"


    @property
    def n_spikes(self) -> int:
        return len(self.spike_times_sec)


    @property
    def n_electrodes(self) -> int:
        if self.n_spikes == 0:
            return 0
        return int(np.max(self.electrode_indices)) + 1


    def __len__(self) -> int:
        return self.n_spikes


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n n_spikes: {self.n_spikes}\n n_electrodes: {self.n_electrodes}\n n_mark_dims: {self.n_mark_dims}\n t_start: {self.t_start}\n t_stop: {self.t_stop}"


    def _copy_with_mask(self, inclusion_mask: np.ndarray, t_start: Optional[float] = None, t_stop: Optional[float] = None) -> "ClusterlessSpikeEvents":
        return ClusterlessSpikeEvents(spike_times_sec=self.spike_times_sec[inclusion_mask].copy(), electrode_indices=self.electrode_indices[inclusion_mask].copy(), marks=self.marks[inclusion_mask].copy(),
            sampling_frequency_hz=self.sampling_frequency_hz, electrode_mode=self.electrode_mode, n_mark_dims=self.n_mark_dims,
            t_start=self.t_start if t_start is None else float(t_start), t_stop=self.t_stop if t_stop is None else float(t_stop), source_phy_path=self.source_phy_path, metadata=self.metadata)


    def time_slice(self, t_start=None, t_stop=None) -> "ClusterlessSpikeEvents":
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        inclusion_mask = (self.spike_times_sec >= t_start) & (self.spike_times_sec <= t_stop)
        return self._copy_with_mask(inclusion_mask, t_start=t_start, t_stop=t_stop)


    def get_by_electrode(self, electrode_indices) -> "ClusterlessSpikeEvents":
        inclusion_mask = np.isin(self.electrode_indices, np.atleast_1d(electrode_indices))
        return self._copy_with_mask(inclusion_mask)


    def to_dataframe(self) -> pd.DataFrame:
        data = {"t_seconds": self.spike_times_sec.copy(), "electrode": self.electrode_indices.copy()}
        for mark_idx in range(self.n_mark_dims):
            data[f"mark_{mark_idx}"] = self.marks[:, mark_idx].copy()
        return pd.DataFrame(data)


    def to_dict(self, recurrsively=False) -> dict:
        return {"spike_times_sec": self.spike_times_sec.copy(), "electrode_indices": self.electrode_indices.copy(), "marks": self.marks.copy(),
            "sampling_frequency_hz": self.sampling_frequency_hz, "electrode_mode": self.electrode_mode, "n_mark_dims": self.n_mark_dims,
            "t_start": self.t_start, "t_stop": self.t_stop, "t_end": self.t_end, "source_phy_path": self.source_phy_path, "metadata": self.metadata}


    @classmethod
    def from_dict(cls, d: dict) -> "ClusterlessSpikeEvents":
        t_stop = d.get("t_stop", d.get("t_end", None))
        return cls(spike_times_sec=d["spike_times_sec"], electrode_indices=d["electrode_indices"], marks=d["marks"], sampling_frequency_hz=d.get("sampling_frequency_hz", 1000.0),
            electrode_mode=d.get("electrode_mode", "channel"), n_mark_dims=d.get("n_mark_dims", None), t_start=d.get("t_start", 0.0), t_stop=t_stop,
            source_phy_path=d.get("source_phy_path", None), metadata=d.get("metadata", None))


    def to_npz(self, filepath: Union[str, Path]) -> Path:
        return save_clusterless_spike_events(filepath, self)


    @classmethod
    def from_npz(cls, filepath: Union[str, Path]) -> "ClusterlessSpikeEvents":
        return load_clusterless_spike_events(filepath)


    @classmethod
    def from_file(cls, f: Union[str, Path]) -> "ClusterlessSpikeEvents":
        return cls.from_npz(f)


    def save(self, status_print=True):
        if self.filename is None:
            raise ValueError("filename must be set before saving ClusterlessSpikeEvents.")
        saved_path = self.to_npz(self.filename)
        if status_print:
            print(f"{Path(saved_path).name} saved")
        return saved_path


def default_clusterless_spike_events_path(session_basedir: Union[str, Path], session_name: str) -> Path:
    return Path(session_basedir) / f"{session_name}.clusterless_spikes.npz"


def save_clusterless_spike_events(filepath: Union[str, Path], events: ClusterlessSpikeEvents) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, version=np.array([CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION], dtype=np.int32), spike_times_sec=np.asarray(events.spike_times_sec, dtype=np.float32), electrode_indices=np.asarray(events.electrode_indices, dtype=np.int16), marks=np.asarray(events.marks, dtype=np.float32), sampling_frequency_hz=np.array([events.sampling_frequency_hz], dtype=np.float64), electrode_mode=np.array([events.electrode_mode]), n_mark_dims=np.array([events.n_mark_dims], dtype=np.int32), t_start=np.array([events.t_start], dtype=np.float64), t_end=np.array([events.t_end], dtype=np.float64), source_phy_path=np.array([events.source_phy_path or ""], dtype=object))
    return filepath


def load_clusterless_spike_events(filepath: Union[str, Path]) -> ClusterlessSpikeEvents:
    with np.load(Path(filepath), allow_pickle=True) as data:
        file_version = int(data["version"].item()) if "version" in data else CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION
        if file_version != CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION:
            raise ValueError(f"Unsupported clusterless spike events file version {file_version}; expected {CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION}.")
        source_phy_path = str(data["source_phy_path"].item()) if "source_phy_path" in data else None
        if source_phy_path == "":
            source_phy_path = None
        t_stop = float(data["t_stop"].item()) if "t_stop" in data else float(data["t_end"].item())
        return ClusterlessSpikeEvents(spike_times_sec=np.asarray(data["spike_times_sec"], dtype=np.float32), electrode_indices=np.asarray(data["electrode_indices"], dtype=np.int16), marks=np.asarray(data["marks"], dtype=np.float32), sampling_frequency_hz=float(data["sampling_frequency_hz"].item()), electrode_mode=str(data["electrode_mode"].item()), n_mark_dims=int(data["n_mark_dims"].item()), t_start=float(data["t_start"].item()), t_stop=t_stop, source_phy_path=source_phy_path)
