"""Sparse clusterless spike-event container and portable NPZ persistence."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from neuropy.core.datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol


CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION: int = 1
_PHY_CLUSTERLESS_REQUIRED_FILES = ("params.py", "spike_times.npy", "spike_templates.npy", "pc_features.npy", "pc_feature_ind.npy")


def _read_phy_params(phy_path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    with (phy_path / "params.py").open("r", encoding="utf-8") as params_file:
        for line in params_file:
            line_values = line.replace("\n", "").replace('r"', '"').replace('"', "").split("=")
            if len(line_values) >= 2:
                params[line_values[0].strip()] = line_values[1].strip()
    if "sample_rate" not in params:
        raise ValueError(f"params.py in {phy_path} is missing sample_rate.")
    return params


def _parse_phy_param_float(params: dict[str, str], key: str) -> Optional[float]:
    if key not in params:
        return None
    return float(params[key])


def _resolve_phy_dat_path(phy_path: Path, params: dict[str, str]) -> Optional[Path]:
    dat_path = params.get("dat_path")
    if dat_path is None or dat_path in {"", "no_path.bin"}:
        return None
    dat_file = Path(dat_path)
    if not dat_file.is_absolute():
        for candidate_path in (phy_path / dat_file, phy_path.parent / dat_file, phy_path.parent.parent / dat_file):
            if candidate_path.is_file():
                return candidate_path.resolve()
        return None
    return dat_file if dat_file.is_file() else None


def _infer_recording_duration_from_dat(params: dict[str, str], phy_path: Path, sample_rate_hz: float) -> Optional[float]:
    dat_file = _resolve_phy_dat_path(phy_path, params)
    if dat_file is None:
        return None
    n_channels = _parse_phy_param_float(params, "n_channels_dat")
    if n_channels is None or n_channels <= 0:
        return None
    dtype_str = params.get("dtype", "int16")
    byte_offset = int(_parse_phy_param_float(params, "offset") or 0.0)
    n_bytes = dat_file.stat().st_size - byte_offset
    if n_bytes <= 0:
        return None
    n_samples = n_bytes // (int(n_channels) * np.dtype(dtype_str).itemsize)
    return float(n_samples) / sample_rate_hz


def _infer_phy_session_times(phy_path: Path, params: dict[str, str], spike_times: np.ndarray, sample_rate_hz: float) -> Tuple[float, float]:
    inferred_t_start = 0.0
    for key in ("t_start", "tmin", "start_time"):
        parsed_value = _parse_phy_param_float(params, key)
        if parsed_value is not None:
            inferred_t_start = parsed_value
            break
    inferred_t_end: Optional[float] = None
    for key in ("t_end", "t_stop", "tmax", "duration"):
        parsed_value = _parse_phy_param_float(params, key)
        if parsed_value is not None:
            inferred_t_end = parsed_value
            break
    if inferred_t_end is None:
        n_samples_dat = _parse_phy_param_float(params, "n_samples_dat")
        if n_samples_dat is not None:
            inferred_t_end = n_samples_dat / sample_rate_hz
    if inferred_t_end is None:
        inferred_t_end = _infer_recording_duration_from_dat(params, phy_path, sample_rate_hz)
    if inferred_t_end is None:
        if len(spike_times) == 0:
            raise ValueError(f"No spikes in Phy folder {phy_path}; cannot infer session t_end.")
        inferred_t_end = float(np.max(spike_times)) / sample_rate_hz
        warnings.warn(f"Could not infer recording duration for {phy_path} from params.py or dat file; using last spike time ({inferred_t_end:.6f} s) as t_end.", stacklevel=2)
    if inferred_t_end <= inferred_t_start:
        raise ValueError(f"Inferred invalid session time range t_start={inferred_t_start} t_end={inferred_t_end} for {phy_path}.")
    return inferred_t_start, inferred_t_end


def _resolve_channel_shanks(phy_path: Path) -> Optional[np.ndarray]:
    candidate_paths = [phy_path / "channel_shanks.npy", phy_path.parent / "sorter_output" / "channel_shanks.npy"]
    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return np.load(candidate_path)
    return None


def _get_epoch_spike_slice(spike_times: np.ndarray, sample_rate_hz: float, t_start: float, t_end: float) -> slice:
    sample_start = int(np.floor(float(t_start) * sample_rate_hz))
    sample_end = int(np.ceil(float(t_end) * sample_rate_hz))
    spike_start = int(np.searchsorted(spike_times, sample_start, side="left"))
    spike_end = int(np.searchsorted(spike_times, sample_end, side="right"))
    return slice(spike_start, spike_end)


def _build_channel_inverse_map(channel_map: np.ndarray) -> np.ndarray:
    channel_map = np.asarray(channel_map, dtype=int)
    inverse_map = np.full(int(channel_map.max()) + 1, -1, dtype=int)
    for recording_idx, probe_channel in enumerate(channel_map):
        inverse_map[int(probe_channel)] = int(recording_idx)
    return inverse_map


def _extract_peak_channel_marks(pc_features: np.ndarray, pc_feature_ind: np.ndarray, spike_templates: np.ndarray, spike_indices: np.ndarray, n_mark_dims: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    n_spikes = len(spike_indices)
    n_slots = int(pc_features.shape[2])
    channels = np.empty(n_spikes, dtype=int)
    marks = np.empty((n_spikes, n_mark_dims), dtype=float)
    for spike_offset, spike_index in enumerate(spike_indices):
        template_index = int(spike_templates[spike_index])
        template_channels = pc_feature_ind[template_index]
        spike_pcs = pc_features[spike_index]
        slot_norms = np.array([np.linalg.norm(spike_pcs[:, slot_idx]) if template_channels[slot_idx] >= 0 else -1.0 for slot_idx in range(n_slots)], dtype=float)
        peak_slot = int(np.argmax(slot_norms))
        channels[spike_offset] = int(template_channels[peak_slot])
        marks[spike_offset, :] = spike_pcs[:n_mark_dims, peak_slot]
    return channels, marks


def _map_channels_to_electrodes(channels: np.ndarray, electrode_mode: str, channel_map: Optional[np.ndarray], channel_shanks: Optional[np.ndarray]) -> np.ndarray:
    channels = np.asarray(channels, dtype=int)
    if electrode_mode == "shank":
        if channel_shanks is None:
            raise ValueError("channel_shanks is required for electrode_mode='shank'.")
        inverse_map = _build_channel_inverse_map(channel_map) if channel_map is not None else None
        electrode_indices = np.empty(len(channels), dtype=int)
        for spike_idx, probe_channel in enumerate(channels):
            recording_idx = int(inverse_map[probe_channel]) if inverse_map is not None and probe_channel < len(inverse_map) and inverse_map[probe_channel] >= 0 else int(probe_channel)
            electrode_indices[spike_idx] = int(channel_shanks[recording_idx])
        return electrode_indices
    if electrode_mode != "channel":
        raise ValueError(f"electrode_mode must be 'shank' or 'channel'; got {electrode_mode!r}")
    if channel_map is not None:
        inverse_map = _build_channel_inverse_map(channel_map)
        return np.array([int(inverse_map[probe_channel]) if probe_channel < len(inverse_map) and inverse_map[probe_channel] >= 0 else int(probe_channel) for probe_channel in channels], dtype=int)
    return channels.astype(int, copy=False)


def _resolve_effective_electrode_mode(phy_path: Path, electrode_mode: str, channel_shanks: Optional[np.ndarray]) -> str:
    if electrode_mode == "shank" and (channel_shanks is None or len(np.unique(channel_shanks)) <= 1):
        warnings.warn(f"channel_shanks missing or degenerate in {phy_path}; falling back to electrode_mode='channel'.", stacklevel=2)
        return "channel"
    return electrode_mode


class ClusterlessSpikeEvents(StartStopTimesMixin, TimeSlicableObjectProtocol, DataWriter):
    """Sparse clusterless spike events for portable transfer, one row per detected spike."""

    def __init__(self, spike_times_sec: np.ndarray, electrode_indices: np.ndarray, marks: np.ndarray, sampling_frequency_hz: float = 1000.0, electrode_mode: str = "channel", n_mark_dims: Optional[int] = None,
                t_start: float = 0.0, t_stop: Optional[float] = None, t_end: Optional[float] = None,
                source_phy_path: Optional[str] = None, metadata: Optional[dict] = None) -> None:
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


    @classmethod
    def from_phy_folder(cls, phy_path: Union[str, Path], t_start: Optional[float] = None, t_end: Optional[float] = None, electrode_mode: str = "channel", n_mark_dims: int = 4, chunk_size: int = 100_000, sampling_frequency_hz: float = 1000.0) -> "ClusterlessSpikeEvents":
        """Extract sparse clusterless spike events from a Phy/Kilosort folder without allocating dense multiunits.

        Reads all detected spikes (ignores spike_clusters). When ``t_start`` and/or ``t_end`` are omitted,
        the session span is inferred from ``params.py``, dat file size, or the last spike time.
        """
        phy_path = Path(phy_path)
        missing_files = [a_file for a_file in _PHY_CLUSTERLESS_REQUIRED_FILES if not (phy_path / a_file).is_file()]
        if missing_files:
            raise FileNotFoundError(f"Phy folder {phy_path} is missing required files: {missing_files}")
        phy_params = _read_phy_params(phy_path)
        sample_rate_hz = float(phy_params["sample_rate"])
        spike_times = np.asarray(np.load(phy_path / "spike_times.npy", mmap_mode="r")).reshape(-1)
        spike_templates = np.asarray(np.load(phy_path / "spike_templates.npy", mmap_mode="r")).reshape(-1)
        pc_features = np.load(phy_path / "pc_features.npy", mmap_mode="r")
        pc_feature_ind = np.load(phy_path / "pc_feature_ind.npy")
        channel_map = np.load(phy_path / "channel_map.npy") if (phy_path / "channel_map.npy").is_file() else None
        channel_shanks = _resolve_channel_shanks(phy_path)
        if t_start is None or t_end is None:
            inferred_t_start, inferred_t_end = _infer_phy_session_times(phy_path, phy_params, spike_times, sample_rate_hz)
            if t_start is None:
                t_start = inferred_t_start
            if t_end is None:
                t_end = inferred_t_end
        epoch_slice = _get_epoch_spike_slice(spike_times, sample_rate_hz, t_start, t_end)
        if epoch_slice.start >= epoch_slice.stop:
            raise ValueError(f"No spikes found in Phy folder for epoch t=[{t_start}, {t_end}] seconds.")
        effective_electrode_mode = _resolve_effective_electrode_mode(phy_path, electrode_mode, channel_shanks)
        spike_times_chunks: list[np.ndarray] = []
        electrode_chunks: list[np.ndarray] = []
        marks_chunks: list[np.ndarray] = []
        for chunk_start in range(epoch_slice.start, epoch_slice.stop, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, epoch_slice.stop)
            spike_indices = np.arange(chunk_start, chunk_stop, dtype=int)
            channels, marks = _extract_peak_channel_marks(pc_features, pc_feature_ind, spike_templates, spike_indices, n_mark_dims=n_mark_dims)
            electrode_indices = _map_channels_to_electrodes(channels, effective_electrode_mode, channel_map, channel_shanks)
            spike_times_sec = (np.asarray(spike_times[spike_indices], dtype=np.float64) / sample_rate_hz).astype(np.float32)
            spike_times_chunks.append(spike_times_sec)
            electrode_chunks.append(electrode_indices.astype(np.int16, copy=False))
            marks_chunks.append(np.asarray(marks, dtype=np.float32))
        return cls(spike_times_sec=np.concatenate(spike_times_chunks), electrode_indices=np.concatenate(electrode_chunks), marks=np.concatenate(marks_chunks), sampling_frequency_hz=float(sampling_frequency_hz), electrode_mode=effective_electrode_mode, n_mark_dims=int(n_mark_dims), t_start=float(t_start), t_stop=float(t_end), source_phy_path=str(phy_path))


    def save(self, status_print=True):
        if self.filename is None:
            raise ValueError("filename must be set before saving ClusterlessSpikeEvents.")
        saved_path = self.to_npz(self.filename)
        if status_print:
            print(f"{Path(saved_path).name} saved")
        return saved_path



# ==================================================================================================================================================================================================================================================================================== #
# Top-level helper functions                                                                                                                                                                                                                                                           #
# ==================================================================================================================================================================================================================================================================================== #
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
        return ClusterlessSpikeEvents(spike_times_sec=np.asarray(data["spike_times_sec"], dtype=np.float32), electrode_indices=np.asarray(data["electrode_indices"], dtype=np.int16), marks=np.asarray(data["marks"], dtype=np.float32),
            sampling_frequency_hz=float(data["sampling_frequency_hz"].item()), electrode_mode=str(data["electrode_mode"].item()), n_mark_dims=int(data["n_mark_dims"].item()),
            t_start=float(data["t_start"].item()), t_stop=t_stop, source_phy_path=source_phy_path,
            # filename=filepath,
            )
