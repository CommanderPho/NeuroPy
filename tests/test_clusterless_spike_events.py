from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neuropy.core.clusterless_spike_events import ClusterlessSpikeEvents, CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION, load_clusterless_spike_events, save_clusterless_spike_events


def _build_events() -> ClusterlessSpikeEvents:
    return ClusterlessSpikeEvents(spike_times_sec=np.array([1.0, 1.25, 1.5, 2.0], dtype=np.float32), electrode_indices=np.array([0, 1, 1, 2], dtype=np.int16),
        marks=np.array([[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3], [3.0, 3.1, 3.2, 3.3], [4.0, 4.1, 4.2, 4.3]], dtype=np.float32),
        sampling_frequency_hz=1000.0, electrode_mode="channel", t_start=1.0, t_stop=2.0, source_phy_path="synthetic/phy")


def test_constructor_exposes_neuropy_style_properties():
    events = _build_events()

    assert len(events) == 4
    assert events.n_spikes == 4
    assert events.n_electrodes == 3
    assert events.n_mark_dims == 4
    assert events.time_variable_name == "spike_times_sec"
    assert events.t_start == 1.0
    assert events.t_stop == 2.0
    assert events.t_end == 2.0
    np.testing.assert_allclose(events.time, events.spike_times_sec)


def test_constructor_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="same number of rows"):
        ClusterlessSpikeEvents(spike_times_sec=np.array([1.0, 2.0]), electrode_indices=np.array([0]), marks=np.ones((2, 4)))

    with pytest.raises(ValueError, match="marks must be a 2-D array"):
        ClusterlessSpikeEvents(spike_times_sec=np.array([1.0]), electrode_indices=np.array([0]), marks=np.ones(4))

    with pytest.raises(ValueError, match="n_mark_dims"):
        ClusterlessSpikeEvents(spike_times_sec=np.array([1.0]), electrode_indices=np.array([0]), marks=np.ones((1, 4)), n_mark_dims=3)


def test_time_slice_masks_parallel_arrays_and_updates_bounds():
    events = _build_events()

    sliced = events.time_slice(t_start=1.2, t_stop=1.6)

    np.testing.assert_allclose(sliced.spike_times_sec, np.array([1.25, 1.5], dtype=np.float32))
    np.testing.assert_array_equal(sliced.electrode_indices, np.array([1, 1], dtype=np.int16))
    np.testing.assert_allclose(sliced.marks, events.marks[1:3])
    assert sliced.t_start == 1.2
    assert sliced.t_stop == 1.6
    assert sliced.source_phy_path == events.source_phy_path


def test_get_by_electrode_filters_events_without_unit_semantics():
    events = _build_events()

    filtered = events.get_by_electrode([1, 2])

    np.testing.assert_allclose(filtered.spike_times_sec, np.array([1.25, 1.5, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(filtered.electrode_indices, np.array([1, 1, 2], dtype=np.int16))
    np.testing.assert_allclose(filtered.marks, events.marks[[1, 2, 3]])


def test_to_dataframe_has_event_and_mark_columns():
    events = _build_events()

    df = events.to_dataframe()

    assert df.columns.to_list() == ["t_seconds", "electrode", "mark_0", "mark_1", "mark_2", "mark_3"]
    assert df["electrode"].to_list() == [0, 1, 1, 2]
    np.testing.assert_allclose(df["mark_0"].to_numpy(), np.array([1.0, 2.0, 3.0, 4.0]))


def test_dict_roundtrip_preserves_arrays_and_t_end_alias():
    events = _build_events()

    roundtrip = ClusterlessSpikeEvents.from_dict(events.to_dict())

    np.testing.assert_allclose(roundtrip.spike_times_sec, events.spike_times_sec)
    np.testing.assert_array_equal(roundtrip.electrode_indices, events.electrode_indices)
    np.testing.assert_allclose(roundtrip.marks, events.marks)
    assert roundtrip.t_stop == events.t_stop
    assert roundtrip.t_end == events.t_end
    assert roundtrip.source_phy_path == events.source_phy_path


def test_npz_roundtrip_keeps_existing_schema(tmp_path: Path):
    events = _build_events()
    events_path = tmp_path / "test.clusterless_spikes.npz"

    saved_path = save_clusterless_spike_events(events_path, events)
    loaded = load_clusterless_spike_events(saved_path)

    with np.load(saved_path, allow_pickle=True) as data:
        assert "t_end" in data.files
        assert int(data["version"].item()) == CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION

    np.testing.assert_allclose(loaded.spike_times_sec, events.spike_times_sec)
    np.testing.assert_array_equal(loaded.electrode_indices, events.electrode_indices)
    np.testing.assert_allclose(loaded.marks, events.marks)
    assert loaded.t_stop == events.t_stop
    assert loaded.t_end == events.t_end
    assert loaded.source_phy_path == events.source_phy_path


def test_load_rejects_unsupported_npz_version(tmp_path: Path):
    events_path = tmp_path / "bad_version.clusterless_spikes.npz"
    np.savez_compressed(events_path, version=np.array([CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION + 1], dtype=np.int32), spike_times_sec=np.array([], dtype=np.float32), electrode_indices=np.array([], dtype=np.int16), marks=np.empty((0, 4), dtype=np.float32), sampling_frequency_hz=np.array([1000.0]), electrode_mode=np.array(["channel"]), n_mark_dims=np.array([4], dtype=np.int32), t_start=np.array([0.0]), t_end=np.array([0.0]), source_phy_path=np.array([""], dtype=object))

    with pytest.raises(ValueError, match="Unsupported clusterless spike events file version"):
        load_clusterless_spike_events(events_path)


def test_core_init_exports_clusterless_symbols():
    from neuropy.core import ClusterlessSpikeEvents as ExportedClusterlessSpikeEvents
    from neuropy.core import load_clusterless_spike_events as exported_loader

    assert ExportedClusterlessSpikeEvents is ClusterlessSpikeEvents
    assert exported_loader is load_clusterless_spike_events


def test_bapun_loader_attaches_saved_clusterless_spike_events(tmp_path: Path):
    from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass

    events = _build_events()
    session_name = "Synthetic"
    events_path = tmp_path / f"{session_name}.clusterless_spikes.npz"
    save_clusterless_spike_events(events_path, events)
    session = SimpleNamespace(filePrefix=tmp_path / session_name, config=SimpleNamespace(session_name=session_name))

    loaded_session = getattr(BapunDataSessionFormatRegisteredClass, "_try_load_clusterless_spike_events_file")(session)

    assert loaded_session is session
    assert isinstance(session.clusterless_spike_events, ClusterlessSpikeEvents)
    np.testing.assert_allclose(session.clusterless_spike_events.spike_times_sec, events.spike_times_sec)
    np.testing.assert_array_equal(session.clusterless_spike_events.electrode_indices, events.electrode_indices)
    np.testing.assert_allclose(session.clusterless_spike_events.marks, events.marks)
    assert session.clusterless_spike_events.filename == events_path
