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


def test_time_sliced_supports_multiple_intervals():
    events = ClusterlessSpikeEvents(spike_times_sec=np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32), electrode_indices=np.array([0, 0, 1, 1, 2], dtype=np.int16),
        marks=np.ones((5, 4), dtype=np.float32), sampling_frequency_hz=1000.0, electrode_mode="channel", t_start=0.0, t_stop=3.0)

    sliced = events.time_sliced(t_start=np.array([0.0, 1.25]), t_stop=np.array([0.75, 1.75]))

    np.testing.assert_allclose(sliced.spike_times_sec, np.array([0.5, 1.5], dtype=np.float32))
    np.testing.assert_array_equal(sliced.electrode_indices, np.array([0, 1], dtype=np.int16))
    assert sliced.t_start == 0.0
    assert sliced.t_stop == 1.75


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


def test_bapun_session_spec_accepts_clusterless_spikes_without_neurons(tmp_path: Path):
    from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass

    session_name = "Synthetic"
    for suffix in (".xml", ".probegroup.npy", ".position.npy", ".paradigm.npy"):
        (tmp_path / f"{session_name}{suffix}").touch()
    save_clusterless_spike_events(tmp_path / f"{session_name}.clusterless_spikes.npz", _build_events())

    spec = BapunDataSessionFormatRegisteredClass.get_session_spec(session_name)
    required_filenames = {file_spec.filename for file_spec in spec.required_files}
    optional_filenames = {file_spec.filename for file_spec in spec.optional_files}
    meets_spec, _, _ = getattr(spec, "validate")(tmp_path, False)

    assert f"{session_name}.neurons.npy" not in required_filenames
    assert f"{session_name}.neurons.npy" in optional_filenames
    assert f"{session_name}.clusterless_spikes.npz" in optional_filenames
    assert meets_spec


def test_bapun_session_spec_rejects_missing_neurons_and_clusterless_spikes(tmp_path: Path):
    from neuropy.core.session.Formats.SessionSpecifications import RequiredValidationFailedError
    from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass

    session_name = "Synthetic"
    for suffix in (".xml", ".probegroup.npy", ".position.npy", ".paradigm.npy"):
        (tmp_path / f"{session_name}{suffix}").touch()
    spec = BapunDataSessionFormatRegisteredClass.get_session_spec(session_name)

    with pytest.raises(RequiredValidationFailedError):
        getattr(spec, "validate")(tmp_path, False)


def test_bapun_optional_spike_source_loader_accepts_clusterless_without_neurons(tmp_path: Path):
    from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass

    events = _build_events()
    session_name = "Synthetic"
    events_path = tmp_path / f"{session_name}.clusterless_spikes.npz"
    save_clusterless_spike_events(events_path, events)
    session = SimpleNamespace(filePrefix=tmp_path / session_name, config=SimpleNamespace(session_name=session_name))

    loaded_session, loaded_files = getattr(BapunDataSessionFormatRegisteredClass, "_try_load_optional_spike_sources")(session, loaded_file_record_list=[])

    assert loaded_session is session
    assert loaded_files == [events_path]
    assert not hasattr(session, "neurons")
    assert isinstance(session.clusterless_spike_events, ClusterlessSpikeEvents)


def _write_synthetic_phy_folder(phy_folder: Path, sample_rate_hz: float = 30000.0) -> None:
    phy_folder.mkdir(parents=True, exist_ok=True)
    (phy_folder / "params.py").write_text(f"sample_rate = {sample_rate_hz}\n", encoding="utf-8")
    spike_times = np.array([int(1.0 * sample_rate_hz), int(1.001 * sample_rate_hz), int(1.002 * sample_rate_hz), int(1.5 * sample_rate_hz), int(2.0 * sample_rate_hz)], dtype=np.int64)
    spike_templates = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    pc_feature_ind = np.array([[0, 1, -1, -1], [1, 2, -1, -1]], dtype=np.int64)
    pc_features = np.zeros((len(spike_times), 4, 4), dtype=np.float32)
    pc_features[0, :, 0] = np.array([1.0, 0.1, 0.2, 0.3], dtype=np.float32)
    pc_features[1, :, 1] = np.array([2.0, 0.4, 0.5, 0.6], dtype=np.float32)
    pc_features[2, :, 0] = np.array([3.0, 0.7, 0.8, 0.9], dtype=np.float32)
    pc_features[3, :, 1] = np.array([4.0, 1.0, 1.1, 1.2], dtype=np.float32)
    pc_features[4, :, 0] = np.array([5.0, 1.3, 1.4, 1.5], dtype=np.float32)
    channel_map = np.array([0, 1, 2], dtype=np.int32)
    np.save(phy_folder / "spike_times.npy", spike_times)
    np.save(phy_folder / "spike_templates.npy", spike_templates)
    np.save(phy_folder / "pc_feature_ind.npy", pc_feature_ind)
    np.save(phy_folder / "pc_features.npy", pc_features)
    np.save(phy_folder / "channel_map.npy", channel_map)


def test_from_phy_folder_synthetic(tmp_path: Path):
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder)
    events = ClusterlessSpikeEvents.from_phy_folder(phy_folder, t_start=1.0, t_end=2.0, electrode_mode="channel")
    assert events.spike_times_sec.dtype == np.float32
    assert events.electrode_indices.dtype == np.int16
    assert events.marks.dtype == np.float32
    assert events.marks.shape[1] == 4
    assert len(events.spike_times_sec) == 5
    assert np.all((events.spike_times_sec >= 1.0) & (events.spike_times_sec <= 2.0))
    assert events.source_phy_path == str(phy_folder)


def test_build_clusterless_spikes_from_phy_creates_npz(tmp_path: Path):
    from neuropy.core.session.init_from_raw_data import RawDataInitializationMixin

    session_name = "Synthetic"
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder)
    sess = SimpleNamespace(name=session_name, filePrefix=tmp_path / session_name, config=SimpleNamespace(session_name=session_name))
    clusterless_save_path = tmp_path / f"{session_name}.clusterless_spikes.npz"
    assert not clusterless_save_path.exists()

    result = RawDataInitializationMixin.build_clusterless_spikes_from_phy(sess, basedir=tmp_path, phy_folder=phy_folder)

    assert result is not None
    assert clusterless_save_path.exists()
    assert isinstance(sess.clusterless_spike_events, ClusterlessSpikeEvents)
    assert sess.clusterless_spike_events.n_spikes == 5
    loaded = load_clusterless_spike_events(clusterless_save_path)
    assert loaded.n_spikes == 5


def test_build_clusterless_spikes_from_phy_skips_existing(tmp_path: Path):
    from neuropy.core.session.init_from_raw_data import RawDataInitializationMixin

    session_name = "Synthetic"
    events = _build_events()
    clusterless_save_path = tmp_path / f"{session_name}.clusterless_spikes.npz"
    save_clusterless_spike_events(clusterless_save_path, events)
    sess = SimpleNamespace(name=session_name, filePrefix=tmp_path / session_name, config=SimpleNamespace(session_name=session_name))

    result = RawDataInitializationMixin.build_clusterless_spikes_from_phy(sess, basedir=tmp_path, phy_folder=tmp_path / "missing_phy")

    assert result is not None
    assert isinstance(sess.clusterless_spike_events, ClusterlessSpikeEvents)
    np.testing.assert_allclose(sess.clusterless_spike_events.spike_times_sec, events.spike_times_sec)


def test_data_session_time_slice_filters_clusterless_spike_events():
    import pandas as pd
    from neuropy.core.epoch import Epoch
    from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
    from neuropy.core.neurons import Neurons
    from neuropy.core.position import Position
    from neuropy.core.session.dataSession import DataSession

    events = ClusterlessSpikeEvents(spike_times_sec=np.array([0.3, 0.9, 1.2, 1.5], dtype=np.float32), electrode_indices=np.array([0, 1, 1, 2], dtype=np.int16),
        marks=np.ones((4, 4), dtype=np.float32), sampling_frequency_hz=1000.0, electrode_mode="channel", t_start=0.0, t_stop=2.0)
    spiketrains = np.array([np.array([0.3, 0.9, 1.2]), np.array([1.5])], dtype=object)
    neurons_obj = Neurons(spiketrains, t_stop=2.0, sampling_rate=30000, neuron_ids=np.array([1, 2]))
    spikes_df = pd.DataFrame({"t_seconds": [0.3, 0.9, 1.2, 1.5], "aclu": [1, 1, 1, 2]})
    position_df = pd.DataFrame({"t": [0.0, 0.5, 1.0, 1.5, 2.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [0.0, 0.0, 0.0, 0.0, 0.0]})
    paradigm = Epoch.from_starts_stops_arrays(starts=np.array([0.0]), stops=np.array([2.0]), labels=np.array(["global"]))
    sess = DataSession(config=SimpleNamespace(session_name="test"), neurons=neurons_obj, position=Position(position_df), paradigm=paradigm,
        flattened_spiketrains=FlattenedSpiketrains(spikes_df, time_variable_name="t_seconds"), clusterless_spike_events=events)

    filtered_sess = sess.time_slice(0.0, 1.0, enable_debug=False)

    assert filtered_sess.clusterless_spike_events is not None
    np.testing.assert_allclose(filtered_sess.clusterless_spike_events.spike_times_sec, np.array([0.3, 0.9], dtype=np.float32))
    assert filtered_sess.clusterless_spike_events.t_start == 0.0
    assert filtered_sess.clusterless_spike_events.t_stop == 1.0
