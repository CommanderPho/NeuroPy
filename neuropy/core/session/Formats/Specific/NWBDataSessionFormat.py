from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from neuropy.core import Epoch, FlattenedSpiketrains, Neurons, Position
from neuropy.core.parameters import ParametersContainer
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec
from neuropy.core.session.dataSession import DataSession
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.gettable_mixin import KeypathsAccessibleMixin
from neuropy.utils.result_context import IdentifyingContext


class NWBDataSessionFormatRegisteredClass(DataSessionFormatBaseRegisteredClass):
    """Loads DANDI NWB sessions into NeuroPy using NWB as source and `.npy` files as cache.

    v1 is targeted at DANDI 000978 folders like:
        download/000978/sub-JDS-SingleDay-ER1

    Known limitations:
    - Epoch labels use the notebook's alternating run/sleep heuristic.
    - Loaded units default to pyramidal neuron type.
    - ProbeGroup, LFP, MUA, ripple, laps, and replay loading are not implemented.
    - W-track linearization may fail without manual track configuration.
    """

    _session_class_name = "dandi_nwb"
    _session_default_relative_basedir = "download/000978/sub-JDS-SingleDay-ER1"
    _session_default_basedir = "/media/halechr/BETAMAX1/Data/CRCNS/download/000978/sub-JDS-SingleDay-ER1"
    _session_basepath_to_context_parsing_keys = ["format_name", "animal", "exper_name", "session_name"]
    _time_variable_name = "t_seconds"
    _single_day_prefix = "sub-JDS-SingleDay-"

    @classmethod
    def build_default_preprocessing_parameters(cls, **kwargs) -> ParametersContainer:
        override_parameters_flat_keypaths_dict = kwargs.pop("override_parameters_flat_keypaths_dict", {}) or {}
        override_parameters_nested_dicts = KeypathsAccessibleMixin.keypath_dict_to_nested_dict(override_parameters_flat_keypaths_dict)
        nwb_overrides = override_parameters_nested_dicts.get("nwb", {}) | {k: v for k, v in override_parameters_flat_keypaths_dict.items() if k in {"unit_location_filter", "nwb_filename", "epoch_label_mode", "export_root"}}
        preprocessing_parameters = ParametersContainer(epoch_estimation_parameters=DynamicContainer.init_from_dict({}))
        preprocessing_parameters.nwb = DynamicContainer(unit_location_filter="CA1", nwb_filename=None, epoch_label_mode="alternating_run_sleep", export_root=None).override(nwb_overrides)
        return preprocessing_parameters


    @classmethod
    def get_session_name(cls, basedir) -> str:
        subject = cls._parse_subject_from_basedir(basedir)
        return f"{subject}_SingleDay"


    @classmethod
    def parse_session_basepath_to_context(cls, basedir) -> IdentifyingContext:
        basedir = Path(basedir)
        subject = cls._parse_subject_from_basedir(basedir)
        dandiset_id = cls._parse_dandiset_id_from_basedir(basedir)
        return IdentifyingContext(format_name=cls.get_session_format_name(), animal=subject, exper_name=dandiset_id, session_name="SingleDay")


    @classmethod
    def get_session_spec(cls, session_name) -> SessionFolderSpec:
        return SessionFolderSpec(required_files=[], optional_files=[])


    @classmethod
    def load_session(cls, session, debug_print=False):
        loaded_file_record_list = []
        session = cls._fallback_recinfo(None, session)
        cache_paths = cls._build_cache_paths(session)

        if cls._core_cache_exists(cache_paths):
            if debug_print:
                print(f"Loading NWB DataSession cache from {session.filePrefix.parent}")
            session = cls._load_core_cache_files(session, cache_paths)
            loaded_file_record_list.extend([cache_paths["neurons"], cache_paths["position"], cache_paths["paradigm"]])
        else:
            if debug_print:
                print(f"NWB cache missing. Loading source NWB from {session.basepath}")
            session = cls._load_session_from_nwb(session)
            cls._save_core_cache_files(session, cache_paths)
            loaded_file_record_list.append(cls.find_nwb_file(session.basepath, nwb_filename=cls._get_nwb_parameters(session).nwb_filename))

        session = cls._load_or_compute_flattened_spikes(session, cache_paths)
        if not session.position.has_linear_pos:
            session = cls._compute_linear_position_if_possible(session)
        session, _spikes_df = cls._default_compute_spike_interpolated_positions_if_needed(session, session.spikes_df, time_variable_name=cls._time_variable_name, force_recompute=False)
        spikes_df = session.spikes_df
        cls._add_missing_spikes_df_columns(spikes_df, session.neurons)
        return session, loaded_file_record_list


    @classmethod
    def build_filters_run_epochs(cls, sess, filter_name_suffix=None):
        from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters

        run_only_name_filter_fn = lambda names: list(filter(lambda elem: elem == "run", names))
        return build_custom_epochs_filters(sess, epoch_name_includelist=run_only_name_filter_fn, filter_name_suffix=filter_name_suffix)


    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_includelist=None, filter_name_suffix=None, include_global_epoch=False):
        return cls.build_filters_run_epochs(sess, filter_name_suffix=filter_name_suffix)


    @classmethod
    def find_nwb_file(cls, basedir, nwb_filename=None) -> Path:
        basedir = Path(basedir)
        if basedir.suffix == ".nwb":
            if nwb_filename is not None and basedir.name != nwb_filename:
                raise FileNotFoundError(f"NWB filename override {nwb_filename!r} did not match {basedir}")
            return basedir

        if nwb_filename is not None:
            selected_path = basedir / nwb_filename
            if not selected_path.exists():
                raise FileNotFoundError(f"NWB filename override {nwb_filename!r} was not found under {basedir}")
            return selected_path

        candidates = sorted(basedir.glob("*.nwb")) if basedir.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"No .nwb file found under {basedir}")
        if len(candidates) > 1:
            warnings.warn(f"Multiple NWB files for {basedir.name}; using {candidates[0].name}. Set nwb_filename to override.")
        return candidates[0]


    @classmethod
    def _fallback_recinfo(cls, filepath, session):
        session.filePrefix = cls._build_file_prefix(session)
        session.filePrefix.parent.mkdir(parents=True, exist_ok=True)
        session.recinfo = DynamicContainer(source_file=None, channel_groups=None, skipped_channels=None, discarded_channels=None, n_channels=None, dat_sampling_rate=None, eeg_sampling_rate=None)
        return session


    @classmethod
    def _load_neurons_file(cls, filepath, session):
        session.neurons = Neurons.from_file(filepath)
        return session


    @classmethod
    def _load_position_file(cls, filepath, session):
        session.position = Position.from_file(filepath)
        return session


    @classmethod
    def _load_paradigm_file(cls, filepath, session):
        session.paradigm = Epoch.from_file(filepath)
        return session


    @classmethod
    def _load_flattened_spikes_file(cls, filepath, session):
        session.flattened_spiketrains = FlattenedSpiketrains.from_file(filepath)
        return session


    @classmethod
    def _load_session_from_nwb(cls, session):
        from pynwb import NWBHDF5IO

        nwb_parameters = cls._get_nwb_parameters(session)
        nwb_path = cls.find_nwb_file(session.basepath, nwb_filename=nwb_parameters.nwb_filename)
        with NWBHDF5IO(str(nwb_path), mode="r") as io:
            nwbf = io.read()
            timestamps = cls._load_position_timestamps(nwbf)
            t0 = float(timestamps[0])
            session.config.absolute_start_timestamp = t0
            session.position = cls._load_position_from_nwb(nwbf, timestamps=timestamps, t0=t0)
            session.neurons = cls._load_neurons_from_nwb(nwbf, t0=t0, t_stop=float(session.position.time[-1]), unit_location_filter=nwb_parameters.unit_location_filter)
            session.paradigm = cls._load_paradigm_from_nwb(nwbf, t0=t0, epoch_label_mode=nwb_parameters.epoch_label_mode)
            session.recinfo.source_file = nwb_path
        if len(session.position.time) > 1:
            session.config.position_sampling_rate_Hz = float(1.0 / np.nanmean(np.diff(session.position.time)))
        return session


    @classmethod
    def _load_position_timestamps(cls, nwbf):
        spatial_series = cls._get_position_spatial_series(nwbf)
        return np.asarray(spatial_series.timestamps[:], dtype=float)


    @classmethod
    def _load_position_from_nwb(cls, nwbf, timestamps, t0):
        spatial_series = cls._get_position_spatial_series(nwbf)
        t_rel = timestamps - t0
        xy = np.asarray(spatial_series.data[:, 0:2], dtype=float)
        return Position.from_separate_arrays(t_rel, xy[:, 0], xy[:, 1])


    @classmethod
    def _load_neurons_from_nwb(cls, nwbf, t0, t_stop, unit_location_filter="CA1"):
        units_df = nwbf.units.to_dataframe()
        spiketrains, neuron_ids, shank_ids = [], [], []
        for unit_id, row in units_df.iterrows():
            location = cls._unit_electrode_field(row["electrodes"], "location")
            if unit_location_filter is not None and location != unit_location_filter:
                continue
            spiketrains.append(np.asarray(row["spike_times"], dtype=float) - t0)
            neuron_ids.append(int(unit_id))
            group_name = cls._unit_electrode_field(row["electrodes"], "group_name")
            shank_ids.append(cls._parse_shank_from_group_name(group_name))
        if not spiketrains:
            raise ValueError(f"No units matched location filter {unit_location_filter!r}")
        neuron_type = np.array(["pyr"] * len(neuron_ids))
        return Neurons(np.array(spiketrains, dtype=object), t_stop=t_stop, t_start=0.0, neuron_ids=neuron_ids, shank_ids=np.array(shank_ids, dtype=np.int64), neuron_type=neuron_type)


    @classmethod
    def _load_paradigm_from_nwb(cls, nwbf, t0, epoch_label_mode="alternating_run_sleep"):
        epochs_df = nwbf.intervals["epoch intervals"].to_dataframe().reset_index(drop=True)
        if epoch_label_mode != "alternating_run_sleep":
            raise ValueError(f"Unsupported epoch_label_mode: {epoch_label_mode!r}")
        # labels = np.array(["run" if i % 2 == 0 else "sleep" for i in range(len(epochs_df))], dtype=str)
        labels = np.array([f"maze{int(i/2)}" if i % 2 == 0 else f"sleep{int(i/2)}" for i in range(len(epochs_df))], dtype=str)
        epoch_types = ['maze' if 'maze' in k else 'sleep' for k in labels]
        return Epoch(pd.DataFrame({"start": epochs_df["start_time"].values - t0, "stop": epochs_df["stop_time"].values - t0, "label": labels, 'behavior': epoch_types}))


    @classmethod
    def _get_position_spatial_series(cls, nwbf):
        return nwbf.processing["behavior"]["Position"].spatial_series["SpatialSeries"]


    @classmethod
    def _unit_electrode_field(cls, electrodes_row, field: str) -> str:
        if hasattr(electrodes_row, "__getitem__") and field in getattr(electrodes_row, "columns", []):
            return str(electrodes_row[field].values[0])
        return ""


    @classmethod
    def _parse_shank_from_group_name(cls, group_name: str) -> int:
        digits = "".join(c for c in group_name if c.isdigit())
        return int(digits) if digits else 0


    @classmethod
    def _load_or_compute_flattened_spikes(cls, session, cache_paths):
        if cache_paths["flattened_spikes"].exists():
            session.flattened_spiketrains = FlattenedSpiketrains.from_file(cache_paths["flattened_spikes"])
            if session.flattened_spiketrains is not None:
                return session
        session = cls._default_compute_flattened_spikes(session, timestamp_scale_factor=1.0, spike_timestamp_column_name=cls._time_variable_name)
        session.flattened_spiketrains.filename = cache_paths["flattened_spikes"]
        session.flattened_spiketrains.save()
        return session


    @classmethod
    def _compute_linear_position_if_possible(cls, session):
        for epoch_label in session.epochs.labels:
            if epoch_label != "run":
                continue
            try:
                epoch_indices, _active_positions, linearized_positions = DataSession._perform_compute_session_linearized_position(session, epoch_label)
            except Exception as e:
                warnings.warn(f"Could not compute NWB linear position for epoch {epoch_label!r}: {e}")
                return session
            if not session.position.has_linear_pos:
                session.position.linear_pos = np.full_like(session.position.time, np.nan)
            session.position.linear_pos[epoch_indices] = linearized_positions.traces
        session.position.filename = session.filePrefix.with_suffix(".position.npy")
        session.position.save()
        return session


    @classmethod
    def _build_cache_paths(cls, session) -> dict:
        return {"neurons": session.filePrefix.with_suffix(".neurons.npy"), "position": session.filePrefix.with_suffix(".position.npy"), "paradigm": session.filePrefix.with_suffix(".paradigm.npy"), "flattened_spikes": session.filePrefix.with_suffix(".flattened.spikes.npy")}


    @classmethod
    def _core_cache_exists(cls, cache_paths) -> bool:
        return cache_paths["neurons"].exists() and cache_paths["position"].exists() and cache_paths["paradigm"].exists()


    @classmethod
    def _load_core_cache_files(cls, session, cache_paths):
        session = cls._load_neurons_file(cache_paths["neurons"], session)
        session = cls._load_position_file(cache_paths["position"], session)
        session = cls._load_paradigm_file(cache_paths["paradigm"], session)
        return session


    @classmethod
    def _save_core_cache_files(cls, session, cache_paths):
        session.neurons.filename = cache_paths["neurons"]
        session.position.filename = cache_paths["position"]
        session.paradigm.filename = cache_paths["paradigm"]
        session.neurons.save()
        session.position.save()
        session.paradigm.save()


    @classmethod
    def _build_file_prefix(cls, session) -> Path:
        nwb_parameters = cls._get_nwb_parameters(session)
        subject = cls._parse_subject_from_basedir(session.basepath)
        dandiset_id = cls._parse_dandiset_id_from_basedir(session.basepath)
        export_root = Path(nwb_parameters.export_root).expanduser() if nwb_parameters.export_root is not None else cls._derive_repo_root(session.basepath) / "export" / dandiset_id
        return export_root / subject / cls.get_session_name(session.basepath)


    @classmethod
    def _get_nwb_parameters(cls, session):
        preprocessing_parameters = session.config.preprocessing_parameters
        if not hasattr(preprocessing_parameters, "nwb"):
            preprocessing_parameters.nwb = DynamicContainer(unit_location_filter="CA1", nwb_filename=None, epoch_label_mode="alternating_run_sleep", export_root=None)
        return preprocessing_parameters.nwb


    @classmethod
    def _parse_subject_from_basedir(cls, basedir) -> str:
        basedir = Path(basedir)
        subject_dir_name = basedir.parent.name if basedir.suffix == ".nwb" else basedir.name
        if subject_dir_name.startswith(cls._single_day_prefix):
            return subject_dir_name[len(cls._single_day_prefix):]
        if subject_dir_name.startswith("sub-"):
            return subject_dir_name.removeprefix("sub-")
        return subject_dir_name


    @classmethod
    def _parse_dandiset_id_from_basedir(cls, basedir) -> str:
        basedir = Path(basedir)
        subject_dir = basedir.parent if basedir.suffix == ".nwb" else basedir
        return subject_dir.parent.name


    @classmethod
    def _derive_repo_root(cls, basedir) -> Path:
        basedir = Path(basedir).resolve()
        subject_dir = basedir.parent if basedir.suffix == ".nwb" else basedir
        return subject_dir.parent.parent.parent
