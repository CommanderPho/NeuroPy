from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional
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
from neuropy.core.session.Formats.BaseDataSessionFormats import HardcodedProcessingParameters


class NWBDataSessionFormatRegisteredClass(DataSessionFormatBaseRegisteredClass):
    """Loads DANDI NWB sessions into NeuroPy using NWB as source and `.npy` files as cache.

    v1 is targeted at DANDI 000978 folders like:
        download/000978/sub-JDS-SingleDay-ER1

    Known limitations:
    - Epoch labels use the notebook's alternating run/sleep heuristic.
    - Loaded units default to pyramidal neuron type.
    - ProbeGroup, LFP, MUA, ripple, laps, and replay loading are not implemented.
    - W-track linearization may fail without manual track configuration.


    Usage:
        from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass

        pos_df = curr_active_pipeline.sess.position.to_dataframe()
        ((pos_df['x'].min(), pos_df['x'].max()), pos_df['y'].min(), pos_df['y'].max()) # ((41.37033775405482, 157.72257208195566), (9.76773097884994, 122.8597412183469))

        maze_only_epochs = epochs_df[epochs_df['behavior'] == 'maze']['label'].tolist() # ['maze0', 'maze1', 'maze2', 'maze3', 'maze4', 'maze5', 'maze6', 'maze7', 'maze8']
        maze_only_epochs


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
    def _get_session_specific_parameters(cls, session_context: IdentifyingContext) -> HardcodedProcessingParameters:
        """ session-specific type parameters 
         
        #TODO 2025-09-20 19:26: - [ ] Is this redudndant with preprocessing parameters?
        """
        
        maze_grid_bin_bounds = (((41.37033775405482, 157.72257208195566), (9.76773097884994, 122.8597412183469)))
        # Custom Lap Building Functions ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #

        # def _subfn_rat_U_Day4Openfield_build_Bapun_Day5OpenfieldSD_laps_from_reward_zones(session):
        #     """ captures: cls, _subfn_rat_U_Day5OpenfieldSD_reward_zones """
        #     bapun_OpenField_reward_zones = _subfn_rat_U_Day5OpenfieldSD_reward_zones(session=session)
        #     return cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=bapun_OpenField_reward_zones)

        # lambda session: cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=_subfn_rat_U_Day5OpenfieldSD_reward_zones(session=session))

        the_dict: Dict[IdentifyingContext, HardcodedProcessingParameters]  = { #  
            # W MAze _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # IdentifyingContext(format_name= 'dandi_nwb', animal= 'ER1', exper_name= '000978', session_name= 'SingleDay'): HardcodedProcessingParameters( # format_name='DANDI', exper_name='SingleDayWTrackLearning', animal='ER1', dandi_id='000978', session_name='sub-JDS-SingleDay-ER1'
            #     decoder_building_session_names=['maze0', 'maze1', 'maze2', 'maze3', 'maze4', 'maze5', 'maze6', 'maze7', 'maze8', 'maze_GLOBAL'],
            #     global_session_name='maze_GLOBAL',
            #     non_global_activity_session_names=['maze0', 'maze1', 'maze2', 'maze3', 'maze4', 'maze5', 'maze6', 'maze7', 'maze8'],
            #     grid_bin_bounds=maze_grid_bin_bounds,
            #     lap_estimation_parameters=dict(reward_zones=None, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
            #     linearization_parameters=dict(method='umap', all_session_mazes=None),
            # ),

            IdentifyingContext(format_name= 'dandi_nwb', animal= 'ER1', exper_name= '000978', session_name= 'SingleDay'): HardcodedProcessingParameters( # format_name='DANDI', exper_name='SingleDayWTrackLearning', animal='ER1', dandi_id='000978', session_name='sub-JDS-SingleDay-ER1'
                decoder_building_session_names=['maze0', 'maze1', 'maze2', 'maze3', 'maze4', 'maze5', 'maze6', 'maze7', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze0', 'maze1', 'maze2', 'maze3', 'maze4', 'maze5', 'maze6', 'maze7'],
                grid_bin_bounds=maze_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=None, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            

            ## Fallback defaults:
            IdentifyingContext(format_name= 'dandi_nwb'): HardcodedProcessingParameters(decoder_building_session_names=['maze0', 'maze1', 'maze2', 'maze3', 'maze4', 'maze5', 'maze6', 'maze7', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze0', 'maze1', 'maze2', 'maze3', 'maze4', 'maze5', 'maze6', 'maze7'],
                grid_bin_bounds=maze_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=None, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0,),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),									
        }

        best_match = IdentifyingContext.matching(the_dict, criteria=session_context.get_subset(subset_includelist=cls._session_basepath_to_context_parsing_keys).to_dict())
        return list(best_match.values())[0] ## return the first match


    @classmethod
    def _standardize_alternating_run_sleep_epoch_labels(cls, epochs_df: pd.DataFrame) -> pd.DataFrame:
        """Convert DANDI alternating run/sleep epoch rows to maze0/sleep0 labels."""
        standardized_epochs_df = epochs_df.copy().reset_index(drop=True)
        # labels = np.array([f"maze{int(i/2)}" if i % 2 == 0 else f"sleep{int(i/2)}" for i in range(len(standardized_epochs_df))], dtype=str) ## if it starts with maze, and then sleep
        labels = np.array([f"sleep{int(i/2)}" if i % 2 == 0 else f"maze{int(i/2)}" for i in range(len(standardized_epochs_df))], dtype=str) ## starts with sleep, then maze.
        standardized_epochs_df['label'] = labels
        standardized_epochs_df['behavior'] = ['maze' if a_label.startswith('maze') else 'sleep' for a_label in labels]
        standardized_epochs_df['duration'] = standardized_epochs_df['stop'] - standardized_epochs_df['start']
        return standardized_epochs_df


    @classmethod
    def _paradigm_labels_are_legacy(cls, paradigm) -> bool:
        if paradigm is None:
            return False
        epochs_df = paradigm.to_dataframe() if hasattr(paradigm, 'to_dataframe') else paradigm.copy()
        labels = epochs_df['label'].astype(str).tolist()
        non_global_labels = [a_label for a_label in labels if a_label != 'maze_GLOBAL']
        has_legacy_run_sleep_labels = any(a_label in {'run', 'sleep'} for a_label in non_global_labels)
        missing_expected_first_maze = 'maze0' not in non_global_labels
        return len(non_global_labels) > 0 and (has_legacy_run_sleep_labels or missing_expected_first_maze)


    @classmethod
    def _get_activity_epoch_labels(cls, sess) -> list:
        if sess is None or sess.epochs is None:
            return []
        return [str(a_label) for a_label in sess.epochs.get_unique_labels() if str(a_label).startswith('maze') and str(a_label) != 'maze_GLOBAL']


    @classmethod
    def _ensure_standard_paradigm_epoch_labels(cls, session, save_if_changed: bool=True) -> bool:
        if session is None or session.paradigm is None or not cls._paradigm_labels_are_legacy(session.paradigm):
            return False

        paradigm_filename = getattr(session.paradigm, 'filename', None)
        paradigm_metadata = getattr(session.paradigm, 'metadata', None)
        epochs_df = session.paradigm.to_dataframe()
        non_global_epochs_df = epochs_df[epochs_df['label'].astype(str) != 'maze_GLOBAL'].copy().reset_index(drop=True)
        standardized_epochs_df = cls._standardize_alternating_run_sleep_epoch_labels(non_global_epochs_df)
        standardized_paradigm = Epoch(standardized_epochs_df, metadata=paradigm_metadata)
        if paradigm_filename is not None:
            standardized_paradigm.filename = Path(paradigm_filename)
        session.paradigm = standardized_paradigm
        session.epochs = standardized_paradigm

        if hasattr(session, 'epochs_bak'):
            delattr(session, 'epochs_bak')

        if save_if_changed and session.paradigm.filename is not None:
            print(f'migrating stale NWB paradigm epoch labels and saving: {session.paradigm.filename}')
            session.paradigm.save(status_print=False)
        else:
            print('migrating stale NWB paradigm epoch labels in-memory.')
        return True


    @classmethod
    def _epoch_labels_include(cls, epochs_obj, required_epoch_names) -> bool:
        if epochs_obj is None:
            return False
        epoch_labels = set(str(a_label) for a_label in epochs_obj.get_unique_labels())
        return all(a_name in epoch_labels for a_name in required_epoch_names)


    @classmethod
    def _deduplicate_spikes_df_columns(cls, spikes_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        duplicate_column_mask = spikes_df.columns.duplicated()
        if not duplicate_column_mask.any():
            return spikes_df, False
        duplicate_column_names = spikes_df.columns[duplicate_column_mask].tolist()
        print(f'migrating stale NWB spikes_df duplicate columns; keeping first occurrence for: {duplicate_column_names}')
        return spikes_df.loc[:, ~duplicate_column_mask].copy(), True


    @classmethod
    def _ensure_flattened_spikes_df_columns_unique(cls, session, save_if_changed: bool=True) -> bool:
        if session is None or getattr(session, 'flattened_spiketrains', None) is None:
            return False
        spikes_df, did_change = cls._deduplicate_spikes_df_columns(session.flattened_spiketrains.spikes_df)
        if not did_change:
            return False
        session.flattened_spiketrains._spikes_df = spikes_df
        if save_if_changed and getattr(session.flattened_spiketrains, 'filename', None) is not None:
            session.flattened_spiketrains.save(status_print=False)
        return True


    @classmethod
    def session_fixup_epochs(cls, sess, override_session_epochs: Optional[Epoch]=None, enable_global_epoch: bool=True, override_extant: bool=True, **kwargs) -> Epoch:
        """Add derived epochs (e.g. maze_GLOBAL) for DANDI NWB W-track sessions."""
        cls._ensure_standard_paradigm_epoch_labels(sess, save_if_changed=True)
        hardcoded_params = cls._get_session_specific_parameters(session_context=sess.get_context())
        required_epoch_names = hardcoded_params.non_global_activity_session_names
        updated_epochs: Epoch = deepcopy(sess.epochs) if override_session_epochs is None else deepcopy(override_session_epochs)

        if not hasattr(sess, 'epochs_bak'):
            print('fixing up NWB session computation epochs...')
            sess.epochs_bak = deepcopy(updated_epochs)
        else:
            print('WARN: already fixedup session epochs.')
            if override_extant:
                if cls._epoch_labels_include(sess.epochs_bak, required_epoch_names):
                    print('\trestoring backed up epochs:')
                    sess.epochs = deepcopy(sess.epochs_bak)
                    updated_epochs = deepcopy(sess.epochs)
                else:
                    print('\tdiscarding incompatible epochs_bak because it does not contain the expected NWB maze labels.')
                    delattr(sess, 'epochs_bak')
                    sess.epochs_bak = deepcopy(updated_epochs)

        epochs_df = updated_epochs.to_dataframe()
        if enable_global_epoch and hardcoded_params.global_session_name not in epochs_df['label'].tolist():
            available_non_global_names = [a_name for a_name in required_epoch_names if a_name in epochs_df['label'].astype(str).tolist()]
            if len(available_non_global_names) < 1:
                raise ValueError(f"Could not add {hardcoded_params.global_session_name!r}; none of the expected NWB maze labels were present. expected={required_epoch_names}, actual={epochs_df['label'].astype(str).tolist()}")
            epochs_df = epochs_df.epochs.adding_global_epoch_row(global_epoch_name=hardcoded_params.global_session_name, first_included_epoch_name=available_non_global_names[0], last_included_epoch_name=available_non_global_names[-1], inplace=False)
            updated_epochs = Epoch(epochs_df, metadata=updated_epochs.metadata)
            existing_filename = getattr(sess.epochs, 'filename', None)
            if existing_filename is not None:
                updated_epochs.filename = existing_filename
        sess.epochs = updated_epochs
        print(f'\tdone. new epochs: \n{updated_epochs}\n')
        return updated_epochs


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
        cls._ensure_flattened_spikes_df_columns_unique(session, save_if_changed=True)
        spikes_df = session.spikes_df
        cls._add_missing_spikes_df_columns(spikes_df, session.neurons)
        return session, loaded_file_record_list


    @classmethod
    def build_filters_run_epochs(cls, sess, filter_name_suffix=None):
        from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters

        cls._ensure_standard_paradigm_epoch_labels(sess, save_if_changed=True)
        maze_epoch_names = cls._get_activity_epoch_labels(sess)
        return build_custom_epochs_filters(sess, epoch_name_includelist=maze_epoch_names, filter_name_suffix=filter_name_suffix)


    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_includelist=None, filter_name_suffix=None, include_global_epoch=False):
        from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters

        cls._ensure_standard_paradigm_epoch_labels(sess, save_if_changed=True)
        if epoch_name_includelist is None:
            epoch_name_includelist = cls._get_activity_epoch_labels(sess)
        return build_custom_epochs_filters(sess, epoch_name_includelist=epoch_name_includelist, filter_name_suffix=filter_name_suffix)


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
        cls._ensure_standard_paradigm_epoch_labels(session, save_if_changed=True)
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
        epochs_df = pd.DataFrame({"start": epochs_df["start_time"].values - t0, "stop": epochs_df["stop_time"].values - t0})
        return Epoch(cls._standardize_alternating_run_sleep_epoch_labels(epochs_df))


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
                cls._ensure_flattened_spikes_df_columns_unique(session, save_if_changed=True)
                return session
        session = cls._default_compute_flattened_spikes(session, timestamp_scale_factor=1.0, spike_timestamp_column_name=cls._time_variable_name)
        cls._ensure_flattened_spikes_df_columns_unique(session, save_if_changed=False)
        session.flattened_spiketrains.filename = cache_paths["flattened_spikes"]
        session.flattened_spiketrains.save()
        return session


    @classmethod
    def _compute_linear_position_if_possible(cls, session):
        cls._ensure_standard_paradigm_epoch_labels(session, save_if_changed=True)
        for epoch_label in cls._get_activity_epoch_labels(session):
            try:
                epoch_indices, _active_positions, linearized_positions = DataSession._perform_compute_session_linearized_position(session, epoch_label)
            except Exception as e:
                warnings.warn(f"Could not compute NWB linear position for epoch {epoch_label!r}: {e}")
                continue
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
