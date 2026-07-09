from __future__ import annotations

import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from neuropy.core import Epoch, Neurons, Position
from neuropy.core.session.Formats.BaseDataSessionFormats import HardcodedProcessingParameters
from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.SessionSpecifications import SessionConfig
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
from neuropy.core.session.dataSession import DataSession
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.gettable_mixin import KeypathsAccessibleMixin
from neuropy.utils.result_context import IdentifyingContext


_SLEEP_STATE_BEHAVIOR_MAP = {'WAKE': 'wake', 'NREM': 'nrem', 'REM': 'rem', 'Ripple': 'ripple'}
_ACTIVITY_BEHAVIORS = frozenset({'wake'})
_CELL_TYPE_TO_NEURON_TYPE = {'Pyramidal Cell': 'pyr', 'Narrow Interneuron': 'intr', 'Wide Interneuron': 'intr'}


class DANDI001695NWBDataSessionFormatRegisteredClass(NWBDataSessionFormatRegisteredClass):
    """Loads DANDI 001695 (HighDensityCrossBrain) NWB sessions into NeuroPy.

    Target layout:
        HighDensityCrossBrain/001695/sub-M01/sub-M01_ses-20240312T100000_behavior+ecephys.nwb

    Usage:
        from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

        sess = DANDI001695NWBDataSessionFormatRegisteredClass.get_session(
            basedir=Path(r'H:\\Data\\DANDI\\HighDensityCrossBrain\\001695\\sub-M01'),
            override_parameters_flat_keypaths_dict={'nwb.nwb_filename': 'sub-M01_ses-20240312T100000_behavior+ecephys.nwb'},  # optional; defaults to first *.nwb under basedir
        )
    """

    _session_class_name = "dandi_nwb_001695"
    _session_default_relative_basedir = "HighDensityCrossBrain/001695/sub-M01"
    _session_default_basedir = "H:/Data/DANDI/HighDensityCrossBrain/001695/sub-M01"
    _session_basepath_to_context_parsing_keys = ["format_name", "animal", "exper_name", "session_name"]
    _dandiset_id = "001695"


    @classmethod
    def build_default_preprocessing_parameters(cls, **kwargs):
        override_parameters_flat_keypaths_dict = kwargs.pop("override_parameters_flat_keypaths_dict", {}) or {}
        preprocessing_parameters = super().build_default_preprocessing_parameters(override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict, **kwargs)
        preprocessing_parameters.epoch_estimation_parameters.laps.use_direction_dependent_laps = False
        override_parameters_nested_dicts = KeypathsAccessibleMixin.keypath_dict_to_nested_dict(override_parameters_flat_keypaths_dict)
        nwb_overrides = override_parameters_nested_dicts.get("nwb", {}) | {k: v for k, v in override_parameters_flat_keypaths_dict.items() if k in {"unit_location_filter", "nwb_filename", "epoch_label_mode", "export_root", "force_recompute_linear_position"}}
        preprocessing_parameters.nwb = DynamicContainer(unit_location_filter="CA1", nwb_filename=None, epoch_label_mode="sleep_states", export_root=None, force_recompute_linear_position=False).override(nwb_overrides)
        return preprocessing_parameters


    @classmethod
    def get_known_data_session_type_properties(cls, override_basepath=None, override_parameters_flat_keypaths_dict=None):
        if override_basepath is not None:
            basepath = override_basepath
        else:
            basepath = Path(cls._session_default_basedir)
        return KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: cls.get_session(basedir=a_base_dir, override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)), basedir=basepath, post_load_functions=[lambda a_loaded_sess: cls.POSTLOAD_estimate_laps_and_replays(a_loaded_sess)])


    @classmethod
    def _parse_session_id_from_nwb_filename(cls, nwb_filename: str) -> str:
        match = re.search(r"(ses-[^_.]+)", str(nwb_filename))
        if match is not None:
            return match.group(1)
        return Path(nwb_filename).stem


    @classmethod
    def _resolve_nwb_filename(cls, basedir, override_parameters_flat_keypaths_dict=None) -> Optional[str]:
        basedir = Path(basedir)
        if override_parameters_flat_keypaths_dict:
            override_filename = override_parameters_flat_keypaths_dict.get("nwb.nwb_filename")
            if override_filename is not None:
                return str(override_filename)
        if basedir.suffix == ".nwb":
            return basedir.name
        return None


    @classmethod
    def _nwb_filename_from_basedir(cls, basedir, nwb_filename: Optional[str] = None) -> str:
        if nwb_filename is not None:
            return nwb_filename
        return cls.find_nwb_file(Path(basedir)).name


    @classmethod
    def parse_session_basepath_to_context(cls, basedir, nwb_filename: Optional[str] = None) -> IdentifyingContext:
        basedir = Path(basedir)
        nwb_filename = cls._nwb_filename_from_basedir(basedir, nwb_filename=basedir.name if basedir.suffix == ".nwb" else nwb_filename)
        return IdentifyingContext(format_name=cls.get_session_format_name(), animal=cls._parse_subject_from_basedir(basedir), exper_name=cls._parse_dandiset_id_from_basedir(basedir), session_name=cls._parse_session_id_from_nwb_filename(nwb_filename))


    @classmethod
    def build_session(cls, basedir, override_parameters_flat_keypaths_dict=None, enable_continue_on_required_path_failure: bool = False):
        basedir = Path(basedir)
        nwb_filename = cls._resolve_nwb_filename(basedir, override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)
        if nwb_filename is None:
            nwb_filename = cls.find_nwb_file(basedir).name
        session_name = cls._parse_session_id_from_nwb_filename(nwb_filename)
        session_context = cls.parse_session_basepath_to_context(basedir, nwb_filename=nwb_filename)
        session_spec = cls.get_session_spec(session_name)
        format_name = cls.get_session_format_name()
        preprocessing_parameters = cls.build_default_preprocessing_parameters(override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict, session_context=session_context)
        session_config = SessionConfig(basedir, format_name=format_name, session_spec=session_spec, session_name=session_name, session_context=session_context, preprocessing_parameters=preprocessing_parameters, enable_continue_on_required_path_failure=enable_continue_on_required_path_failure)
        if not enable_continue_on_required_path_failure:
            assert session_config.is_resolved, "active_sess_config could not be resolved!"
        else:
            print('ERROR: active_sess_config could not be resolved! BUT since `enable_continue_on_required_path_failure == True` continuing on anyway...')
        return DataSession(session_config)


    @classmethod
    def get_session_name(cls, basedir) -> str:
        basedir = Path(basedir)
        nwb_filename = basedir.name if basedir.suffix == ".nwb" else None
        return cls._parse_session_id_from_nwb_filename(cls._nwb_filename_from_basedir(basedir, nwb_filename=nwb_filename))


    @classmethod
    def _get_session_specific_parameters(cls, session_context: IdentifyingContext) -> HardcodedProcessingParameters:
        default_grid_bin_bounds = (((0.0, 100.0), (0.0, 100.0)))
        default_lap_params = dict(reward_zones=None, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration=2.5, minimum_run_speed=5.0, merging_adjacent_max_separation_sec=6.0)
        default_decoder_names = ['WAKE0', 'activity_GLOBAL']
        the_dict: Dict[IdentifyingContext, HardcodedProcessingParameters] = {
            IdentifyingContext(format_name='dandi_nwb_001695', animal='M01', exper_name='001695'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='activity_GLOBAL',
                non_global_activity_session_names=['WAKE0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name='dandi_nwb_001695', animal='M02', exper_name='001695'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='activity_GLOBAL',
                non_global_activity_session_names=['WAKE0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name='dandi_nwb_001695', animal='M03', exper_name='001695'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='activity_GLOBAL',
                non_global_activity_session_names=['WAKE0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name='dandi_nwb_001695', animal='M04', exper_name='001695'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='activity_GLOBAL',
                non_global_activity_session_names=['WAKE0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name='dandi_nwb_001695', animal='M05', exper_name='001695'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='activity_GLOBAL',
                non_global_activity_session_names=['WAKE0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),

            IdentifyingContext(format_name='dandi_nwb_001695', exper_name='001695'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='activity_GLOBAL',
                non_global_activity_session_names=['WAKE0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name='dandi_nwb_001695'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='activity_GLOBAL',
                non_global_activity_session_names=['WAKE0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
        }
        best_match = IdentifyingContext.matching(the_dict, criteria=session_context.get_subset(subset_includelist=cls._session_basepath_to_context_parsing_keys).to_dict())
        if len(list(best_match.values())) > 0:
            return list(best_match.values())[0]
        best_match, _max_num_matching_context_attributes = IdentifyingContext.find_best_matching_context(session_context.get_subset(subset_includelist=cls._session_basepath_to_context_parsing_keys), context_iterable=the_dict)
        return the_dict[best_match]


    @classmethod
    def _get_ecephys_t0(cls, nwbf) -> float:
        units_df = nwbf.units.to_dataframe()
        min_spike_time = np.inf
        for _, row in units_df.iterrows():
            spike_times = np.asarray(row['spike_times'], dtype=float)
            if spike_times.size > 0:
                min_spike_time = min(min_spike_time, float(np.min(spike_times)))

        ## END for _, row in units_df.iterrows()...

        return 0.0 if not np.isfinite(min_spike_time) else float(min_spike_time)


    @classmethod
    def _wake_epoch_labels_overlap_position(cls, sess, wake_labels: List[str]) -> List[str]:
        if sess is None or sess.position is None or len(sess.position.time) < 1 or len(wake_labels) < 1:
            return wake_labels
        pos_start, pos_stop = float(sess.position.time[0]), float(sess.position.time[-1])
        epochs_df = sess.epochs.to_dataframe()
        overlapping_wake_labels: List[str] = []
        for a_label in wake_labels:
            epoch_rows = epochs_df[epochs_df['label'].astype(str) == str(a_label)]
            if len(epoch_rows) < 1:
                continue
            epoch_row = epoch_rows.iloc[0]
            if float(epoch_row['stop']) > pos_start and float(epoch_row['start']) < pos_stop:
                overlapping_wake_labels.append(str(a_label))

        ## END for a_label in wake_labels...

        return overlapping_wake_labels


    @classmethod
    def _session_epoch_position_alignment_is_valid(cls, sess) -> bool:
        if sess is None or sess.paradigm is None or sess.position is None or len(sess.position.time) < 1:
            return True
        pos_start = float(sess.position.time[0])
        paradigm_df = sess.paradigm.to_dataframe()
        wake_epochs_df = paradigm_df[paradigm_df['label'].astype(str).str.startswith('WAKE')]
        if len(wake_epochs_df) < 1:
            return True
        if pos_start < 10.0 and float(wake_epochs_df['start'].min()) < 0.0:
            return False
        wake_labels = wake_epochs_df['label'].astype(str).tolist()
        return len(cls._wake_epoch_labels_overlap_position(sess, wake_labels)) > 0


    @classmethod
    def load_session(cls, session, debug_print=False):
        if getattr(session, '_dandi_001695_reloaded_for_alignment', False):
            return super().load_session(session, debug_print=debug_print)
        session, loaded_file_record_list = super().load_session(session, debug_print=debug_print)
        if cls._session_epoch_position_alignment_is_valid(session):
            return session, loaded_file_record_list
        if debug_print:
            print('DANDI 001695 epoch/position timebase misaligned; invalidating cache and reloading from NWB...')
        cache_paths = cls._build_cache_paths(session)
        for cache_key in ['neurons', 'position', 'paradigm']:
            cache_path = cache_paths[cache_key]
            if cache_path.exists():
                cache_path.unlink()
        session._dandi_001695_reloaded_for_alignment = True
        return cls.load_session(session, debug_print=debug_print)


    @classmethod
    def _get_activity_epoch_labels(cls, sess) -> list:
        if sess is None or sess.epochs is None:
            return []
        epochs_df = sess.epochs.to_dataframe()
        if 'behavior' in epochs_df.columns:
            wake_labels = [str(a_label) for a_label in epochs_df.loc[epochs_df['behavior'].astype(str).isin(_ACTIVITY_BEHAVIORS), 'label'].tolist()]
        else:
            wake_labels = [str(a_label) for a_label in epochs_df['label'].astype(str).tolist() if str(a_label).startswith('WAKE') and str(a_label) != 'activity_GLOBAL']
        return cls._wake_epoch_labels_overlap_position(sess, wake_labels)


    @classmethod
    def _get_decoder_building_epoch_labels(cls, sess) -> list:
        activity_labels = cls._get_activity_epoch_labels(sess)
        if len(activity_labels) == 0:
            return []
        return activity_labels + ['activity_GLOBAL']


    @classmethod
    def _ensure_standard_paradigm_epoch_labels(cls, session, save_if_changed: bool = True) -> bool:
        return False


    @classmethod
    def _paradigm_labels_are_legacy(cls, paradigm) -> bool:
        return False


    @classmethod
    def session_fixup_epochs(cls, sess, override_session_epochs: Optional[Epoch] = None, enable_global_epoch: bool = True, override_extant: bool = True, **kwargs) -> Epoch:
        hardcoded_params = cls._get_session_specific_parameters(session_context=sess.get_context())
        required_epoch_names = cls._get_activity_epoch_labels(sess) or hardcoded_params.non_global_activity_session_names
        updated_epochs: Epoch = deepcopy(sess.epochs) if override_session_epochs is None else deepcopy(override_session_epochs)
        if not hasattr(sess, 'epochs_bak'):
            print('fixing up DANDI 001695 session computation epochs...')
            sess.epochs_bak = deepcopy(updated_epochs)
        else:
            print('WARN: already fixedup session epochs.')
            if override_extant:
                if cls._epoch_labels_include(sess.epochs_bak, required_epoch_names):
                    print('\trestoring backed up epochs:')
                    sess.epochs = deepcopy(sess.epochs_bak)
                    updated_epochs = deepcopy(sess.epochs)
                else:
                    print('\tdiscarding incompatible epochs_bak because it does not contain the expected activity labels.')
                    delattr(sess, 'epochs_bak')
                    sess.epochs_bak = deepcopy(updated_epochs)
        epochs_df = updated_epochs.to_dataframe()
        if enable_global_epoch and hardcoded_params.global_session_name not in epochs_df['label'].astype(str).tolist():
            available_non_global_names = [a_name for a_name in required_epoch_names if a_name in epochs_df['label'].astype(str).tolist()]
            if len(available_non_global_names) < 1:
                raise ValueError(f"Could not add {hardcoded_params.global_session_name!r}; none of the expected activity epoch labels were present. expected={required_epoch_names}, actual={epochs_df['label'].astype(str).tolist()}")
            epochs_df = epochs_df.epochs.adding_global_epoch_row(global_epoch_name=hardcoded_params.global_session_name, first_included_epoch_name=available_non_global_names[0], last_included_epoch_name=available_non_global_names[-1], inplace=False)
            updated_epochs = Epoch(epochs_df, metadata=updated_epochs.metadata)
            existing_filename = getattr(sess.epochs, 'filename', None)
            if existing_filename is not None:
                updated_epochs.filename = existing_filename
        sess.epochs = updated_epochs
        print(f'\tdone. new epochs: \n{updated_epochs}\n')
        return updated_epochs


    @classmethod
    def build_session_basedirs_dict(cls, global_data_root_parent_path, debug_print=False) -> Dict[IdentifyingContext, Path]:
        if not isinstance(global_data_root_parent_path, Path):
            global_data_root_parent_path = Path(global_data_root_parent_path).resolve()
        fmt = cls._session_class_name
        dandiset_id = cls._dandiset_id
        relative_parent_candidates = [Path('DANDI') / 'HighDensityCrossBrain' / dandiset_id, Path('HighDensityCrossBrain') / dandiset_id]
        out: Dict[IdentifyingContext, Path] = {}
        for rel_parent in relative_parent_candidates:
            dandiset_dir = global_data_root_parent_path.joinpath(rel_parent)
            if not dandiset_dir.is_dir():
                if debug_print:
                    print(f'DANDI 001695 build_session_basedirs_dict: skip missing dandiset dir {dandiset_dir}')
                continue
            for subject_dir in sorted(dandiset_dir.glob('sub-*')):
                if not subject_dir.is_dir():
                    continue
                behavior_nwb_files = sorted(subject_dir.glob('*behavior+ecephys.nwb'))
                if len(behavior_nwb_files) < 1:
                    if debug_print:
                        print(f'DANDI 001695 build_session_basedirs_dict: skip {subject_dir} (no behavior+ecephys NWB)')
                    continue
                default_nwb = behavior_nwb_files[-1]
                animal = cls._parse_subject_from_basedir(subject_dir)
                session_name = cls._parse_session_id_from_nwb_filename(default_nwb.name)
                ctx = IdentifyingContext(format_name=fmt, animal=animal, exper_name=dandiset_id, session_name=session_name)
                out[ctx] = subject_dir.resolve()
                if debug_print:
                    print(f'DANDI 001695 build_session_basedirs_dict: registered {ctx} -> {subject_dir}')
        return out


    @classmethod
    def _get_nwb_parameters(cls, session):
        preprocessing_parameters = session.config.preprocessing_parameters
        if not hasattr(preprocessing_parameters, "nwb"):
            preprocessing_parameters.nwb = DynamicContainer(unit_location_filter="CA1", nwb_filename=None, epoch_label_mode="sleep_states", export_root=None, force_recompute_linear_position=False)
        return preprocessing_parameters.nwb


    @classmethod
    def _derive_repo_root(cls, basedir) -> Path:
        basedir = Path(basedir).resolve()
        subject_dir = basedir.parent if basedir.suffix == ".nwb" else basedir
        return subject_dir.parent.parent


    @classmethod
    def _build_file_prefix(cls, session) -> Path:
        nwb_parameters = cls._get_nwb_parameters(session)
        subject = cls._parse_subject_from_basedir(session.basepath)
        dandiset_id = cls._parse_dandiset_id_from_basedir(session.basepath)
        session_id = session.config.session_name or cls._parse_session_id_from_nwb_filename(cls._nwb_filename_from_basedir(session.basepath, nwb_filename=nwb_parameters.nwb_filename))
        export_root = Path(nwb_parameters.export_root).expanduser() if nwb_parameters.export_root is not None else cls._derive_repo_root(session.basepath) / "export" / dandiset_id
        return export_root / subject / session_id


    @classmethod
    def _get_position_spatial_series(cls, nwbf):
        if nwbf.processing is None or 'behavior' not in nwbf.processing:
            raise FileNotFoundError('NWB file has no behavior processing module; position is unavailable (ecephys-only sessions are not supported).')
        behavior = nwbf.processing['behavior']
        if 'AnimalPosition' in behavior.data_interfaces:
            animal_position = behavior.data_interfaces['AnimalPosition']
            if animal_position.spatial_series is not None and len(animal_position.spatial_series) > 0:
                if 'Position' in animal_position.spatial_series:
                    return animal_position.spatial_series['Position']
                return list(animal_position.spatial_series.values())[0]
        raise FileNotFoundError('NWB behavior module has no AnimalPosition/Position spatial series data.')


    @classmethod
    def _get_nwb_interval(cls, nwbf, interval_name: str):
        if nwbf.intervals is None:
            return None
        if hasattr(nwbf.intervals, interval_name):
            return getattr(nwbf.intervals, interval_name)
        try:
            return nwbf.intervals[interval_name]
        except (KeyError, TypeError):
            return None


    @classmethod
    def _load_paradigm_from_nwb(cls, nwbf, t0, epoch_label_mode="sleep_states"):
        if nwbf.intervals is None:
            raise ValueError('Expected NWB intervals for DANDI 001695 sessions.')
        sleep_states = cls._get_nwb_interval(nwbf, 'SleepStates')
        if sleep_states is not None:
            if epoch_label_mode != "sleep_states":
                raise ValueError(f"Unsupported epoch_label_mode: {epoch_label_mode!r}")
            epochs_df = sleep_states.to_dataframe().reset_index(drop=True)
            state_counts: Dict[str, int] = {}
            labels: List[str] = []
            behaviors: List[str] = []
            for _, row in epochs_df.iterrows():
                state_name = str(row['state'])
                state_index = state_counts.get(state_name, 0)
                state_counts[state_name] = state_index + 1
                labels.append(f"{state_name}{state_index}")
                behaviors.append(_SLEEP_STATE_BEHAVIOR_MAP.get(state_name, state_name.lower()))

            ## END for _, row in epochs_df.iterrows()...

            result_df = pd.DataFrame({'start': epochs_df['start_time'].values - t0, 'stop': epochs_df['stop_time'].values - t0, 'label': labels, 'behavior': behaviors, 'state': epochs_df['state'].astype(str).values})
            result_df['duration'] = result_df['stop'] - result_df['start']
            return Epoch(result_df)
        odor_stimulus = cls._get_nwb_interval(nwbf, 'Odor Stimulus')
        if odor_stimulus is not None:
            odor_df = odor_stimulus.to_dataframe().reset_index(drop=True)
            labels, behaviors = [], []
            odor_counts: Dict[str, int] = {}
            for row_idx, row in odor_df.iterrows():
                odor_label = str(row.get('odor', row.get('stimulus', f"odor_{row_idx}")))
                odor_index = odor_counts.get(odor_label, 0)
                odor_counts[odor_label] = odor_index + 1
                labels.append(f"odor_{odor_index}")
                behaviors.append('odor')

            ## END for row_idx, row in odor_df.iterrows()...

            result_df = pd.DataFrame({'start': odor_df['start_time'].values - t0, 'stop': odor_df['stop_time'].values - t0, 'label': labels, 'behavior': behaviors})
            if 'odor' in odor_df.columns:
                result_df['odor'] = odor_df['odor'].astype(str).values
            result_df['duration'] = result_df['stop'] - result_df['start']
            return Epoch(result_df)
        raise ValueError('Expected NWB intervals["SleepStates"] or intervals["Odor Stimulus"] for DANDI 001695 sessions.')


    @classmethod
    def _load_ripples_from_sleep_states(cls, nwbf, t0) -> Optional[Epoch]:
        sleep_states = cls._get_nwb_interval(nwbf, 'SleepStates')
        if sleep_states is None:
            return None
        sleep_df = sleep_states.to_dataframe().reset_index(drop=True)
        ripple_df = sleep_df[sleep_df['state'].astype(str) == 'Ripple'].copy()
        if len(ripple_df) == 0:
            return None
        ripple_df = ripple_df.reset_index(drop=True)
        ripple_df['label'] = [str(an_idx) for an_idx in ripple_df.index]
        result_df = pd.DataFrame({'start': ripple_df['start_time'].values - t0, 'stop': ripple_df['stop_time'].values - t0, 'label': ripple_df['label'].astype(str).values})
        result_df['duration'] = result_df['stop'] - result_df['start']
        return Epoch(result_df)


    @classmethod
    def _map_cell_type_to_neuron_type(cls, cell_type_value) -> str:
        cell_type_str = str(cell_type_value)
        return _CELL_TYPE_TO_NEURON_TYPE.get(cell_type_str, 'pyr')


    @classmethod
    def _load_neurons_from_nwb(cls, nwbf, t0, t_stop, unit_location_filter="CA1"):
        units_df = nwbf.units.to_dataframe()
        spiketrains, neuron_ids, shank_ids, neuron_types = [], [], [], []
        area_to_shank = {'CA1': 1, 'CA3': 2, 'RSC': 3}
        for unit_id, row in units_df.iterrows():
            cell_area = str(row.get('cell_area', ''))
            if unit_location_filter is not None and cell_area != unit_location_filter:
                continue
            spiketrains.append(np.asarray(row["spike_times"], dtype=float) - t0)
            neuron_ids.append(int(unit_id))
            shank_ids.append(area_to_shank.get(cell_area, 0))
            neuron_types.append(cls._map_cell_type_to_neuron_type(row.get('cell_type', 'Pyramidal Cell')))

        ## END for unit_id, row in units_df.iterrows()...

        if not spiketrains:
            raise ValueError(f"No units matched cell_area filter {unit_location_filter!r}")
        return Neurons(np.array(spiketrains, dtype=object), t_stop=t_stop, t_start=0.0, neuron_ids=neuron_ids, shank_ids=np.array(shank_ids, dtype=np.int64), neuron_type=np.array(neuron_types))


    @classmethod
    def _load_session_from_nwb(cls, session):
        from pynwb import NWBHDF5IO

        nwb_parameters = cls._get_nwb_parameters(session)
        nwb_path = cls.find_nwb_file(session.basepath, nwb_filename=nwb_parameters.nwb_filename)
        with NWBHDF5IO(str(nwb_path), mode="r") as io:
            nwbf = io.read()
            timestamps = cls._load_position_timestamps(nwbf)
            t0 = cls._get_ecephys_t0(nwbf)
            session.config.absolute_start_timestamp = t0
            session.position = cls._load_position_from_nwb(nwbf, timestamps=timestamps, t0=t0)
            session.neurons = cls._load_neurons_from_nwb(nwbf, t0=t0, t_stop=float(session.position.time[-1]), unit_location_filter=nwb_parameters.unit_location_filter)
            session.paradigm = cls._load_paradigm_from_nwb(nwbf, t0=t0, epoch_label_mode=nwb_parameters.epoch_label_mode)
            session.ripple = cls._load_ripples_from_sleep_states(nwbf, t0=t0)
            session.recinfo.source_file = nwb_path
        if len(session.position.time) > 1:
            session.config.position_sampling_rate_Hz = float(1.0 / np.nanmean(np.diff(session.position.time)))
        return session


    @classmethod
    def _resolve_track_definition_for_session(cls, session) -> Optional[object]:
        return None


    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_includelist=None, filter_name_suffix=None, include_global_epoch=False):
        from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters

        if epoch_name_includelist is None:
            epoch_name_includelist = cls._get_decoder_building_epoch_labels(sess)
        return build_custom_epochs_filters(sess, epoch_name_includelist=epoch_name_includelist, filter_name_suffix=filter_name_suffix)


    @classmethod
    def build_filters_run_epochs(cls, sess, filter_name_suffix=None):
        from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters

        task_epoch_names = cls._get_activity_epoch_labels(sess)
        return build_custom_epochs_filters(sess, epoch_name_includelist=task_epoch_names, filter_name_suffix=filter_name_suffix)


    @classmethod
    def _estimate_and_enrich_laps_from_preprocessing_config(cls, sess, should_plot_laps_2d=False):
        from neuropy.analyses.laps import estimate_session_laps
        from neuropy.core.epoch import ensure_dataframe
        from neuropy.core.laps import LapsAccessor
        from neuropy.utils.mixins.indexing_helpers import get_dict_subset

        cls.ensure_preprocessing_epoch_estimation_parameters(sess)
        lap_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.laps
        custom_lap_estimation_fn = lap_estimation_parameters.get('custom_lap_estimation_fn', None)
        if custom_lap_estimation_fn is not None:
            custom_lap_estimation_fn(sess)
        else:
            estimate_session_laps(sess, should_plot_laps_2d=should_plot_laps_2d, **get_dict_subset(lap_estimation_parameters.to_dict(), subset_excludelist=['custom_lap_estimation_fn', 'reward_zones', 'use_direction_dependent_laps', 'should_backup_extant_laps_obj', 'N', 'linearization_method']))
        active_task_epochs_df = ensure_dataframe(deepcopy(sess.epochs.to_dataframe()))
        laps_df: pd.DataFrame = sess.laps.to_dataframe()
        laps_df = laps_df.epochs.adding_maze_id_if_needed(active_maze_epochs_df=active_task_epochs_df, replace_existing=True)
        label_to_task_id = {str(a_label): task_idx for task_idx, a_label in enumerate(active_task_epochs_df['label'].astype(str).tolist())}
        laps_df['maze_id'] = laps_df['maze_id'].map(lambda a_value: label_to_task_id.get(str(a_value), -1)).astype(int)
        sess.laps._df = laps_df
        try:
            LapsAccessor.non_kdiba_laps_determine_directions(sess=sess)
        except Exception as e:
            warnings.warn(f'Could not determine lap directions for DANDI 001695 session {sess.get_context()}: {e}')
        return sess


    @classmethod
    def POSTLOAD_estimate_laps_and_replays(cls, sess):
        from neuropy.core.epoch import Epoch

        print('POSTLOAD_estimate_laps_and_replays()...')
        cls.ensure_preprocessing_epoch_estimation_parameters(sess)
        cls.session_fixup_epochs(sess, enable_global_epoch=True)
        cls._estimate_and_enrich_laps_from_preprocessing_config(sess, should_plot_laps_2d=False)
        non_running_periods = Epoch.from_PortionInterval(sess.laps.as_epoch_obj().to_PortionInterval().complement())
        PBE_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.PBEs
        assert PBE_estimation_parameters is not None
        PBE_estimation_parameters.require_intersecting_epoch = non_running_periods
        new_pbe_epochs = sess.compute_pbe_epochs(sess, active_parameters=PBE_estimation_parameters)
        sess.pbe = new_pbe_epochs
        sess.compute_spikes_PBEs()
        replay_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.replays
        assert replay_estimation_parameters is not None
        replay_estimation_parameters.require_intersecting_epoch = non_running_periods
        replay_estimation_parameters.min_inclusion_fr_active_thresh = 1.0
        replay_estimation_parameters.min_num_unique_aclu_inclusions = 5
        try:
            sess.replace_session_replays_with_estimates(**replay_estimation_parameters)
        except Exception as e:
            warnings.warn(f'Could not estimate replays for DANDI 001695 session {sess.get_context()}: {e}')
        new_non_pbe_epochs = sess.compute_non_PBE_epochs(sess, active_parameters=PBE_estimation_parameters, save_on_compute=True)
        sess.non_pbe = new_non_pbe_epochs
        return sess
