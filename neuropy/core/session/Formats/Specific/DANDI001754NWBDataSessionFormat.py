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


_SESSION_TYPE_BEHAVIOR_MAP = {'ES': 'escher', 'MC': 'magic_carpet', 'BL': 'baseline'}
_TASK_ACTIVITY_BEHAVIORS = frozenset({'escher', 'magic_carpet'})


class DANDI001754NWBDataSessionFormatRegisteredClass(NWBDataSessionFormatRegisteredClass):
    """Loads DANDI 001754 (Neurolab / ThreeDimSpatial) NWB sessions into NeuroPy.

    Target layout:
        ThreeDimSpatial/001754/sub-Rat1/sub-Rat1_ses-19980425T124500_behavior+ecephys.nwb

    Usage:
        from neuropy.core.session.Formats.Specific.DANDI001754NWBDataSessionFormat import DANDI001754NWBDataSessionFormatRegisteredClass

        sess = DANDI001754NWBDataSessionFormatRegisteredClass.get_session(
            basedir=Path(r'H:\\Data\\DANDI\\ThreeDimSpatial\\001754\\sub-Rat1'),
            override_parameters_flat_keypaths_dict={'nwb.nwb_filename': 'sub-Rat1_ses-19980425T124500_behavior+ecephys.nwb'},  # optional; defaults to first *.nwb under basedir
        )
    """

    _session_class_name = "dandi_nwb_001754"
    _session_default_relative_basedir = "ThreeDimSpatial/001754/sub-Rat1"
    _session_default_basedir = "/media/halechr/BETAMAX1/Data/DANDI/ThreeDimSpatial/001754/sub-Rat1"
    _session_basepath_to_context_parsing_keys = ["format_name", "animal", "exper_name", "session_name"]
    _dandiset_id = "001754"

    @classmethod
    def build_default_preprocessing_parameters(cls, **kwargs):
        override_parameters_flat_keypaths_dict = kwargs.pop("override_parameters_flat_keypaths_dict", {}) or {}
        preprocessing_parameters = super().build_default_preprocessing_parameters(override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict, **kwargs)
        preprocessing_parameters.epoch_estimation_parameters.laps.use_direction_dependent_laps = False
        override_parameters_nested_dicts = KeypathsAccessibleMixin.keypath_dict_to_nested_dict(override_parameters_flat_keypaths_dict)
        nwb_overrides = override_parameters_nested_dicts.get("nwb", {}) | {k: v for k, v in override_parameters_flat_keypaths_dict.items() if k in {"unit_location_filter", "nwb_filename", "epoch_label_mode", "export_root", "force_recompute_linear_position"}}
        preprocessing_parameters.nwb = DynamicContainer(unit_location_filter="CA1", nwb_filename=None, epoch_label_mode="session_type", export_root=None, force_recompute_linear_position=False).override(nwb_overrides)
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
            override_filename = override_parameters_flat_keypaths_dict.get("preprocessing.nwb.nwb_filename", None) ## try the preprocessing variable first
            if override_filename is not None:
                return str(override_filename)

            override_filename = override_parameters_flat_keypaths_dict.get("nwb.nwb_filename", None)
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
        default_grid_bin_bounds = (((0.0, 255.0), (0.0, 255.0)))
        default_lap_params = dict(reward_zones=None, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration=2.5, minimum_run_speed=5.0, merging_adjacent_max_separation_sec=6.0)
        default_decoder_names = ['ES0', 'MC0', 'task_GLOBAL']
        the_dict: Dict[IdentifyingContext, HardcodedProcessingParameters] = {
            IdentifyingContext(format_name='dandi_nwb_001754', animal='Rat1', exper_name='001754'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='task_GLOBAL',
                non_global_activity_session_names=['ES0', 'MC0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name='dandi_nwb_001754', exper_name='001754'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='task_GLOBAL',
                non_global_activity_session_names=['ES0', 'MC0'],
                grid_bin_bounds=default_grid_bin_bounds,
                lap_estimation_parameters=default_lap_params,
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name='dandi_nwb_001754'): HardcodedProcessingParameters(
                decoder_building_session_names=default_decoder_names,
                global_session_name='task_GLOBAL',
                non_global_activity_session_names=['ES0', 'MC0'],
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
    def _get_activity_epoch_labels(cls, sess) -> list:
        if sess is None or sess.epochs is None:
            return []
        epochs_df = sess.epochs.to_dataframe()
        if 'behavior' in epochs_df.columns:
            return [str(a_label) for a_label in epochs_df.loc[epochs_df['behavior'].astype(str).isin(_TASK_ACTIVITY_BEHAVIORS), 'label'].tolist()]
        return [str(a_label) for a_label in epochs_df['label'].astype(str).tolist() if str(a_label).startswith(('ES', 'MC')) and str(a_label) != 'task_GLOBAL']

    @classmethod
    def _get_decoder_building_epoch_labels(cls, sess) -> list:
        activity_labels = cls._get_activity_epoch_labels(sess)
        if len(activity_labels) == 0:
            return []
        return activity_labels + ['task_GLOBAL']

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
            print('fixing up DANDI 001754 session computation epochs...')
            sess.epochs_bak = deepcopy(updated_epochs)
        else:
            print('WARN: already fixedup session epochs.')
            if override_extant:
                if cls._epoch_labels_include(sess.epochs_bak, required_epoch_names):
                    print('\trestoring backed up epochs:')
                    sess.epochs = deepcopy(sess.epochs_bak)
                    updated_epochs = deepcopy(sess.epochs)
                else:
                    print('\tdiscarding incompatible epochs_bak because it does not contain the expected task labels.')
                    delattr(sess, 'epochs_bak')
                    sess.epochs_bak = deepcopy(updated_epochs)
        epochs_df = updated_epochs.to_dataframe()
        if enable_global_epoch and hardcoded_params.global_session_name not in epochs_df['label'].astype(str).tolist():
            available_non_global_names = [a_name for a_name in required_epoch_names if a_name in epochs_df['label'].astype(str).tolist()]
            if len(available_non_global_names) < 1:
                raise ValueError(f"Could not add {hardcoded_params.global_session_name!r}; none of the expected task epoch labels were present. expected={required_epoch_names}, actual={epochs_df['label'].astype(str).tolist()}")
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
        relative_parent_candidates = [Path('DANDI') / 'ThreeDimSpatial' / dandiset_id, Path('ThreeDimSpatial') / dandiset_id]
        out: Dict[IdentifyingContext, Path] = {}
        for rel_parent in relative_parent_candidates:
            dandiset_dir = global_data_root_parent_path.joinpath(rel_parent)
            if not dandiset_dir.is_dir():
                if debug_print:
                    print(f'DANDI 001754 build_session_basedirs_dict: skip missing dandiset dir {dandiset_dir}')
                continue
            for subject_dir in sorted(dandiset_dir.glob('sub-*')):
                if not subject_dir.is_dir():
                    continue
                behavior_nwb_files = sorted(subject_dir.glob('*behavior+ecephys.nwb'))
                if len(behavior_nwb_files) < 1:
                    if debug_print:
                        print(f'DANDI 001754 build_session_basedirs_dict: skip {subject_dir} (no behavior+ecephys NWB)')
                    continue
                default_nwb = behavior_nwb_files[-1]
                animal = cls._parse_subject_from_basedir(subject_dir)
                session_name = cls._parse_session_id_from_nwb_filename(default_nwb.name)
                ctx = IdentifyingContext(format_name=fmt, animal=animal, exper_name=dandiset_id, session_name=session_name)
                out[ctx] = subject_dir.resolve()
                if debug_print:
                    print(f'DANDI 001754 build_session_basedirs_dict: registered {ctx} -> {subject_dir}')
        return out

    @classmethod
    def _get_nwb_parameters(cls, session):
        preprocessing_parameters = session.config.preprocessing_parameters
        if not hasattr(preprocessing_parameters, "nwb"):
            preprocessing_parameters.nwb = DynamicContainer(unit_location_filter="CA1", nwb_filename=None, epoch_label_mode="session_type", export_root=None, force_recompute_linear_position=False)
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
        if 'position' not in behavior.data_interfaces:
            raise FileNotFoundError('NWB behavior module has no position interface.')
        position = behavior.data_interfaces['position']
        if position.spatial_series is None or len(position.spatial_series) < 1:
            raise FileNotFoundError('NWB position interface has no spatial_series data.')
        if 'spatial_series' in position.spatial_series:
            return position.spatial_series['spatial_series']
        return list(position.spatial_series.values())[0]

    @classmethod
    def _load_paradigm_from_nwb(cls, nwbf, t0, epoch_label_mode="session_type"):
        if nwbf.intervals is None or 'epochs' not in nwbf.intervals:
            raise ValueError('Expected NWB intervals["epochs"] for DANDI 001754 sessions.')
        if epoch_label_mode != "session_type":
            raise ValueError(f"Unsupported epoch_label_mode: {epoch_label_mode!r}")
        epochs_table = nwbf.intervals['epochs']
        epochs_df = epochs_table.to_dataframe().reset_index(drop=True)
        type_counts: Dict[str, int] = {}
        labels: List[str] = []
        behaviors: List[str] = []
        for _, row in epochs_df.iterrows():
            session_type = str(row['session_type'])
            type_index = type_counts.get(session_type, 0)
            type_counts[session_type] = type_index + 1
            labels.append(f"{session_type}{type_index}")
            behaviors.append(_SESSION_TYPE_BEHAVIOR_MAP.get(session_type, session_type.lower()))

        ## END for _, row in epochs_df.iterrows()...

        result_df = pd.DataFrame({'start': epochs_df['start_time'].values - t0, 'stop': epochs_df['stop_time'].values - t0, 'label': labels, 'behavior': behaviors, 'session_type': epochs_df['session_type'].astype(str).values})
        if 'session_type_description' in epochs_df.columns:
            result_df['session_type_description'] = epochs_df['session_type_description'].astype(str).values
        result_df['duration'] = result_df['stop'] - result_df['start']
        return Epoch(result_df)

    @classmethod
    def _unit_matches_location_filter(cls, location: str, unit_location_filter: Optional[str]) -> bool:
        if unit_location_filter is None:
            return True
        return unit_location_filter.lower() in str(location).lower()

    @classmethod
    def _load_neurons_from_nwb(cls, nwbf, t0, t_stop, unit_location_filter="CA1"):
        units_df = nwbf.units.to_dataframe()
        spiketrains, neuron_ids, shank_ids = [], [], []
        for unit_id, row in units_df.iterrows():
            location = cls._unit_electrode_field(row["electrodes"], "location")
            if not cls._unit_matches_location_filter(location, unit_location_filter):
                continue
            spiketrains.append(np.asarray(row["spike_times"], dtype=float) - t0)
            neuron_ids.append(int(unit_id))
            group_name = cls._unit_electrode_field(row["electrodes"], "group_name")
            shank_ids.append(cls._parse_shank_from_group_name(group_name))

        ## END for unit_id, row in units_df.iterrows()...

        if not spiketrains:
            raise ValueError(f"No units matched location filter {unit_location_filter!r}")
        neuron_type = np.array(["pyr"] * len(neuron_ids))
        return Neurons(np.array(spiketrains, dtype=object), t_stop=t_stop, t_start=0.0, neuron_ids=neuron_ids, shank_ids=np.array(shank_ids, dtype=np.int64), neuron_type=neuron_type)

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
            warnings.warn(f'Could not determine lap directions for DANDI 001754 session {sess.get_context()}: {e}')
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
            warnings.warn(f'Could not estimate replays for DANDI 001754 session {sess.get_context()}: {e}')
        new_non_pbe_epochs = sess.compute_non_PBE_epochs(sess, active_parameters=PBE_estimation_parameters, save_on_compute=True)
        sess.non_pbe = new_non_pbe_epochs
        return sess
