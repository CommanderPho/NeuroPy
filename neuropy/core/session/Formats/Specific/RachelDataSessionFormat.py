from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec

# For specific load functions:
from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains, Shank, Probe, ProbeGroup
from neuropy.io import OptitrackIO, PhyIO
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta
from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch, EpochsAccessor
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters # used particularly to build Bapun-style filters

from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.npy_pickle_compat import load_npy_pickled_item

## Pho's Custom Libraries:
# from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path
from neuropy.core.session.Formats.BaseDataSessionFormats import HardcodedProcessingParameters


def find_data_files(project_path, glob_str: str = f"**/*.npy", exclude_dirs=[]):
    """ Find all files matching the glob in the project directory and its subdirectories
    Usage:
        glob_str: str = f"**/*.npy"
        included_files = find_data_files(basedir, glob_str=glob_str)
        included_files

    """
    if not isinstance(project_path, Path):
        project_path = Path(project_path)
    found_files = project_path.glob(glob_str)
    found_files = [file_path for file_path in found_files] # to list

    excluded_files = []
    if exclude_dirs is not None:
        # Find all .py files in the project directory and its subdirectories, excluding the 'my_exclude_dir' directory
        exclude_paths = [project_path.joinpath(a_dir) for a_dir in exclude_dirs]
        for an_exclude_path in exclude_paths:
            excluded_files.extend([file_path for file_path in an_exclude_path.glob(glob_str)])

    included_files = [x for x in found_files if x not in excluded_files]
    return included_files


class RachelDataSessionFormat(BapunDataSessionFormatRegisteredClass):
    """

    # Example Filesystem Hierarchy:
    📦Rachel
    ┣ 📂merged_M1_20211123_raw_phy
    ┃ ┣ 📜whitening_mat_inv.npy
    ┃ ┣ 📜spike_templates.npy
    ┃ ┣ 📜tempClustering.klg.3
    ┃ ┣ 📜cluster_info.tsv
    ┃ ┣ 📜channel_shanks.npy
    ┃ ┣ 📜merged_M1_20211123_raw.eeg
    ┃ ┣ 📜cluster_purity.tsv
    ┃ ┣ 📜spike_clusters.npy
    ┃ ┣ 📜similar_templates.npy
    ┃ ┣ 📜pc_feature_ind.npy
    ┃ ┣ 📜phy.log
    ┃ ┣ 📜merged_M1_20211123_raw.paradigm.npy
    ┃ ┣ 📜cluster_group.tsv
    ┃ ┣ 📜spike_times.npy
    ┃ ┣ 📜tempClustering.fet.3
    ┃ ┣ 📜tempClustering.clu.3
    ┃ ┣ 📜merged_M1_20211123_raw.xml
    ┃ ┣ 📜whitening_mat.npy
    ┃ ┣ 📜templates.npy
    ┃ ┣ 📜params.py
    ┃ ┣ 📜channel_map.npy
    ┃ ┣ 📜pc_features.npy
    ┃ ┣ 📜channel_positions.npy
    ┃ ┣ 📜ttl_check.ipynb
    ┃ ┣ 📜amplitudes.npy
    ┃ ┣ 📜merged_M1_20211123_raw.position.npy
    ┃ ┣ 📜tempClustering.temp.clu.3
    ┃ ┣ 📜merged_M1_20211123_raw.neurons.npy
    ┃ ┣ 📜merged_M1_20211123_raw.probegroup.npy
    ┃ ┗ 📜cluster_q.tsv
    
    
    By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file:
        basedir: Path(r'R:\data\Rachel\merged_M1_20211123_raw_phy')
        session_name: 'merged_M1_20211123_raw'
    
    From here, a list of known files to load from is determined:
        
    Usage:
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
        from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat

        _test_session = RachelDataSessionFormat.build_session(Path(r'R:\data\Rachel\merged_M1_20211123_raw_phy'))
        _test_session, loaded_file_record_list = RachelDataSessionFormat.load_session(_test_session)
        _test_session
        
    """
    _session_class_name = 'rachel'
    _session_default_relative_basedir = r'data/Rachel/merged_M1_20211123_raw_phy'
    _session_default_basedir = r'R:\data\Rachel\merged_M1_20211123_raw_phy' # Windows
    # _session_default_basedir = '/run/media/halechr/MoverNew/data/Rachel/merged_M1_20211123_raw_phy' # LINUX
    # _session_basepath_to_context_parsing_keys = ['format_name', 'session_name']
    _session_basepath_to_context_parsing_keys = ['format_name','animal', 'session_name'] # 2025-02-27 11:57 



    _time_variable_name = 't_seconds' # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data

    @classmethod
    def _get_session_specific_parameters(cls, session_context: IdentifyingContext) -> HardcodedProcessingParameters:
        """ session-specific type parameters 
         
        #TODO 2025-09-20 19:26: - [ ] Is this redudndant with preprocessing parameters?
        """
        # ((-109.38643591109636, 106.07262807889244), (-15.03109955262099, 19.85259224492544))
        grid_bin_bounds = ((-110.0, 110.0), (-20.0, 20.0)) ## #TODO 2025-10-01 15:44: - [ ] Specifically Cho 2024-11-18
        
        
        the_dict: Dict[IdentifyingContext, HardcodedProcessingParameters]  = { #  
            IdentifyingContext(format_name= 'rachel', animal= 'Cho', region= 'CA1', session_name= '2024-11-17'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=grid_bin_bounds,
            ),
            IdentifyingContext(format_name= 'rachel', animal= 'Cho', region= 'CA1', session_name= '2024-11-18'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=grid_bin_bounds,
            ),
            ## Fallback defaults:
            IdentifyingContext(format_name= 'rachel'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
            ),									
        }

        best_match = IdentifyingContext.matching(the_dict, criteria=session_context.get_subset(subset_includelist=cls._session_basepath_to_context_parsing_keys).to_dict())
        return list(best_match.values())[0] ## return the first match


   
    @classmethod
    def get_session_name(cls, basedir):
        """ returns the session_name for this basedir, which determines the files to load. """
        # Find the only .xml file to obtain the session name
        return DataSessionFormatBaseRegisteredClass.find_session_name_from_sole_xml_file(basedir) # 'merged_M1_20211123_raw'

    @classmethod
    def get_session_spec(cls, session_name):
        return SessionFolderSpec(required_files=[SessionFileSpec('{}.xml', session_name, 'The primary .xml configuration file', cls._load_xml_file),
                                           SessionFileSpec('{}.neurons.npy', session_name, 'The numpy data file containing information about neural activity.', cls._load_neurons_file),
                                        #    SessionFileSpec('{}.probegroup.npy', session_name, 'The numpy data file containing information about the spatial layout of recording probes', cls._load_probegroup_file),
                                           SessionFileSpec('{}.position.npy', session_name, 'The numpy data file containing the recorded animal positions (as generated by optitrack) over time.', cls._load_position_file),
                                           SessionFileSpec('{}.paradigm.npy', session_name, 'The numpy data file containing the recording epochs. Each epoch is defined as a: (label:str, t_start: float (in seconds), t_end: float (in seconds))', cls._load_paradigm_file)]
                    )
    ### Specific Load Functions used in the session_spec
    @classmethod
    def _load_neurons_file(cls, filepath, session): # .neurons
        session.neurons = Neurons.from_file(filepath)
        return session
    @classmethod
    def _load_probegroup_file(cls, filepath, session): # .probegroup
        session.probegroup = ProbeGroup.from_file(filepath)
        return session
    @classmethod
    def _load_position_file(cls, filepath, session): # .position
        session.position = Position.from_file(filepath)
        return session
    @classmethod
    def _load_paradigm_file(cls, filepath, session): # .paradigm
        session.paradigm = Epoch.from_file(filepath)  # "epoch" field of file
        return session
    
    @classmethod
    def parse_session_basepath_to_context(cls, basedir) -> IdentifyingContext:
        """ parses the session's path to determine its proper context. Depends on the data_type.
        finds global_data_root

        USED: Called only in `cls.build_session(.)`

        KDIBA: 'W:/Data/KDIBA/gor01/one/2006-6-09_3-23-37' | context_keys = ['format_name','animal','exper_name', 'session_name']
        HIRO: 'W:/Data/Hiro/RoyMaze1' | context_keys = ['format_name', 'session_name']
            # Additional parsing needed: W:\Data\Hiro\RoyMaze2

        BAPUN: 'W:/Data/Bapun/RatS/Day5TwoNovel' | context_keys = ['format_name','animal', 'session_name']
        RACHEL: 'W:/Data/Rachel/merged_M1_20211123_raw_phy' | context_keys = ['format_name', 'session_name']

        """
        # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
        basedir = Path(basedir) # basedir WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-07_11-26-53')
        dir_parts = basedir.parts # ('W:\\', 'Data', 'KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')
        # Finds the index of the 'Data' or 'data' (global_data_root) part of the path to include only what's after that.

        # 'umms-kdiba' in '/nfs/turbo/umms-kdiba/Data' is basically a Data folder
        if basedir.resolve().is_relative_to(Path('/nfs/turbo/umms-kdiba').resolve()):
            _parent_path = Path('/nfs/turbo/umms-kdiba').resolve()
            # relative_basedir = basedir.resolve().relative_to(_parent_path)
            # dir_parts = basedir.parts
            data_index = len(_parent_path.parts) - 1 # -1 for index
        else:
            try:        
                data_index = tuple(map(str.casefold, dir_parts)).index('DATA'.casefold()) # .casefold is equivalent to .lower, but works for unicode characters
            except ValueError:
                # Enables looking for 'FASTDATA' in the path when DATA is not found
                data_index = tuple(map(str.casefold, dir_parts)).index('FASTDATA'.casefold()) # .casefold is equivalent to .lower, but works for unicode characters
            except Exception:
                raise # unhandled exception

        post_data_root_dir_parts = dir_parts[data_index+1:] # ('KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')
        num_parts = len(post_data_root_dir_parts)
        context_keys = cls.get_session_basepath_to_context_parsing_keys()
        if (len(context_keys) != num_parts):
            ## for Bapun sessions
            try:        
                data_index = tuple(map(str.casefold, dir_parts)).index('DATA'.casefold()) # .casefold is equivalent to .lower, but works for unicode characters
            except ValueError:
                # Enables looking for 'FASTDATA' in the path when DATA is not found
                data_index = tuple(map(str.casefold, dir_parts)).index('FASTDATA'.casefold()) # .casefold is equivalent to .lower, but works for unicode characters
            except Exception:
                raise # unhandled exception
            post_data_root_dir_parts = dir_parts[data_index+1:] # ('KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')
            num_parts = len(post_data_root_dir_parts)
            context_keys = cls.get_session_basepath_to_context_parsing_keys()
                
        # ('Rachel', 'Petunia', 'Recordings', 'CA1', '2024-12-09')
        if ('rachel' == post_data_root_dir_parts[0].lower()) and (num_parts == 5) and ('recordings' == post_data_root_dir_parts[2].lower()):
            # 2025-09-11 Format
            context_kwargs_dict = {'format_name': post_data_root_dir_parts[0], 
                'animal': post_data_root_dir_parts[1],
                # post_data_root_dir_parts[2] == 'Recordings' and is skipped 
                'region': post_data_root_dir_parts[3], # ('CA1', 'DG', 'Cortex')
                'session_name': post_data_root_dir_parts[-1],
            }
            
        else:
            ## Fallback to older/default way of parsing:
            assert len(context_keys) == num_parts, f"context_keys: {context_keys}, post_data_root_dir_parts: {post_data_root_dir_parts}"
            context_kwargs_dict = dict(zip(context_keys, post_data_root_dir_parts))
        ## END if
                

        curr_sess_ctx = IdentifyingContext(**context_kwargs_dict)
        
        # want to replace the 'format_name' with the one known for this session (e.g. 'KDIBA' vs. 'kdiba')
        format_name = cls.get_session_format_name() 
        curr_sess_ctx.format_name = format_name
        return curr_sess_ctx # IdentifyingContext<('KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')>
        



    @classmethod
    def session_fixup_epochs(cls, session_epochs: Epoch, enable_global_epoch: bool=True, **kwargs) -> Epoch:
        """ fixes up the loaded epochs
        has two conflicting (overlapping) epochs:
        "maze"
        "sprinkle"
        
        
                start   stop     label  duration
        0      0   7407       pre      7407
        1   7423  11483      maze      4060
        3  10186  11483  sprinkle      1297
        2  11497  25987      post     14490

        [4 rows x 4 columns]

        --- we want to compute the disjoint set 'roam', 'sprinkle'
        
        
        Usage:
        
        
        bapun_epochs: Epoch = deepcopy(curr_active_pipeline.sess.epochs)
        bapun_epochs = BapunDataSessionFormatRegisteredClass._bapun_session_fixup_epochs_to_be_non_overlapping(bapun_epochs=bapun_epochs)
        
        """
        bapun_epochs_df: pd.DataFrame = session_epochs.to_dataframe()
        if (len(bapun_epochs_df) == 4) and ('roam' not in bapun_epochs_df['label'].to_list()):
            ## Applicable to Day4OpenField only: add the 'roam' row if it doesn't already exist
            bapun_epochs_arr = bapun_epochs_df.to_numpy()
            new_roam_row = [bapun_epochs_arr[1, 0], (bapun_epochs_arr[2, 0]-1), 'roam', 0.0] # ['start', 'stop', 'label', 'duration']
            
            # bapun_epochs_arr[1, 1] = bapun_epochs_arr[2, 0] - 1 # overwrite the "maze" portion's end-time
            fixed_bapun_epochs_arr = deepcopy(bapun_epochs_arr)
            fixed_bapun_epochs_arr[1, :] = new_roam_row ## replace the entire 2nd row (the 'maze' row) with the new 'roam' row.
            # build final epochs_df from fixed_bapun_epochs_arr
            bapun_epochs_df: pd.DataFrame = pd.DataFrame(fixed_bapun_epochs_arr, columns=['start', 'stop', 'label', 'duration'])
            bapun_epochs_df[['start', 'stop', 'duration']] = bapun_epochs_df[['start', 'stop', 'duration']].astype(float)
            bapun_epochs_df['duration'] = bapun_epochs_df['stop'] - bapun_epochs_df['start']

        if enable_global_epoch:
            # maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            bapun_epochs_df = bapun_epochs_df.epochs.adding_global_epoch_row()

        # cls.make_last_epoch_finite_from_dataseries(
        return Epoch(bapun_epochs_df, metadata=session_epochs.metadata)



    @classmethod
    def build_filters_any_epochs(cls, sess, filter_name_suffix=None):
        return build_custom_epochs_filters(sess, filter_name_suffix=filter_name_suffix)
    
    # Any epoch on the maze, not limited to pyramidal cells, etc
    @classmethod
    def build_filters_any_maze_epochs(cls, sess, filter_name_suffix=None, override_epoch_name_includelist: Optional[List[str]]=None):
        all_epochs_names: List[str] = sess.epochs.to_dataframe()['label'].to_list()
        if override_epoch_name_includelist is not None:
            ## use `override_epoch_name_includelist`
            maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.lower() in override_epoch_name_includelist, names))
            assert (len(maze_only_name_filter_fn(all_epochs_names)) > 0), f"could not match any of the epochs in override_epoch_name_includelist: {override_epoch_name_includelist} session name formats. Need an override!\n\tall_epochs_names: {all_epochs_names}"

        else:
            maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.lower().startswith('maze'), names))
            if len(maze_only_name_filter_fn(all_epochs_names)) == 0:
                ## no epochs starting with "maze", assume it's the other type of session
                maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.lower() in ['maze'], names))
                assert (len(maze_only_name_filter_fn(all_epochs_names)) > 0), f"could not match either 'maze*' or ['maze'] session name formats. Need an override!\n\tall_epochs_names: {all_epochs_names}"

        maze_only_filters = build_custom_epochs_filters(sess, epoch_name_includelist=maze_only_name_filter_fn, filter_name_suffix=filter_name_suffix)
        return maze_only_filters


    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_includelist=None, filter_name_suffix=None, include_global_epoch=False):
        ## TODO: currently hard-coded
        # active_session_filter_configurations = cls.build_filters_any_epochs(sess)
        active_session_filter_configurations = cls.build_filters_any_maze_epochs(sess, filter_name_suffix=filter_name_suffix, override_epoch_name_includelist=epoch_name_includelist)
        return active_session_filter_configurations
    
        



    #######################################################
    ## Rachel Nupy Format Only Methods:
    @classmethod
    def _rachel_add_missing_spikes_df_columns(cls, spikes_df, neurons_obj):
        spikes_df, neurons_obj._reverse_cellID_index_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        spikes_df['t'] = spikes_df['t_seconds'] # add the 't' column required for visualization
    
    
    ## Main load function:
    @classmethod
    def load_session(cls, session, debug_print=False):
        session, loaded_file_record_list = DataSessionFormatBaseRegisteredClass.load_session(session, debug_print=debug_print) # call the super class load_session(...) to load the common things (.recinfo, .filePrefix, .eegfile, .datfile)
        remaining_required_filespecs = {k: v for k, v in session.config.resolved_required_filespecs_dict.items() if k not in loaded_file_record_list}
        if debug_print:
            print(f'remaining_required_filespecs: {remaining_required_filespecs}')
        
        for file_path, file_spec in remaining_required_filespecs.items():
            session = file_spec.session_load_callback(file_path, session)
            loaded_file_record_list.append(file_path)
        
        # ['.neurons.npy','.probegroup.npy','.position.npy','.paradigm.npy']
        # session = DataSessionLoader.__default_compute_bapun_flattened_spikes(session)
        
        # # Load or compute linear positions if needed:        
        # if (not session.position.has_linear_pos):
        #     # compute linear positions:
        #     print('computing linear positions for all active epochs for session...')
        #     # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
        #     session.position.linear_pos = np.full_like(session.position.time, np.nan)
        #     acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession._perform_compute_session_linearized_position(session, 'maze')
        #     session.position.linear_pos[acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
        #     session.position.filename = session.filePrefix.with_suffix(".position.npy")
        #     # print('Saving updated position results to {}...'.format(session.position.filename))
        #     with ProgressMessagePrinter(session.position.filename, action='Saving', contents_description='updated position results'):
        #         session.position.save()
        #     # print('done.\n')
        # else:
        #     print('linearized position loaded from file.')

        ## Load or compute flattened spikes since this format of data has the spikes ordered only by cell_id:
        ## flattened.spikes:
        # active_file_suffix = '.flattened.spikes.npy'
        active_file_suffix = '.flattened.spikes.npy'
        found_datafile = FlattenedSpiketrains.from_file(session.filePrefix.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.flattened_spiketrains = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session = cls._default_compute_flattened_spikes(session, spike_timestamp_column_name=cls._time_variable_name) # sets session.flattened_spiketrains
            
            ## Testing: Fixing spike positions
            spikes_df = session.spikes_df
            session, spikes_df = cls._default_compute_spike_interpolated_positions_if_needed(session, spikes_df, time_variable_name=cls._time_variable_name)
            cls._rachel_add_missing_spikes_df_columns(spikes_df, session.neurons) # add the missing columns to the dataframe             
            
            
            session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix) # '.flattened.spikes.npy'
            print('\t Saving computed flattened spiketrains results to {}...'.format(session.flattened_spiketrains.filename), end='')
            session.flattened_spiketrains.save()
            print('\t done.\n')
        
        # Common Extended properties:
        session = cls._default_extended_postload(session.filePrefix, session)
        
        return session, loaded_file_record_list
    
    
    @classmethod
    def neurontype_classCutoffMap(cls) -> dict:
        """ For each qclu value in 0-10, return a str in ['pyr','cont','intr']
            Kamran 2023-07-18:
                cluq=[1,2,4,9] all passed.
                3 were noisy
                [6,7]: double fields.
                5: interneurons

        ## Post 2023-07-18:
            for i in np.arange(10):
                _out_map[i] = "cont" # initialize all to 'contaminated'/noisy
            for i in [1,2,4,6,7,9]:
                _out_map[i] = 'pyr' # pyramidal
            _out_map[5] = "intr" # interneurons

        ## Post 2023-07-31 - Excluding "double fields" qclues on Kamran's request
        ## 2023-12-07 12:22: - [ ] Kamran wants me to exclude [6, 7]
        ## 2023-12-07 20:59: - [ ] Updated to only include [1,2,4,9] as pyramidal
        ## 2024-10-25 - Including [1,2,4,6,7,9]


        
        ## Rachel
        

        """
        _out_map = dict() # start with an empty dict
        for i in np.arange(10):
            _out_map[i] = "cont" # initialize all to 'contaminated'/noisy
        for i in [1,2,3]:
            _out_map[i] = 'pyr' # pyramidal

        for i in [4, 6]:
            _out_map[i] = 'intr' # interneurons

        return _out_map
    

    

    @classmethod
    def make_last_epoch_finite_from_dataseries(cls, sess, epochs_df: Optional[pd.DataFrame]=None) -> Epoch:
        max_spike_t: float = sess.spikes_df['t'].max()
        max_pos_t: float = sess.position.to_dataframe()['t'].max()
        max_pos_t
        max_spike_t

        max_any_dataseries_t: float = max(max_pos_t, max_spike_t)
        max_any_dataseries_t
        if epochs_df is None:
            epochs_df = sess.epochs.to_dataframe()
        epochs_df = ensure_dataframe(epochs_df)
        epochs_df['stop'] = [*epochs_df['stop'].values[:-1], max_any_dataseries_t]
        epochs_df['duration'] = epochs_df['stop'] - epochs_df['start']
        return ensure_Epoch(epochs_df)

    @classmethod
    def session_fixup_epochs(cls, sess, override_session_epochs: Optional[Epoch]=None, enable_global_epoch: bool=True, **kwargs) -> Epoch:
        """ fixes up the loaded epochs
        has two conflicting (overlapping) epochs:
        "maze"
        "sprinkle"
        
        
                start   stop     label  duration
        0      0   7407       pre      7407
        1   7423  11483      maze      4060
        3  10186  11483  sprinkle      1297
        2  11497  25987      post     14490

        [4 rows x 4 columns]

        --- we want to compute the disjoint set 'roam', 'sprinkle'
        
        
        Usage:
        
            session_epochs: Epoch = BapunDataSessionFormatRegisteredClass.session_fixup_epochs(sess=curr_active_pipeline.sess)
        
        """
        if override_session_epochs is None:
            override_session_epochs = deepcopy(sess.epochs)

        updated_epochs: Epoch = deepcopy(sess.epochs)
        if (not hasattr(sess, 'epochs_bak')):
            print(f'fixing up session computation epochs...')
            # updated_epochs = cls._bapun_session_fixup_epochs_to_be_non_overlapping(bapun_epochs=override_session_epochs, enable_global_epoch=enable_global_epoch, **kwargs)
            updated_epochs = cls.make_last_epoch_finite_from_dataseries(sess=sess, epochs_df=updated_epochs)
            sess.epochs_bak = deepcopy(sess.epochs) ## backup the bad ones
            sess.epochs = updated_epochs
            print(f'\tdone. new epochs: \n{updated_epochs}\n')
        else:
            print(f'WARN: already fixedup session epochs.')
            

        assert np.isfinite(ensure_dataframe(updated_epochs)[['start', 'stop', 'duration']].to_numpy()).all(), f"ensure_dataframe(updated_epochs): {ensure_dataframe(updated_epochs)} has non-finite elements!"

        return updated_epochs
    
    ## Initial Function required to wrangle the data from the raw output to a format like Bapun's .npy format:
    @classmethod
    def initialize_data_directory(cls, filepath=Path('/home/halechr/FastData/Rachel/20230614_Rachel'), filename: str = '20230614_Rachel', **kwargs):
        """ TODO: this function is supposed to combine all the steps needed to process a freshly output recording directory to generate the required *.npy files that are used to build the session. 
        
            I did my best to piece together the relevant looking parts of Rachel's pre-processing scripts/notebooks (`test.py` and `ttl_check.ipynb`) but they don't appear sufficient to perform all the pre-processing. I think this was becuase Rachel did some of the conversion in MATLAB. These scripts will need to be converted to folded in to this function. 
            
        """
        return cls.initialize_data_directory_circular_maze(filepath=filepath, filename=filename, **kwargs)        


    @classmethod
    def _OLD_initialize_data_directory(cls, filepath=Path('/home/halechr/FastData/Rachel/20230614_Rachel'), filename: str = '20230614_Rachel'):
        """ TODO: this function is supposed to combine all the steps needed to process a freshly output recording directory to generate the required *.npy files that are used to build the session. 
        
            I did my best to piece together the relevant looking parts of Rachel's pre-processing scripts/notebooks (`test.py` and `ttl_check.ipynb`) but they don't appear sufficient to perform all the pre-processing. I think this was becuase Rachel did some of the conversion in MATLAB. These scripts will need to be converted to folded in to this function. 
            
        """
        
        # global_data_root_parent_path = find_first_extant_path([Path(r'W:\Data'), Path(r'/home/halechr/FastData'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data'), Path(r'/home/halechr/cloud/turbo/Data')])
        # assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"


        # ## Rachel:
        # active_data_mode_name = 'rachel'
        # local_session_root_parent_context = IdentifyingContext(format_name=active_data_mode_name) # , animal_name='', configuration_name='one', session_name=a_sess.session_name
        # local_session_root_parent_path = global_data_root_parent_path.joinpath('Rachel')

        # basedir: Path = Path(r'W:\Data\Rachel\20230614_Rachel').resolve()

        # basedir: Path = Path('/home/halechr/FastData/Rachel/20230614_Rachel/merged_20230614_2crs.GUI').resolve()

        # basedir: Path = Path('/home/halechr/FastData/Rachel/20230614_Rachel').resolve()
        basedir: Path = filepath.resolve()
        assert basedir.exists()
        print(f'basedir: {basedir}')

        # filename: str = '20230614_Rachel'
        print(f'filename: {filename}')

        # RachelDataSessionFormat.initialize_data_directory(basedir)
        ## Builds the .neurons.npy:
        # folder = Path('/home/wahlberg/Exp_Data/M1_Nov2021/20211123/merged_M1_20211123_raw/merged_M1_20211123_raw_phy')
        # folder = Path(r'W:\Data\Rachel\20230614_Rachel')
        folder = basedir.resolve()
        phydata = PhyIO(folder)
        # /home/halechr/FastData/Rachel/20230614_Rachel/params.py


        # neuronIDs = pd.read_csv(r'W:\Data\Rachel\20230614_Rachel\cluster_q.tsv');

        neuronIDs = pd.read_csv(basedir.joinpath('cluster_q.tsv'))
        neurons = Neurons(spiketrains=phydata.spiketrains, t_stop=2*3600, sampling_rate=30000, neuron_ids = {1:'pyr1',2:'pyr2',3:'pyr3',4:'int1',5:'int2',6:'int3',7:"mua1",8:'mua2',9:'mua3'})
        neurons.filename = folder.joinpath(f'{filename}.neurons.npy')
        neurons.save()

        # Probe Groups file
        # TODO: Probe group generation
        # shanks = []
        # # channel_groups = sess.recinfo.channel_groups
        # for i in range(8):
        #     shank = Shank.auto_generate(
        #         columns=1,
        #         contacts_per_column=128,
        #         xpitch=90,
        #         ypitch=0,
        #         y_shift_per_column=[0, 0],
        #         channel_id=np.arange(0,128,1)
        #         ),
            
        # elec_IDs = np.arange(0,128,1)
        # shanks = Shank.auto_generate(channels=1, contacts_per_column = 128)
        # shanks = pd.read_csv('/home/wahlberg/Exp_Data/M1_Nov2021/20211123/merged_M1_20211123_raw/Probe.csv',delimiter=',',usecols=["ShankNumber"])
        # prb = Probe(shanks)
        # prbgroup = ProbeGroup()
        # prbgroup.add_probe(prb)



        ## Builds the .position.npy:
        # opti_folder = Path(r'W:\Data\Rachel\20230614_Rachel')
        # opti_folder = basedir.resolve()
        # opti_data = OptitrackIO(opti_folder)
        # brelative = pd.read_csv(r'W:\Data\Rachel\20230614_Rachel\merged_M1_20211123_raw_behavior_relativetoLFP.csv',header = None)

        csv_path = basedir.joinpath(f'20230614_positionData.csv')
        # brelative = pd.read_csv(csv_path, header = None)
        brelative = pd.read_csv(csv_path)
        brelative
        # Change column type to timedelta64[ns] for column: 'AbsoluteTime'
        brelative = brelative.astype({'AbsoluteTime': 'timedelta64[ns]'})

        brelative_seconds = brelative.AbsoluteTime.dt.total_seconds().to_numpy()
        start_t = np.min(brelative_seconds)
        # start_t = np.min(brelative.AbsoluteTime.to_numpy())
        print(f'start_t: {start_t}')

        t_relative = brelative_seconds - start_t
        t_relative

        print(f'brelative.shape: {brelative.shape}')
        # d = {'t':brelative[0],'x':opti_data.z,'y':opti_data.x} 
        d = {'t':brelative_seconds,'x':brelative.Z.to_numpy(),'y':brelative.X.to_numpy()} 

        behaviordf = pd.DataFrame(data=d)
        print(f'behaviordf.shape: {behaviordf.shape}')
        position = Position(behaviordf)
        # position.filename = Path(f'W:\Data\Rachel\20230614_Rachel\{filename}.position.npy')
        position.filename = basedir.joinpath(f'{filename}.position.npy')
        position.save()


        ## Builds the .paradigm.npy file from scratch:
        # ## Old example:
        # starts = [0, 5*60]
        # stops = [5*60-1, 3.8398632e+03]
        # labels = ['pre','maze']
        # d = {'start':starts,'stop':stops,'label':labels} 
        
        ## 2023-10-26 - Example from Rachel's new data:
        paradigm_df = pd.DataFrame(dict(label=['Pre','Maze','Post'],
            start = ['11:15:49.000','12:02:19.095','13:08:37.999'],
            stops = ['11:53:19.384','13:04:54.815','14:53:22.047'],
        ))
        # Change column type to timedelta64[ns] for columns: 'Starts', 'Stops'
        paradigm_df = paradigm_df.astype({'start': 'timedelta64[ns]', 'stops': 'timedelta64[ns]'})

        ## Build and save the paradigm.npy
        d = {'start':paradigm_df.start.dt.total_seconds().to_numpy(),
            'stop':paradigm_df.stops.dt.total_seconds().to_numpy(),
            'label':paradigm_df.label.to_list()} 

        paradigmdf = pd.DataFrame(data=d)
        paradigm = Epoch(paradigmdf)
        # paradigm.filename = Path('/home/wahlberg/Exp_Data/M1_Nov2021/20211123/merged_M1_20211123_raw/merged_M1_20211123_raw_phy/merged_M1_20211123_raw.paradigm.npy')
        paradigm.filename = basedir.joinpath(f'{filename}.paradigm.npy')
        paradigm.save()



        # _test_session = RachelDataSessionFormat.build_session(basedir)
        # _test_session, loaded_file_record_list = RachelDataSessionFormat.load_session(_test_session)
        # _test_session


        # # ==================================================================================================================== #
        # # BEGIN PRE 2023-10-26                                                                                                 #
        # # ==================================================================================================================== #
        # ## Builds the .neurons.npy:
        # # folder = Path('/home/wahlberg/Exp_Data/M1_Nov2021/20211123/merged_M1_20211123_raw/merged_M1_20211123_raw_phy')
        # folder = Path(r'W:\Data\Rachel\20230614_Rachel')
        # phydata = PhyIO(folder)

        # neuronIDs = pd.read_csv(r'W:\Data\Rachel\20230614_Rachel\cluster_q.tsv');

        # neurons = Neurons(spiketrains=phydata.spiketrains, t_stop=2*3600, sampling_rate=30000, neuron_ids = {1:'pyr1',2:'pyr2',3:'pyr3',4:'int1',5:'int2',6:'int3',7:"mua1",8:'mua2',9:'mua3'})
        # neurons.filename = folder.joinpath('merged_M1_20211123_raw.neurons.npy')
        # neurons.save()

        # # Probe Groups file
        # # TODO: Probe group generation
        # # shanks = []
        # # # channel_groups = sess.recinfo.channel_groups
        # # for i in range(8):
        # #     shank = Shank.auto_generate(
        # #         columns=1,
        # #         contacts_per_column=128,
        # #         xpitch=90,
        # #         ypitch=0,
        # #         y_shift_per_column=[0, 0],
        # #         channel_id=np.arange(0,128,1)
        # #         ),
            
        # # elec_IDs = np.arange(0,128,1)
        # # shanks = Shank.auto_generate(channels=1, contacts_per_column = 128)
        # # shanks = pd.read_csv('/home/wahlberg/Exp_Data/M1_Nov2021/20211123/merged_M1_20211123_raw/Probe.csv',delimiter=',',usecols=["ShankNumber"])
        # # prb = Probe(shanks)
        # # prbgroup = ProbeGroup()
        # # prbgroup.add_probe(prb)


        # ## Builds the .position.npy:
        # opti_folder = Path(r'W:\Data\Rachel\20230614_Rachel')
        # opti_data = OptitrackIO(opti_folder)
        # brelative = pd.read_csv(r'W:\Data\Rachel\20230614_Rachel\merged_M1_20211123_raw_behavior_relativetoLFP.csv',header = None)
        # print(f'brelative.shape: {brelative.shape}')
        # d = {'t':brelative[0],'x':opti_data.z,'y':opti_data.x} 
        # behaviordf = pd.DataFrame(data=d)
        # print(f'behaviordf.shape: {behaviordf.shape}')
        # position = Position(behaviordf)
        # position.filename = Path('W:\Data\Rachel\20230614_Rachel\merged_M1_20211123_raw.position.npy')
        # position.save()

        # # ## Builds the .paradigm.npy file from scratch:
        # # starts = [0,5*60]
        # # stops = [5*60-1,3.8398632e+03]
        # # labels = ['pre','maze']
        # # d = {'start':starts,'stop':stops,'label':labels} 
        # # paradigmdf = pd.DataFrame(data=d)
        # # paradigm = Epoch(paradigmdf)
        # # paradigm.filename = Path('/home/wahlberg/Exp_Data/M1_Nov2021/20211123/merged_M1_20211123_raw/merged_M1_20211123_raw_phy/merged_M1_20211123_raw.paradigm.npy')
        # # paradigm.save()



    @classmethod
    def perform_initialize_combined_pos_file(cls, filepath=Path('/nfs/turbo/umms-kdiba/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/'), filename: str = 'petunia_241209_merged', override_position_npy_paths=None):
        """ builds session positions from scratch
        Fixes errors like:
            neuropy.core.session.Formats.SessionSpecifications.RequiredFileError: Required File: /nfs/turbo/umms-kdiba/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/petunia_241209_merged.position.npy does not exist.
            
        """
        if isinstance(filepath, str):
            filepath: Path = Path(filepath)
        print(f'Processing Rachel-style datafolder at filepath: "{filepath.as_posix()}"')


        basedir: Path = filepath.resolve()
        assert basedir.exists()
        # print(f'basedir: {basedir}')

        # filename: str = '20230614_Rachel'
        # print(f'filename: {filename}')

        ## Position File
        ## INPUTS: basedir, 
        from neuropy.core.position import Position, PositionAccessor

        # ## Find *.npy files in the basedir folder
        # glob_str: str = f"**/*.position*.npy"
        # position_npy_paths = find_data_files(basedir, glob_str=glob_str)
        # position_npy_paths

        # position_npy_paths = ["W:/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/petunia_241209_merged.position_2.npy",
        #     "W:/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/petunia_241209_merged.position_track_2.npy",
        #     "W:/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/petunia_241209_merged.position_1.npy",
        #     "W:/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/petunia_241209_merged.position_track_1.npy",
        # ]

        ## INPUTS: basedir
        if override_position_npy_paths is not None:
            position_npy_paths = override_position_npy_paths
        else:
            position_npy_paths = [
                # basedir.joinpath('petunia_241209_merged.position_linear_1.npy'),
                basedir.joinpath(f'{filename}.position_track_1.npy'),
                basedir.joinpath(f'{filename}.position_track_2.npy'),
                # basedir.joinpath('petunia_241209_merged.position_linear_2.npy'),
                basedir.joinpath(f'{filename}.position_1.npy'),
                basedir.joinpath(f'{filename}.position_2.npy'),
            ]
            position_npy_paths = [v for v in position_npy_paths if v.exists()]

        # pos_dict = {Path(f).stem.split('.')[-1]:np.load(f, allow_pickle=True).item() for f in position_npy_paths}

        pos_dict: Dict[str, Position] = {}
        
        for f in position_npy_paths:
            key = Path(f).stem.split('.')[-1]
            dict_rep = load_npy_pickled_item(f)
            try:
                a_pos = Position.init(**dict_rep)
                pos_dict[key] = a_pos
            except TypeError as e:
                # TypeError: init() got an unexpected keyword argument 'df'
                a_pos = Position(pos_df=dict_rep['df']) ## newer way? something wrong with how I saved it?
                pos_dict[key] = a_pos
            except Exception as e:
                print(f'FAILED for path f: "{f}" with error {e}')
                raise
            
        # {Path(f).stem.split('.')[-1]:Position.init(**np.load(f, allow_pickle=True).item()) for f in position_npy_paths}
        ## END for f in position_npy_p...
        
        # from_dataframe

        # pos_dict
        # pos_list: List[Position] = list(pos_dict.values())
        # merged_pos: Position = Position.concat(pos_list)
        merged_pos: Position = Position.concat(list(pos_dict.values()))
        merged_pos

        ## OUTPUTS: merged_pos

        pos_file_path: Path = basedir.joinpath(f'{filename}.position.npy')
        # pos_file = sess.filePrefix.with_suffix(".position.npy")
        # pos_file = Path(r"W:\Data\Rachel\Petunia\Recordings\CA1\2024-12-09\petunia_241209_merged.position.npy").resolve() # sess.filePrefix.with_suffix(".position.npy")
        # pos_file = Path(r"W:\Data\Rachel\Petunia\Recordings\CA1\2024-12-09\petunia_241209_merged.position.npy").resolve() # sess.filePrefix.with_suffix(".position.npy")
        merged_pos.filename = pos_file_path
        merged_pos.save()
        print(f'saved required position file "{pos_file_path.as_posix()}" to basedir.')
        
        # position_npy_paths

        ## OUTPUT FILES: pos_file_path
        return pos_file_path, (merged_pos, pos_dict)



    @classmethod
    def perform_initialize_combined_sess_paradigm_epochs_file(cls, pos_dict: Dict, filepath=Path('/nfs/turbo/umms-kdiba/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/'), filename: str = 'petunia_241209_merged'):
        """ builds session paradigm epochs from scratch 
        
        Fixes errors like:
            neuropy.core.session.Formats.SessionSpecifications.RequiredFileError: Required File: /nfs/turbo/umms-kdiba/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/petunia_241209_merged.position.npy does not exist.
            
        """
        if isinstance(filepath, str):
            filepath: Path = Path(filepath)
        print(f'Processing Rachel-style datafolder at filepath: "{filepath.as_posix()}"')


        basedir: Path = filepath.resolve()
        assert basedir.exists()
        # print(f'basedir: {basedir}')

        ### Build epochs from track positions, not sure if correct:
        # paradigm_file_path = Path(r"W:\Data\Rachel\Petunia\Recordings\CA1\2024-12-09\petunia_241209_merged.paradigm.npy").resolve()
        paradigm_file_path: Path = basedir.joinpath(f'{filename}.paradigm.npy').resolve()

        epochs_df = pd.DataFrame.from_records([
            dict(label='maze1', start=pos_dict['position_track_1'].t_start, stop=pos_dict['position_track_1'].t_stop),
            dict(label='maze2', start=pos_dict['position_track_2'].t_start, stop=pos_dict['position_track_2'].t_stop),
        ]).epochs.get_valid_df()

        

        paradigm_epochs: Epoch = Epoch(epochs=epochs_df)

        paradigm_epochs.filename = paradigm_file_path
        paradigm_epochs.save()

        ## OUTPUT FILES: paradigm_file_path
        print(f'saved required epochs paradigm file "{paradigm_file_path.as_posix()}" to basedir.')
        
        # position_npy_paths

        ## OUTPUT FILES: paradigm_file_path
        return paradigm_file_path, paradigm_epochs


    @classmethod
    def initialize_data_directory_circular_maze(cls, filepath=Path('/nfs/turbo/umms-kdiba/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/'), filename: str = 'petunia_241209_merged'):
        """ TODO: this function is supposed to combine all the steps needed to process a freshly output recording directory to generate the required *.npy files that are used to build the session. 
        
            I did my best to piece together the relevant looking parts of Rachel's pre-processing scripts/notebooks (`test.py` and `ttl_check.ipynb`) but they don't appear sufficient to perform all the pre-processing. I think this was becuase Rachel did some of the conversion in MATLAB. These scripts will need to be converted to folded in to this function. 

        Fixes errors like:
        
            neuropy.core.session.Formats.SessionSpecifications.RequiredFileError: Required File: /nfs/turbo/umms-kdiba/Data/Rachel/Petunia/Recordings/CA1/2024-12-09/petunia_241209_merged.position.npy does not exist.
            
        """
        
        # global_data_root_parent_path = find_first_extant_path([Path(r'W:\Data'), Path(r'/home/halechr/FastData'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data'), Path(r'/home/halechr/cloud/turbo/Data')])
        # assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"


        # ## Rachel:
        # active_data_mode_name = 'rachel'
        # local_session_root_parent_context = IdentifyingContext(format_name=active_data_mode_name) # , animal_name='', configuration_name='one', session_name=a_sess.session_name
        # local_session_root_parent_path = global_data_root_parent_path.joinpath('Rachel')

        # basedir: Path = Path(r'W:\Data\Rachel\20230614_Rachel').resolve()

        # basedir: Path = Path('/home/halechr/FastData/Rachel/20230614_Rachel/merged_20230614_2crs.GUI').resolve()

        # basedir: Path = Path('/home/halechr/FastData/Rachel/20230614_Rachel').resolve()
        
        if isinstance(filepath, str):
            filepath: Path = Path(filepath)
        print(f'Processing Rachel-style datafolder at filepath: "{filepath.as_posix()}"')


        basedir: Path = filepath.resolve()
        assert basedir.exists()
        print(f'basedir: {basedir}')

        # filename: str = '20230614_Rachel'
        print(f'filename: {filename}')

        ## Position File
        ## INPUTS: basedir, 
        position_npy_paths = []
        
        if basedir.joinpath(f'{filename}.position.npy').exists():
            ## use that
            position_npy_paths = [basedir.joinpath(f'{filename}.position.npy')]
        else:
            position_npy_paths = [
                # basedir.joinpath('petunia_241209_merged.position_linear_1.npy'),
                basedir.joinpath(f'{filename}.position_track_1.npy'),
                basedir.joinpath(f'{filename}.position_track_2.npy'),
                # basedir.joinpath('petunia_241209_merged.position_linear_2.npy'),
                basedir.joinpath(f'{filename}.position_1.npy'),
                basedir.joinpath(f'{filename}.position_2.npy'),
            ]        
            
        position_npy_paths = [v for v in position_npy_paths if v.exists()]
        assert len(position_npy_paths) > 0


        pos_file_path, (merged_pos, pos_dict) = cls.perform_initialize_combined_pos_file(filepath=filepath, filename=filename, override_position_npy_paths=position_npy_paths)
        
        paradigm_file_path, paradigm_epochs = cls.perform_initialize_combined_sess_paradigm_epochs_file(pos_dict=pos_dict, filepath=filepath, filename=filename)

        # neuronIDs = pd.read_csv(basedir.joinpath('cluster_q.tsv'))
        # neurons = Neurons(spiketrains=phydata.spiketrains, t_stop=2*3600, sampling_rate=30000, neuron_ids = {1:'pyr1',2:'pyr2',3:'pyr3',4:'int1',5:'int2',6:'int3',7:"mua1",8:'mua2',9:'mua3'})
        # neurons.filename = folder.joinpath(f'{filename}.neurons.npy')
        # neurons.save()

        # from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters # used particularly to build Bapun-style filters

        # # active_data_mode_name = 'bapun'
        # active_data_mode_name = 'rachel'
        # print(f'active_data_session_types_registered_classes_dict: {active_data_session_types_registered_classes_dict}')
        # active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
        # active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

        # # basedir = Path('/media/halechr/MAX/Data/Rachel/Cho_241117_Session2').resolve()
        # ## INPUTS: basedir

        # force_reload = force_reload #True
        # print(f'force_reload: {force_reload}')
        # curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties, override_basepath=Path(basedir), force_reload=force_reload) # , override_parameters_flat_keypaths_dict=override_parameters


        print(f'2025-09-17 - Rachel-formatted session is now ready to load with the regular `NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(...)` code.')
        