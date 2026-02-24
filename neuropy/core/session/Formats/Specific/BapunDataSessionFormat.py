 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## typehinting only imports here
    from neuropy.core.position import Position

from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from pathlib import Path
from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec

# For specific load functions:
from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters # used particularly to build Bapun-style filters
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import HardcodedProcessingParameters
from neuropy.utils.position_util import ShapelyMaze, ShapelyMazeCollection
from shapely import box ## used by `build_Bapun_Day4OpenField_laps_from_reward_zones`
# from shapely.geometry import LineString, Point 


# linearization_method: str = 'umap'

# linearization_method: str = 'shapely'
Day5TwoNovel_all_session_mazes: ShapelyMazeCollection = ShapelyMazeCollection(shapelyMazes = {
    # Define the skeletons (re-using the coordinates identified earlier)
    # "N"-shaped maze
    'maze1': ShapelyMaze(nodes = [
        (-66.31, 88.82),  # TL
        (-64.57, -54.62), # BL
        (25.0, 65.0),  # TR
        (73.88, -59.85)   # BR
    ]),
    # "U"-shaped maze
    'maze2': ShapelyMaze(nodes = [
        (-48.62, 63.79),  # Top-Left
        (-33.99, -41.77), # Bot-Left
        (3.32, -49.23),   # Bot-Mid
        (37.16, -34.00),  # Bot-Right
        (52.74, 76.34)    # Top-Right
    ]),
},
    valid_epochs =  {'maze1': (11070.0, 13970.0), 'maze2': (20756.0, 24004.0)}, # 'maze_GLOBAL': (0.0, 42305.0), 
)   



class BapunDataSessionFormatRegisteredClass(DataSessionFormatBaseRegisteredClass):
    """

    # Example Filesystem Hierarchy:
    📦Day5TwoNovel
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.eeg
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.flattened.spikes.npy <-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.maze1.linear.npy     <-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.maze2.linear.npy     <-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.mua.npy              <-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.neurons.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.nrs
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.paradigm.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.pbe.npy              <-OPT-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.position.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.probegroup.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.ripple.npy           <-OPT-GEN
     ┗ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.xml
    
    
    By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file:
        basedir: Path('R:\data\Bapun\Day5TwoNovel')
        session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
    
    From here, a list of known files to load from is determined:
        
    Usage:
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
        from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass

        _test_session = BapunDataSessionFormatRegisteredClass.build_session(Path('R:\data\Bapun\Day5TwoNovel'))
        _test_session, loaded_file_record_list = BapunDataSessionFormatRegisteredClass.load_session(_test_session)
        _test_session
        
    """
    _session_class_name = 'bapun'
    _session_default_relative_basedir = r'data/Bapun/Day5TwoNovel'
    _session_default_basedir = r'W:\data\Bapun\Day5TwoNovel' # WINDOWS
    # _session_default_basedir = r'/run/media/halechr/MoverNew/data/Bapun/Day5TwoNovel'
    _session_basepath_to_context_parsing_keys = ['format_name', 'animal', 'session_name']

    _time_variable_name = 't_seconds' # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data

    
    @classmethod
    def _get_session_specific_parameters(cls, session_context: IdentifyingContext) -> HardcodedProcessingParameters:
        """ session-specific type parameters 
         
        #TODO 2025-09-20 19:26: - [ ] Is this redudndant with preprocessing parameters?
        """
        
        bapun_open_field_grid_bin_bounds = (((-120.0, 120.0), (-120.0, 120.0)))
        
        the_dict: Dict[IdentifyingContext, HardcodedProcessingParameters]  = { #  
            IdentifyingContext(format_name= 'bapun', animal= 'RatN', session_name= 'Day4OpenField'): HardcodedProcessingParameters(decoder_building_session_names=['roam', 'sprinkle', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['roam', 'sprinkle'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(custom_lap_estimation_fn=cls.build_Bapun_Day4OpenField_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=20.0, merging_adjacent_max_separation_sec=6.0,),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatU', session_name= 'RatUDay5OpenfieldSD'): HardcodedProcessingParameters(
                # decoder_building_session_names=['maze', 'sprinkle', 'maze_GLOBAL'],
                decoder_building_session_names=['roam', 'sprinkle', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                # non_global_activity_session_names=['maze', 'sprinkle'],
                non_global_activity_session_names=['roam', 'sprinkle'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(custom_lap_estimation_fn=cls.build_Bapun_Day4OpenField_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),      
            IdentifyingContext(format_name= 'bapun', animal= 'RatK', session_name= 'Day4Openfield'): HardcodedProcessingParameters(
                # decoder_building_session_names=['maze', 'sprinkle', 'maze_GLOBAL'],
                decoder_building_session_names=['maze'],
                # global_session_name='maze_GLOBAL',
                global_session_name='maze',
                # non_global_activity_session_names=['maze', 'sprinkle'],
                non_global_activity_session_names=['maze'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(custom_lap_estimation_fn=cls.build_Bapun_Day4OpenField_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatS', session_name= 'Day5TwoNovel'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='shapely', all_session_mazes=Day5TwoNovel_all_session_mazes),
            ),
            ## Fallback defaults:
            IdentifyingContext(format_name= 'bapun'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0,),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),									
        }

        best_match = IdentifyingContext.matching(the_dict, criteria=session_context.get_subset(subset_includelist=cls._session_basepath_to_context_parsing_keys).to_dict())
        return list(best_match.values())[0] ## return the first match


    @classmethod
    def get_session_name(cls, basedir):
        """ returns the session_name for this basedir, which determines the files to load. """
        # Find the only .xml file to obtain the session name
        return DataSessionFormatBaseRegisteredClass.find_session_name_from_sole_xml_file(basedir) # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'

    @classmethod
    def get_session_spec(cls, session_name):
        return SessionFolderSpec(required_files=[SessionFileSpec('{}.xml', session_name, 'The primary .xml configuration file', cls._load_xml_file),
                                           SessionFileSpec('{}.neurons.npy', session_name, 'The numpy data file containing information about neural activity.', cls._load_neurons_file),
                                           SessionFileSpec('{}.probegroup.npy', session_name, 'The numpy data file containing information about the spatial layout of recording probes', cls._load_probegroup_file),
                                           SessionFileSpec('{}.position.npy', session_name, 'The numpy data file containing the recorded animal positions (as generated by optitrack) over time.', cls._load_position_file),
                                           SessionFileSpec('{}.paradigm.npy', session_name, 'The numpy data file containing the recording epochs. Each epoch is defined as a: (label:str, t_start: float (in seconds), t_end: float (in seconds))', cls._load_paradigm_file)]
                    )
    ### Specific Load Functions used in the session_spec
    @classmethod
    def _load_neurons_file(cls, filepath, session): # .neurons
        ## here we need to handle the "1" type cells
        """
            neuron_type
                array(['pyr', 'pyr', 'pyr', 'mua', '1', 'inter', 'pyr', '1', 'pyr', 'pyr', '1', '1', '1', 'pyr', 'mua', 'mua', 'pyr', '1', 'pyr', 'mua', 'mua', 'pyr', '1', 'mua', '1', 'pyr', 'pyr', 'pyr', 'mua', 'pyr', 'pyr', 'pyr', 'pyr', '1', '1', 'pyr', 'pyr', '1', 'pyr', 'mua', '1', '1', '1', '1', '1', 'inter', 'pyr', 'mua', '1', 'pyr', 'pyr', 'pyr', 'mua', 'pyr', 'pyr', 'inter', 'pyr', 'pyr', 'pyr', '1', 'pyr', 'pyr', 'pyr', 'pyr', 'pyr', '1', 'pyr', '1', 'pyr', 'pyr', '1', 'pyr', 'mua', 'pyr', 'mua', '1', 'pyr'], dtype='<U5')
            neuron_type[neuron_type == '1']
                array(['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], dtype='<U5')
            neuron_type[neuron_type == '1'] = 'mua'

        """
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
    def _bapun_session_fixup_epochs_to_be_non_overlapping(cls, curr_sess_context: IdentifyingContext, bapun_epochs: Epoch, enable_global_epoch: bool=True) -> Epoch:
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
        from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch
        
        bapun_epochs_df: pd.DataFrame = ensure_dataframe(bapun_epochs) #.to_dataframe()
        bapun_params = cls._get_session_specific_parameters(session_context=curr_sess_context)

        is_bapun_Day4OpenField_sess: bool = False
        needs_update: bool = True
        if curr_sess_context is not None:
            is_bapun_Day4OpenField_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'session_name': 'Day4OpenField'}) ## all must match, 'animal': 'RatN'
            is_bapun_ratK_Day4OpenField_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'animal': 'RatK', 'session_name': 'Day4Openfield'}) ## all must match, 'animal': 'RatK'
            is_bapun_RatUDay5OpenfieldSD_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'session_name': 'RatUDay5OpenfieldSD'}) ## all must match, 'animal': 'RatN'

            if is_bapun_ratK_Day4OpenField_sess:
                assert (len(bapun_epochs_df) == 3), f"{len(bapun_epochs_df)}"
                needs_update = not ((len(bapun_epochs_df) == 3) and (['pre', 'maze', 'post'] == bapun_epochs_df['label'].to_list()))
                if not needs_update:
                    print(f'overrinding enable_global_epoch == False since this session only has one maze epoch')
                    enable_global_epoch = False
                # assert (len(bapun_epochs_df) == 4), f"{len(bapun_epochs_df)}"
                # needs_update = (len(bapun_epochs_df) == 4) and ('roam' not in bapun_epochs_df['label'].to_list())
            elif is_bapun_Day4OpenField_sess:
                assert (len(bapun_epochs_df) == 4), f"{len(bapun_epochs_df)}"
                needs_update = (len(bapun_epochs_df) == 4) and ('roam' not in bapun_epochs_df['label'].to_list())
            elif is_bapun_RatUDay5OpenfieldSD_sess:
                assert (len(bapun_epochs_df) >= 6), f"{len(bapun_epochs_df)}"
                needs_update = (len(bapun_epochs_df) >= 6) and ('roam' not in bapun_epochs_df['label'].to_list())


        if ((is_bapun_Day4OpenField_sess or is_bapun_RatUDay5OpenfieldSD_sess) and needs_update):
            ## Applicable to Day4OpenField only: add the 'roam' row if it doesn't already exist
            # bapun_epochs_arr = bapun_epochs_df.to_numpy()
            # new_roam_row = [bapun_epochs_arr[1, 0], (bapun_epochs_arr[2, 0]-1), 'roam', 0.0] # ['start', 'stop', 'label', 'duration']
            # # bapun_epochs_arr[1, 1] = bapun_epochs_arr[2, 0] - 1 # overwrite the "maze" portion's end-time
            # fixed_bapun_epochs_arr = deepcopy(bapun_epochs_arr)
            # fixed_bapun_epochs_arr[1, :] = new_roam_row ## replace the entire 2nd row (the 'maze' row) with the new 'roam' row.
            # # build final epochs_df from fixed_bapun_epochs_arr
            # bapun_epochs_df: pd.DataFrame = pd.DataFrame(fixed_bapun_epochs_arr, columns=['start', 'stop', 'label', 'duration'])
            # bapun_epochs_df['label'] = bapun_epochs_df['label'].str.replace('maze', 'roam', n=1) ## rename 'maze' to 'roam'
            bapun_epochs_df['label'] = bapun_epochs_df['label'].map(lambda k: {'maze':'roam'}.get(k, k))  ## rename 'maze' to 'roam'
            bapun_epochs_df.loc[(bapun_epochs_df['label'] == 'roam'), 'stop'] = (bapun_epochs_df[bapun_epochs_df['label'] == 'sprinkle']['start'].item() - 1e-6)  # make sure 'roam' row doesn't overlap the 'sprinkle' row
            bapun_epochs_df[['start', 'stop', 'duration']] = bapun_epochs_df[['start', 'stop', 'duration']].astype(float)
            bapun_epochs_df['duration'] = bapun_epochs_df['stop'] - bapun_epochs_df['start'] ## recompute duration

        if enable_global_epoch:
            # maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            if 'maze_GLOBAL' not in bapun_epochs_df['label']:
                assert len(bapun_params.non_global_activity_session_names) >= 2, f"bapun_params.non_global_activity_session_names: {non_global_activity_session_names}"
                bapun_epochs_df = bapun_epochs_df.epochs.adding_global_epoch_row(global_epoch_name='maze_GLOBAL',
                                             first_included_epoch_name=bapun_params.non_global_activity_session_names[0], last_included_epoch_name=bapun_params.non_global_activity_session_names[-1], inplace=False)

        return Epoch(bapun_epochs_df, metadata=bapun_epochs.metadata)



    @classmethod
    def session_fixup_epochs(cls, sess, override_session_epochs: Optional[Epoch]=None, enable_global_epoch: bool=True, override_extant: bool=True, **kwargs) -> Epoch:
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
        # curr_sess_context = sess.get_session_context()
        curr_sess_context = sess.get_context()
        
        # is_bapun_Day4OpenField_sess: bool = curr_sess_context.query(criteria={'format_name':'bapun', 'animal': 'RatN', 'session_name': 'Day4OpenField'}) ## all must match
        updated_epochs: Epoch = deepcopy(sess.epochs)
        needs_update: bool = False        

        if (not hasattr(sess, 'epochs_bak')):
            print(f'fixing up session computation epochs...')
            sess.epochs_bak = deepcopy(sess.epochs) ## backup the bad ones
            needs_update = True
        else:
            print(f'WARN: already fixedup session epochs.')
            if override_extant:
                print(f'\trestoring backed up epochs:')
                sess.epochs = deepcopy(sess.epochs_bak)
                needs_update = True


        if needs_update:
            if override_session_epochs is None:
                override_session_epochs = deepcopy(sess.epochs)
            
            updated_epochs = cls._bapun_session_fixup_epochs_to_be_non_overlapping(curr_sess_context=curr_sess_context, bapun_epochs=override_session_epochs, enable_global_epoch=enable_global_epoch, **kwargs)
            sess.epochs = updated_epochs
            print(f'\tdone. new epochs: \n{updated_epochs}\n')

            
        return updated_epochs



    
    # Not limited:
    @classmethod
    def build_filters_any_epochs(cls, sess, filter_name_suffix=None):
        return build_custom_epochs_filters(sess, filter_name_suffix=filter_name_suffix)
    
    # Any epoch on the maze, not limited to pyramidal cells, etc
    @classmethod
    def build_filters_any_maze_epochs(cls, sess, filter_name_suffix=None, override_epoch_name_includelist: Optional[List[str]]=None):
        all_epochs_names: List[str] = sess.epochs.to_dataframe()['label'].to_list()
        if override_epoch_name_includelist is not None:
            ## use `override_epoch_name_includelist`
            maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem in override_epoch_name_includelist, names))
            assert (len(maze_only_name_filter_fn(all_epochs_names)) > 0), f"could not match any of the epochs in override_epoch_name_includelist: {override_epoch_name_includelist} session name formats. Need an override!\n\tall_epochs_names: {all_epochs_names}"

        else:
            maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.startswith('maze'), names))
            if len(maze_only_name_filter_fn(all_epochs_names)) == 0:
                ## no epochs starting with "maze", assume it's the other type of session
                maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem in ['roam', 'sprinkle'], names))
                assert (len(maze_only_name_filter_fn(all_epochs_names)) > 0), f"could not match either 'maze*' or ['roam', 'sprinkle'] session name formats. Need an override!\n\tall_epochs_names: {all_epochs_names}"


        maze_only_filters = build_custom_epochs_filters(sess, epoch_name_includelist=maze_only_name_filter_fn, filter_name_suffix=filter_name_suffix)
        return maze_only_filters


    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_includelist=None, filter_name_suffix=None, include_global_epoch=False):
        ## TODO: currently hard-coded
        # active_session_filter_configurations = cls.build_filters_any_epochs(sess)
        active_session_filter_configurations = cls.build_filters_any_maze_epochs(sess, filter_name_suffix=filter_name_suffix, override_epoch_name_includelist=epoch_name_includelist)
        return active_session_filter_configurations
    
        
    #######################################################
    ## Bapun Nupy Format Only Methods:
    # @classmethod
    # def _default_compute_flattened_spikes(cls, session, timestamp_scale_factor=(1/1E4), spike_timestamp_column_name='t_seconds', progress_tracing=True):
    #     spikes_df = FlattenedSpiketrains.build_spike_dataframe(session, timestamp_scale_factor=timestamp_scale_factor, spike_timestamp_column_name=spike_timestamp_column_name, progress_tracing=progress_tracing)
    #     print(f'spikes_df.columns: {spikes_df.columns}')
    #     session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=spike_timestamp_column_name, t_start=session.neurons.t_start) # FlattenedSpiketrains(spikes_df)
    #     print('\t Done!')
    #     return session
    
    # @classmethod
    # def _add_missing_spikes_df_columns(cls, spikes_df, neurons_obj):
    #     spikes_df, neurons_obj._reverse_cellID_index_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
    #     spikes_df['t'] = spikes_df[cls._time_variable_name] # add the 't' column required for visualization
        
    
    ## Main load function:
    @classmethod
    def load_session(cls, session, debug_print=False):
        session, loaded_file_record_list = DataSessionFormatBaseRegisteredClass.load_session(session, debug_print=debug_print) # call the super class load_session(...) to load the common things (.recinfo, .filePrefix, .eegfile, .datfile)
        remaining_required_filespecs = {k: v for k, v in session.config.resolved_required_filespecs_dict.items() if k not in loaded_file_record_list}
        if debug_print:
            print(f'remaining_required_filespecs: {remaining_required_filespecs}')
        
        for file_path, file_spec in remaining_required_filespecs.items():
            print(f'loading file_path: "{file_path}"...')
            session = file_spec.session_load_callback(file_path, session)
            loaded_file_record_list.append(file_path)
        
        # ['.neurons.npy','.probegroup.npy','.position.npy','.paradigm.npy']
        # session = DataSessionLoader.__default_compute_bapun_flattened_spikes(session)
        # active_session_computation_configs[0].pf_params.linearization_method = "umap"
        session.config.preprocessing_parameters.epoch_estimation_parameters.laps.linearization_method = "umap"
        

        # Load or compute linear positions if needed:        
        if (not session.position.has_linear_pos):
            # compute linear positions:
            print(f'computing linear positions for all active epochs ({session.epochs}) for session...')
            # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
            session.position.linear_pos = np.full_like(session.position.time, np.nan)
            
            # ['pre', 'maze', 'sprinkle', 'post']

            # only_included_pos_computation_labels tries to work around a memory error            
            # NOTE: WARNING: DataSession.compute_linearized_position(session, an_epoch_label) causes MemoryErrors when called for an_epoch_label that doesn't have position data. For Bapun's r'W:\Data\Bapun\RatS\Day5TwoNovel' data, ['maze1', 'maze2'] were the acceptable epochs (and they were hardcoded).
            # I encountered the MemoryError when I ran the same load_session function for r'W:\Data\Bapun\RatN\Day4OpenField', which has the epochs ['pre', 'maze', 'sprinkle', 'post']. Restricting these computations to ['maze'] solves the problem for this session. It seems like I could just detect if there are any position samples left in the filtered position dataframe within the DataSession.compute_linearized_position(session, an_epoch_label), and handle the error there if there aren't (which I believe would prevent the MemoryError and excessive computation).
            
            #TODO 2025-12-17 12:16: - [ ] Needs generalization for the other session names (e.g. 'roam', )
            only_included_pos_computation_labels = ['maze']
            all_epoch_labels = list(session.epochs.labels)
            
            if 'sprinkle' in all_epoch_labels:
                only_included_pos_computation_labels = ['maze', 'sprinkle']

            else:
                only_included_pos_computation_labels = ['maze']

            # only_included_pos_computation_labels = None
            if only_included_pos_computation_labels is None:
                only_included_pos_computation_labels = session.epochs.labels # all labels if no restrictions are specified
                

            for an_epoch_label in session.epochs.labels:
                if an_epoch_label in only_included_pos_computation_labels:
                    an_epoch_timeslice_indicies, active_positions_maze1, linearized_positions_curr_epoch = DataSession._perform_compute_session_linearized_position(session, an_epoch_label, method='umap')
                    session.position.linear_pos[an_epoch_timeslice_indicies] = linearized_positions_curr_epoch.traces
                

            ## Previous 'manual' maze1 and maze2 way that fails for any sessions without these epochs:    
            # acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, 'maze1')
            # acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, 'maze2')
            # session.position.linear_pos[acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            # session.position.linear_pos[acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            
            session.position.filename = session.filePrefix.with_suffix(".position.npy")
            # print('Saving updated position results to {}...'.format(session.position.filename))
            with ProgressMessagePrinter(session.position.filename, action='Saving', contents_description='updated position results'):
                session.position.save()
            # print('done.\n')
        else:
            print('linearized position loaded from file.')

        ## Load or compute flattened spikes since this format of data has the spikes ordered only by cell_id:
        ## flattened.spikes:
        active_file_suffix = '.flattened.spikes.npy'
        # active_file_suffix = '.new.flattened.spikes.npy'
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
            cls._add_missing_spikes_df_columns(spikes_df, session.neurons) # add the missing columns to the dataframe
            session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix) # '.flattened.spikes.npy'
            print('\t Saving computed flattened spiketrains results to {}...'.format(session.flattened_spiketrains.filename), end='')
            session.flattened_spiketrains.save()
            print('\t done.\n')
        
        # Common Extended properties:
        session = cls._default_extended_postload(session.filePrefix, session)
        
        return session, loaded_file_record_list
    
    

    # @function_attributes(short_name=None, tags=['laps', 'shapely', 'segmentation', 'trajectories', 'position', 'Day4OpenField'], input_requires=[], output_provides=[], uses=['shapely'], used_by=[], creation_date='2026-02-20 06:56', related_items=[])
    @classmethod
    def _perform_build_Bapun_Day4OpenField_laps_from_reward_zones(cls, pos: Position, bapun_Day4OpenField_reward_zones: Dict=None):
        """ builds correct laps (transitions between the two reward zones on the open field maze for the 'roam' experiment

        Usage:

            curr_session = curr_active_pipeline.filtered_sessions['roam']
            pos: Position = curr_session.position
            laps_obj, pos = build_Bapun_Day4OpenField_laps_from_reward_zones(pos=pos)
            ## Update the current session
            curr_session.position = pos
            curr_session.laps = laps_obj
            ## get the output dataframe:
            pos_df: pd.DataFrame = pos.to_dataframe()
            pos_df

        """
        from shapely import box
        from shapely.geometry import LineString, Point

        ## Define the two reward zones
        zone1 = bapun_Day4OpenField_reward_zones.get('zone1', box(-np.inf, 0.0, -60.0, 40.0)) # box(minx, miny, maxx, maxy, ccw=True)
        zone2 = bapun_Day4OpenField_reward_zones.get('zone2', box(80.0, 0.0, np.inf, 40.0)) # box(minx, miny, maxx, maxy, ccw=True)

        pos_df: pd.DataFrame = pos.to_dataframe()

        points = pos_df.apply(lambda row: Point(row['x'], row['y']), axis=1)
        pos_df['zone_id'] = -1 ## initialize column
        is_zone1 = [p.within(zone1) for p in points]
        is_zone2 = [p.within(zone2) for p in points]

        pos_df.loc[is_zone1, 'zone_id'] = 1
        pos_df.loc[is_zone2, 'zone_id'] = 2

        # changes = pos_df['zone_id'].diff()
        pos_df['zone_id_prev_next'] = list(
            zip(
                pos_df['zone_id'].shift(1),
                pos_df['zone_id'].shift(-1)
            )
        )
        # np.unique(pos_df['zone_id_prev_next']) # [(nan, -1.0), (-1.0, -1.0), (-1.0, 1.0), (-1.0, 2.0), (-1.0, nan), (-1.0, -1.0), (-1.0, 1.0), (-1.0, 2.0), (1.0, -1.0), (1.0, 1.0), (2.0, -1.0), (2.0, 2.0)]
        pos._data['zone_id'] = pos_df['zone_id']
        # pos._data['zone_id_prev_next'] = curr_position_df['zone_id_prev_next'] ## this one we don't need to add, it's just for building laps/transitions
        
        # ## Define Zone Enter/Exit times:
        # enter_zone1_times = (pos_df['zone_id_prev_next'] == (-1.0, 1.0))
        # exit_zone1_times = (pos_df['zone_id_prev_next'] == (1.0, -1.0))

        # enter_zone2_times = (pos_df['zone_id_prev_next'] == (-1.0, 2.0))
        # exit_zone2_times = (pos_df['zone_id_prev_next'] == (2.0, -1.0))

        new_lap_epochs_df = []
        last_zone1_exit = None
        last_zone2_exit = None
        last_successful_zone_id = None

        lap_dir_to_lap_dir_integer_mapping = {'L': 0.0, 'R': 1.0}
        for a_row in pos_df.itertuples():
            if a_row.zone_id_prev_next == (-1.0, 1.0):
                ## zone_1_enter
                if (last_successful_zone_id is not None) and (last_successful_zone_id != 1.0) and (last_zone2_exit is not None): ## last condition assumes only 2 zones
                    ## this ends a successful leftward lap
                    new_lap_epochs_df.append({'lap_dir': lap_dir_to_lap_dir_integer_mapping['L'], 'start': last_zone2_exit, 'stop': a_row.t})
                last_successful_zone_id = 1.0
            elif a_row.zone_id_prev_next == (1.0, -1.0):
                ## zone_1_exit
                last_successful_zone_id = 1.0
                last_zone1_exit = a_row.t
            elif a_row.zone_id_prev_next == (-1.0, 2.0):
                ## zone_2_enter
                if (last_successful_zone_id is not None) and (last_successful_zone_id != 2.0) and (last_zone1_exit is not None): ## last condition assumes only 2 zones
                    ## this ends a successful rightward lap
                    new_lap_epochs_df.append({'lap_dir': lap_dir_to_lap_dir_integer_mapping['R'], 'start': last_zone1_exit, 'stop': a_row.t})
                last_successful_zone_id = 2.0
            elif a_row.zone_id_prev_next == (2.0, -1.0):
                ## zone_2_exit
                last_successful_zone_id = 2.0
                last_zone2_exit = a_row.t
            else:
                ## catches all self-transitions
                pass

        ## Build the dataframe:
        new_lap_epochs_df = pd.DataFrame.from_records(new_lap_epochs_df)
        new_lap_epochs_df['duration'] = new_lap_epochs_df['stop'] - new_lap_epochs_df['start']
        new_lap_epochs_df['label'] = new_lap_epochs_df.index.astype(int)
        new_lap_epochs_df['lap_id'] = new_lap_epochs_df.index.astype(int)

        if 'is_LR_dir' not in new_lap_epochs_df.columns:
            new_lap_epochs_df['is_LR_dir'] = (new_lap_epochs_df['lap_dir'] == 0)

        if 'lap' not in new_lap_epochs_df.columns:
            new_lap_epochs_df['lap'] = new_lap_epochs_df['lap_id'].astype(int)

        # new_lap_epochs_df

        new_laps_obj: Laps = Laps(new_lap_epochs_df)
        new_lap_epochs_df = new_laps_obj.to_dataframe()
        # new_laps_obj

        pos_df = pos_df.position.adding_lap_info(laps_df=new_lap_epochs_df, inplace=False)
        ## OUTPUTS: new_laps_obj, pos_df 
        ## UPDATES: pos_df -- added lap, lap_dir

        # update:
        pos._data['lap'] = pos_df['lap']
        pos._data['lap_dir'] = pos_df['lap_dir']
        
        if 'lap_dir_1D' in pos._data:
            pos._data['lap_dir_1D'] = pos_df['lap_dir']
            
        if 'lap_dir_2D' in pos._data:
            pos._data['lap_dir_2D'] = pos_df['lap_dir']
        
        return new_laps_obj, pos


    @classmethod
    def build_Bapun_Day4OpenField_laps_from_reward_zones(cls, session, bapun_Day4OpenField_reward_zones: Dict=None):
        """ builds correct laps (transitions between the two reward zones on the open field maze for the 'roam' experiment
        
        session = curr_active_pipeline.filtered_sessions['roam']
                
        """            
        from neuropy.core import Laps

        if bapun_Day4OpenField_reward_zones is None:
            xmin: float = -85.75619321393464
            xmax: float = 112.57838773103435
            ymin: float = -96.44772761274268
            ymax: float = 98.6220528078153
            bapun_Day4OpenField_grid_bin_bounds = box(xmin, ymin, xmax, ymax)

            bapun_Day4OpenField_reward_zones = dict(    ## Define the two reward zones
                zone1 = box(xmin, 0.0, -60.0, 40.0),  # box(minx, miny, maxx, maxy, ccw=True)
                zone2 = box(80.0, 0.0, xmax, 40.0), # box(minx, miny, maxx, maxy, ccw=True)
            )


        pos: Position = session.position
        laps_obj, pos = cls._perform_build_Bapun_Day4OpenField_laps_from_reward_zones(pos=pos, bapun_Day4OpenField_reward_zones=bapun_Day4OpenField_reward_zones)
        ## Update the current session
        session.position = pos
        session.laps = laps_obj
        ## get the output dataframe:
        pos_df: pd.DataFrame = pos.to_dataframe()
        pos_df

        return session
