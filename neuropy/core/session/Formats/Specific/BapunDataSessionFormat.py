import numpy as np
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available
from pathlib import Path
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec

# For specific load functions:
from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters # used particularly to build Bapun-style filters
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta

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
    _session_default_basedir = r'R:\data\Bapun\Day5TwoNovel' # WINDOWS
    # _session_default_basedir = r'/run/media/halechr/MoverNew/data/Bapun/Day5TwoNovel'
    _session_basepath_to_context_parsing_keys = ['format_name','animal', 'session_name']

    _time_variable_name = 't_seconds' # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data
   
    @classmethod
    def get_session_name(cls, basedir):
        """ returns the session_name for this basedir, which determines the files to load. """
        # Find the only .xml file to obtain the session name
        return DataSessionFormatBaseRegisteredClass.find_session_name_from_sole_xml_file(basedir) # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'

    @classmethod
    def get_session_spec(cls, session_name):
        return SessionFolderSpec(required=[SessionFileSpec('{}.xml', session_name, 'The primary .xml configuration file', cls._load_xml_file),
                                           SessionFileSpec('{}.neurons.npy', session_name, 'The numpy data file containing information about neural activity.', cls._load_neurons_file),
                                           SessionFileSpec('{}.probegroup.npy', session_name, 'The numpy data file containing information about the spatial layout of recording probes', cls._load_probegroup_file),
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
    
    
    # Not limited:
    @classmethod
    def build_filters_any_epochs(cls, sess, filter_name_suffix=None):
        return build_custom_epochs_filters(sess, filter_name_suffix=filter_name_suffix)
    
    # Any epoch on the maze, not limited to pyramidal cells, etc
    @classmethod
    def build_filters_any_maze_epochs(cls, sess, filter_name_suffix=None):
        maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.startswith('maze'), names))
        maze_only_filters = build_custom_epochs_filters(sess, epoch_name_whitelist=maze_only_name_filter_fn, filter_name_suffix=filter_name_suffix)
        return maze_only_filters


    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_whitelist=None, filter_name_suffix=None, include_global_epoch=False):
        ## TODO: currently hard-coded
        # active_session_filter_configurations = cls.build_filters_any_epochs(sess)
        active_session_filter_configurations = cls.build_filters_any_maze_epochs(sess, filter_name_suffix=filter_name_suffix)
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
            session = file_spec.session_load_callback(file_path, session)
            loaded_file_record_list.append(file_path)
        
        # ['.neurons.npy','.probegroup.npy','.position.npy','.paradigm.npy']
        # session = DataSessionLoader.__default_compute_bapun_flattened_spikes(session)
        
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
            
            only_included_pos_computation_labels = ['maze']
            # only_included_pos_computation_labels = None
            if only_included_pos_computation_labels is None:
                only_included_pos_computation_labels = session.epochs.labels # all labels if no restrictions are specified
                

            for an_epoch_label in session.epochs.labels:
                if an_epoch_label in only_included_pos_computation_labels:
                    an_epoch_timeslice_indicies, active_positions_maze1, linearized_positions_curr_epoch = DataSession.compute_linearized_position(session, an_epoch_label)
                    session.position.linear_pos[an_epoch_timeslice_indicies] = linearized_positions_curr_epoch.traces
                
            ## Previous 'manual' maze1 and maze2 way that fails for any sessions without these epochs:    
            # acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, 'maze1')
            # acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, 'maze2')
            # session.position.linear_pos[acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            # session.position.linear_pos[acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            
            session.position.filename = session.filePrefix.with_suffix(".position.npy")
            # print('Saving updated position results to {}...'.format(session.position.filename))
            with ProgressMessagePrinter(session.position.filename, 'Saving', 'updated position results'):
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
    
    
    