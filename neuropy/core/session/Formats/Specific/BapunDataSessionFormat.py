import numpy as np
import pandas as pd
from pathlib import Path
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.data_session_loader import SessionFolderSpec, SessionFileSpec

# For specific load functions:
from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta


class BapunDataSessionFormatRegisteredClass(DataSessionFormatBaseRegisteredClass):
    """

    # Example Filesystem Hierarchy:
    📦Day5TwoNovel
     ┣ 📂position
     ┃ ┣ 📜Take 2020-12-04 02.05.58 PM.csv
     ┃ ┣ 📜Take 2020-12-04 02.13.28 PM.csv
     ┃ ┣ 📜Take 2020-12-04 11.11.32 AM.csv
     ┃ ┗ 📜Take 2020-12-04 11.11.32 AM_001.csv
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.eeg
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.flattened.spikes.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.flattened.spikes.npy.bak
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.maze1.linear.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.maze2.linear.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.mua.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.neurons.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.nrs
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.paradigm.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.pbe.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.position.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.probegroup.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.ripple.npy
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
    
    #######################################################
    ## Bapun Nupy Format Only Methods:
    @staticmethod
    def __default_compute_bapun_flattened_spikes(session, timestamp_scale_factor=(1/1E4)):
        spikes_df = FlattenedSpiketrains.build_spike_dataframe(session)
        session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, t_start=session.neurons.t_start) # FlattenedSpiketrains(spikes_df)
        print('\t Done!')
        return session
    
    ## Main load function:
    @classmethod
    def load_session(cls, session):
        session, loaded_file_record_list = DataSessionFormatBaseRegisteredClass.load_session(session) # call the super class load_session(...) to load the common things (.recinfo, .filePrefix, .eegfile, .datfile)
        remaining_required_filespecs = {k: v for k, v in session.config.resolved_required_filespecs_dict.items() if k not in loaded_file_record_list}
        print(f'remaining_required_filespecs: {remaining_required_filespecs}')
        
        for file_path, file_spec in remaining_required_filespecs.items():
            session = file_spec.session_load_callback(file_path, session)
            loaded_file_record_list.append(file_path)
        
        # ['.neurons.npy','.probegroup.npy','.position.npy','.paradigm.npy']
        #  [fname.format(session_name) for fname in ['{}.xml','{}.neurons.npy','{}.probegroup.npy','{}.position.npy','{}.paradigm.npy']]
        
        # session = DataSessionLoader.__default_compute_bapun_flattened_spikes(session)
        
        # Load or compute linear positions if needed:        
        if (not session.position.has_linear_pos):
            # compute linear positions:
            print('computing linear positions for all active epochs for session...')
            # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
            session.position.linear_pos = np.full_like(session.position.time, np.nan)
            acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, 'maze1')
            acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, 'maze2')
            session.position.linear_pos[acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            session.position.linear_pos[acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
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
        found_datafile = FlattenedSpiketrains.from_file(session.filePrefix.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.flattened_spiketrains = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session = cls.__default_compute_bapun_flattened_spikes(session) # sets session.flattened_spiketrains
            session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix) # '.flattened.spikes.npy'
            print('\t Saving computed flattened spiketrains results to {}...'.format(session.flattened_spiketrains.filename), end='')
            session.flattened_spiketrains.save()
            print('\t done.\n')
        
        # Common Extended properties:
        session = cls._default_extended_postload(session.filePrefix, session)
        
        
        
        
        return session, loaded_file_record_list
    
    
    