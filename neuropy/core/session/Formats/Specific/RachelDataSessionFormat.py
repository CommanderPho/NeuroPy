from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec

# For specific load functions:
from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains, Shank, Probe, ProbeGroup
from neuropy.io import OptitrackIO, PhyIO
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta

from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
from neuropy.utils.result_context import IdentifyingContext

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path
from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager


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
    _session_basepath_to_context_parsing_keys = ['format_name', 'session_name']

    _time_variable_name = 't_seconds' # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data

   
    @classmethod
    def get_session_name(cls, basedir):
        """ returns the session_name for this basedir, which determines the files to load. """
        # Find the only .xml file to obtain the session name
        return DataSessionFormatBaseRegisteredClass.find_session_name_from_sole_xml_file(basedir) # 'merged_M1_20211123_raw'

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
        #     with ProgressMessagePrinter(session.position.filename, 'Saving', 'updated position results'):
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
    
    
    
    ## Initial Function required to wrangle the data from the raw output to a format like Bapun's .npy format:
    @classmethod
    def initialize_data_directory(cls, filepath=Path('/home/halechr/FastData/Rachel/20230614_Rachel'), filename: str = '20230614_Rachel'):
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
