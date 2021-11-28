import sys
import numpy as np
import pandas as pd
from pathlib import Path

from pandas.core import base

from neuropy.core.laps import Laps

# Local imports:
## Core:
from .datawriter import DataWriter
from .neurons import NeuronType, Neurons, BinnedSpiketrain, Mua
from .probe import ProbeGroup
from .position import Position
from .epoch import Epoch #, NamedEpoch
from .signal import Signal

from ..io import NeuroscopeIO, BinarysignalIO # from neuropy.io import NeuroscopeIO, BinarysignalIO

from ..utils.load_exported import import_mat_file
from ..utils.mixins.print_helpers import SimplePrintable, OrderedMeta
from ..utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from ..utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
        
class SessionConfig(SimplePrintable, metaclass=OrderedMeta):
    def __init__(self, basepath, session_spec, session_name):
        """[summary]
        Args:
            basepath (pathlib.Path): [description].
            session_spec (SessionFolderSpec): used to load the files
            session_name (str, optional): [description].
        """
        self.basepath = basepath
        self.session_name = session_name
        # Session spec:
        self.session_spec=session_spec
        self.is_resolved, self.resolved_required_files, self.resolved_optional_files = self.session_spec.validate(self.basepath)


class SessionFolderSpec():
    """ Documents the required and optional files for a given session format """
    def __init__(self, required = [], optional = [], additional_validation_requirements=[]) -> None:
        # additiona_validation_requirements: a list of callbacks that are passed the proposed_session_path on self.validate(...) and return True/False. All must return true for validate to succeed.
        self.required_files = required
        self.optional_files = optional
        self.additional_validation_requirements = additional_validation_requirements
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__dict__};>"


    def resolved_paths(self, proposed_session_path):
        """ Gets whether the proposed_session_path meets the requirements and returns the resolved paths if it can.
            Does not check whether any of the files exist, it just builds the paths
        """
        proposed_session_path = Path(proposed_session_path)
        # build absolute paths from the proposed_session_path and the files
        resolved_required_files = [proposed_session_path.joinpath(a_path) for a_path in self.required_files]
        resolved_optional_files = [proposed_session_path.joinpath(a_path) for a_path in self.optional_files]
        return resolved_required_files, resolved_optional_files
        
    def validate(self, proposed_session_path):
        """Check whether the proposed_session_path meets this folder spec's requirements
        Args:
            proposed_session_path ([Path]): [description]

        Returns:
            [Bool]: [description]
        """
        resolved_required_files, resolved_optional_files = self.resolved_paths(proposed_session_path=proposed_session_path)
            
        meets_spec = False
        if not Path(proposed_session_path).exists():
            meets_spec = False # the path doesn't even exist, it can't be valid
        else:
            # the path exists:
            for a_required_file in resolved_required_files:
                if not a_required_file.exists():
                    print('Required File: {} does not exist.'.format(a_required_file))
                    meets_spec = False
                    break
            for a_required_validation_function in self.additional_validation_requirements:
                if not a_required_validation_function(Path(proposed_session_path)):
                    print('Required additional_validation_requirements[i]({}) returned False'.format(proposed_session_path))
                    meets_spec = False
                    break
            meets_spec = True # otherwise it exists
            
        return True, resolved_required_files, resolved_optional_files
    

# session_name = '2006-6-07_11-26-53'
# SessionFolderSpec(required=['{}.xml'.format(session_name),
#                             '{}.spikeII.mat'.format(session_name), 
#                             '{}.position_info.mat'.format(session_name),
#                             '{}.epochs_info.mat'.format(session_name), 
# ])




class DataSessionLoader:
    """ An extensible class that performs session data loading operations. 
        Data might be loaded into a Session object from many different source formats depending on lab, experimenter, and age of the data.
        Often this data needs to be reverse engineered and translated into the correct format, which is a tedious and time-consuming process.
        This class allows clearly defining and documenting the requirements of a given format once it's been reverse-engineered.
        
        Primary usage methods:
            DataSessionLoader.bapun_data_session(basedir)
            DataSessionLoader.kdiba_old_format_session(basedir)
    """
    # def __init__(self, load_function, load_arguments=dict()):        
    #     self.load_function = load_function
    #     self.load_arguments = load_arguments
        
    # def load(self, updated_load_arguments=None):
    #     if updated_load_arguments is not None:
    #         self.load_arguments = updated_load_arguments
                 
    #     return self.load_function(self.load_arguments)
    
    #######################################################
    ## Public Methods:
    #######################################################
    
    # KDiba Old Format:
    @staticmethod
    def bapun_data_session(basedir):
        def bapun_data_get_session_name(basedir):
            # Find the only .xml file to obtain the session name
            xml_files = sorted(basedir.glob("*.xml"))        
            assert len(xml_files) == 1, "Found more than one .xml file"
            file_prefix = xml_files[0].with_suffix("") # gets the session name (basically) without the .xml extension. (R:\data\Bapun\Day5TwoNovel\RatS-Day5TwoNovel-2020-12-04_07-55-09)   
            file_basename = xml_files[0].stem # file_basename: (RatS-Day5TwoNovel-2020-12-04_07-55-09)
            # print('file_prefix: {}\nfile_basename: {}'.format(file_prefix, file_basename))
            return file_basename # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        def get_session_obj(config):
            curr_args_dict = dict()
            curr_args_dict['basepath'] = config.basepath
            curr_args_dict['session_obj'] = DataSession(config)
            return DataSessionLoader._default_load_bapun_npy_session_folder(curr_args_dict)
            
        session_name = bapun_data_get_session_name(basedir) # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        session_spec = SessionFolderSpec(required=[fname.format(session_name) for fname in ['{}.xml','{}.neurons.npy','{}.probegroup.npy','{}.position.npy','{}.paradigm.npy']])
        session_config = SessionConfig(basedir, session_spec=session_spec, session_name=session_name)
        assert session_config.is_resolved, "active_sess_config could not be resolved!"
        return get_session_obj(session_config)
        
    # KDiba Old Format:
    def kdiba_old_format_session(basedir):
        def kdiba_old_format_get_session_name(basedir):
            return Path(basedir).parts[-1]
        def get_session_obj(config):
            curr_args_dict = dict()
            curr_args_dict['basepath'] = config.basepath
            curr_args_dict['session_obj'] = DataSession(config)
            return DataSessionLoader._default_load_kamran_flat_spikes_mat_session_folder(curr_args_dict)
        session_name = kdiba_old_format_get_session_name(basedir) # session_name = '2006-6-07_11-26-53'
        session_spec = SessionFolderSpec(required=[fname.format(session_name) for fname in ['{}.xml','{}.spikeII.mat','{}.position_info.mat','{}.epochs_info.mat']])
        session_config = SessionConfig(basedir, session_spec=session_spec, session_name=session_name)
        assert session_config.is_resolved, "active_sess_config could not be resolved!"
        return get_session_obj(session_config)
    
    #######################################################
    ## Internal Methods:
    #######################################################
    @staticmethod
    def _default_extended_postload(fp, session):
        # Computes Common Extended properties:
        ## Ripples:
        active_file_suffix = '.ripple.npy'
        found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.ripple = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.ripple = DataSession.compute_neurons_ripples(session)

        ## MUA:
        active_file_suffix = '.mua.npy'
        found_datafile = Mua.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.mua = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.mua = DataSession.compute_neurons_mua(session)

        ## PBE Epochs:
        active_file_suffix = '.pbe.npy'
        found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.pbe = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            session.pbe = DataSession.compute_pbe_epochs(session)
        # return the session with the upadated member variables    
        return session
    
    @staticmethod
    def _default_compute_linear_position_if_needed(session):
        # TODO: this is not general, this is only used for this particular flat kind of file:
            # Load or compute linear positions if needed:
        if (not session.position.has_linear_pos):
            # compute linear positions:
            print('computing linear positions for all active epochs for session...')
            # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
            session.position.computed_traces = np.full([1, session.position.traces.shape[1]], np.nan)
            # acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, epochLabelName='maze', method='pca')
            # session.position.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, epochLabelName='maze1', method='pca')
            acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, epochLabelName='maze2', method='pca')
            session.position.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            session.position.computed_traces[0,  acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            
            session.position.filename = session.filePrefix.with_suffix(".position.npy")
            print('Saving updated position results to {}...'.format(session.position.filename))
            session.position.save()
            print('done.\n')
        else:
            print('linearized position loaded from file.')
        # return the session with the upadated member variables
        return session

    
    @staticmethod
    def _default_load_bapun_npy_session_folder(args_dict):
        basepath = args_dict['basepath']
        session = args_dict['session_obj']
        
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("") # gets the session name (basically) without the .xml extension.
        session.filePrefix = fp
        session.recinfo = NeuroscopeIO(xml_files[0])

        # if session.recinfo.eeg_filename.is_file():
        try:
            session.eegfile = BinarysignalIO(
                session.recinfo.eeg_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.eeg_sampling_rate,
            )
        except ValueError:
            print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(session.recinfo.eeg_filename))
            session.eegfile = None

        if session.recinfo.dat_filename.is_file():
            session.datfile = BinarysignalIO(
                session.recinfo.dat_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.dat_sampling_rate,
            )
        else:
            session.datfile = None

        session.neurons = Neurons.from_file(fp.with_suffix(".neurons.npy"))
        session.probegroup = ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))
        session.position = Position.from_file(fp.with_suffix(".position.npy"))
        
        # ['.neurons.npy','.probegroup.npy','.position.npy','.paradigm.npy']
        #  [fname.format(session_name) for fname in ['{}.xml','{}.neurons.npy','{}.probegroup.npy','{}.position.npy','{}.paradigm.npy']]
        # session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy")) # "epoch" field of file
        session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy")) 
        
        # Load or compute linear positions if needed:        
        if (not session.position.has_linear_pos):
            # compute linear positions:
            print('computing linear positions for all active epochs for session...')
            # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
            session.position.computed_traces = np.full([1, session.position.traces.shape[1]], np.nan)
            acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, 'maze1')
            acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, 'maze2')
            session.position.computed_traces[0,  acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
            session.position.computed_traces[0,  acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
            session.position.filename = session.filePrefix.with_suffix(".position.npy")
            print('Saving updated position results to {}...'.format(session.position.filename))
            session.position.save()
            print('done.\n')
        else:
            print('linearized position loaded from file.')

        # Common Extended properties:
        session = DataSessionLoader._default_extended_postload(fp, session)

        return session # returns the session when done

    @staticmethod
    def _default_load_kamran_position_vt_mat(basepath, session_name, timestamp_scale_factor, spikes_df, session):
        # Loads a *vt.mat file that contains position and epoch information for the session
        session_position_mat_file_path = Path(basepath).joinpath('{}vt.mat'.format(session_name))
        # session.position = Position.from_vt_mat_file(position_mat_file_path=session_position_mat_file_path)
        position_mat_file = import_mat_file(mat_import_file=session_position_mat_file_path)
        tt = position_mat_file['tt'] # 1, 63192
        xx = position_mat_file['xx'] # 10 x 63192
        yy = position_mat_file['yy'] # 10 x 63192
        tt = tt.flatten()
        # tt_rel = tt - tt[0] # relative to start of position file timestamps
        # timestamps_conversion_factor = 1e6
        # timestamps_conversion_factor = 1e4
        # timestamps_conversion_factor = 1.0
        t = tt * timestamp_scale_factor  # (63192,)
        # t_rel = tt_rel * timestamp_scale_factor  # (63192,)
        position_sampling_rate_Hz = 1.0 / np.mean(np.diff(tt / 1e6)) # In Hz, returns 29.969777
        num_samples = len(t)
        x = xx[0,:].flatten() # (63192,)
        y = yy[0,:].flatten() # (63192,)
        # active_t_start = t[0] # absolute t_start
        # active_t_start = 0.0 # relative t_start
        active_t_start = (spikes_df.t.loc[spikes_df.x.first_valid_index()] * timestamp_scale_factor) # actual start time in seconds
        session.position = Position(traces=np.vstack((x, y)), computed_traces=np.full([1, num_samples], np.nan), t_start=active_t_start, sampling_rate=position_sampling_rate_Hz)
        
        # Range of the maze epoch (where position is valid):
        # t_maze_start = spikes_df.t.loc[spikes_df.x.first_valid_index()] # 1048
        # t_maze_end = spikes_df.t.loc[spikes_df.x.last_valid_index()] # 68159707

        t_maze_start = spikes_df.t.loc[spikes_df.x.first_valid_index()] * timestamp_scale_factor # 1048
        t_maze_end = spikes_df.t.loc[spikes_df.x.last_valid_index()] * timestamp_scale_factor # 68159707

        # Note needs to be absolute start/stop times: 
        # t_maze_start = session.position.t_start # 1048
        # t_maze_end = session.position.t_stop # 68159707 68,159,707
        
        # spikes_df.t.min() # 88
        # spikes_df.t.max() # 68624338
        # epochs_df = pd.DataFrame({'start':[0.0, t_maze_start, t_maze_end],'stop':[t_maze_start, t_maze_end, spikes_df.t.max()],'label':['pre','maze','post']})
        # epochs_df = pd.DataFrame({'start':[session.neurons.t_start, t_maze_start, t_maze_end],'stop':[t_maze_start, t_maze_end, session.neurons.t_stop],'label':['pre','maze','post']})
        epochs_df = pd.DataFrame({'start':[0.0, t_maze_start, t_maze_end],'stop':[t_maze_start, t_maze_end, session.neurons.t_stop],'label':['pre','maze','post']})
        
        # session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy")) # "epoch" field of file
        # session.paradigm = Epoch.from_file(fp.with_suffix(".paradigm.npy"))
        session.paradigm = Epoch(epochs=epochs_df)
        
        # return the session with the upadated member variables
        return session
        
    @staticmethod
    def default_load_kamran_IIdata_mat(basepath, session_name, session):
        # Loads a IIdata.mat file that contains position and epoch information for the session
                
        # parent_dir = Path(basepath).parent() # the directory above the individual session folder
        # session_all_dataII_mat_file_path = Path(parent_dir).joinpath('IIdata.mat') # get the IIdata.mat in the parent directory
        # position_all_dataII_mat_file = import_mat_file(mat_import_file=session_all_dataII_mat_file_path)        
        
        ## Epoch Data is loaded first so we can define timestamps relative to the absolute start timestamp
        session_epochs_mat_file_path = Path(basepath).joinpath('{}.epochs_info.mat'.format(session_name))
        epochs_mat_file = import_mat_file(mat_import_file=session_epochs_mat_file_path)
        # ['epoch_data','microseconds_to_seconds_conversion_factor']
        epoch_data_array = epochs_mat_file['epoch_data'] # 
        n_epochs = np.shape(epoch_data_array)[0]
        
        session_absolute_start_timestamp = epoch_data_array[0,0].item()
        epoch_data_array_rel = epoch_data_array - session_absolute_start_timestamp # convert to relative by subtracting the first timestamp
        
        # epochs_df = pd.DataFrame({'start':[epoch_data_array[0,0].item(), epoch_data_array[0,1].item()],'stop':[epoch_data_array[1,0].item(), epoch_data_array[1,1].item()],'label':['maze1','maze2']})
        epochs_df_rel = pd.DataFrame({'start':[epoch_data_array_rel[0,0].item(), epoch_data_array_rel[0,1].item()],'stop':[epoch_data_array_rel[1,0].item(), epoch_data_array_rel[1,1].item()],'label':['maze1','maze2']}) # Use the epochs starting at session_absolute_start_timestamp (meaning the first epoch starts at 0.0
        # session.paradigm = Epoch(epochs=epochs_df)
        session.paradigm = Epoch(epochs=epochs_df_rel)
        
        ## Position Data loaded and zeroed to the same session_absolute_start_timestamp, which starts before the first timestamp in 't':
        session_position_mat_file_path = Path(basepath).joinpath('{}.position_info.mat'.format(session_name))
        position_mat_file = import_mat_file(mat_import_file=session_position_mat_file_path)
        # ['microseconds_to_seconds_conversion_factor','samplingRate', 'timestamps', 'x', 'y']
        t = position_mat_file['timestamps'].squeeze() # 1, 63192
        x = position_mat_file['x'].squeeze() # 10 x 63192
        y = position_mat_file['y'].squeeze() # 10 x 63192
        position_sampling_rate_Hz = position_mat_file['samplingRate'].item() # In Hz, returns 29.969777
        microseconds_to_seconds_conversion_factor = position_mat_file['microseconds_to_seconds_conversion_factor'].item()
        # t_rel = t - t[0] # relative to start of position file timestamps
        t_rel = t - session_absolute_start_timestamp # relative to absolute start of the first epoch
        num_samples = len(t)
        
        active_t_start = t_rel[0] # absolute to first epoch t_start
        # active_t_start = t[0] # absolute t_start
        # active_t_start = 0.0 # relative t_start
        # active_t_start = (spikes_df.t.loc[spikes_df.x.first_valid_index()] * timestamp_scale_factor) # actual start time in seconds
        session.position = Position(traces=np.vstack((x, y)), computed_traces=np.full([1, num_samples], np.nan), t_start=active_t_start, sampling_rate=position_sampling_rate_Hz)
        
        # return the session with the upadated member variables
        return session
    
    @staticmethod
    def _load_kamran_spikeII_mat(sess, timestamp_scale_factor=(1/1E4)):
        spike_mat_file = Path(sess.basepath).joinpath('{}.spikeII.mat'.format(sess.session_name))
        if not spike_mat_file.is_file():
            print('ERROR: file {} does not exist!'.format(spike_mat_file))
            return None
        flat_spikes_mat_file = import_mat_file(mat_import_file=spike_mat_file)
        # print('flat_spikes_mat_file.keys(): {}'.format(flat_spikes_mat_file.keys())) # flat_spikes_mat_file.keys(): dict_keys(['__header__', '__version__', '__globals__', 'spike'])
        flat_spikes_data = flat_spikes_mat_file['spike']
        # print("type is: ",type(flat_spikes_data)) # type is:  <class 'numpy.ndarray'>
        # print("dtype is: ", flat_spikes_data.dtype) # dtype is:  [('t', 'O'), ('shank', 'O'), ('cluster', 'O'), ('aclu', 'O'), ('qclu', 'O'), ('cluinfo', 'O'), ('x', 'O'), ('y', 'O'), ('speed', 'O'), ('traj', 'O'), ('lap', 'O'), ('gamma2', 'O'), ('amp2', 'O'), ('ph', 'O'), ('amp', 'O'), ('gamma', 'O'), ('gammaS', 'O'), ('gammaM', 'O'), ('gammaE', 'O'), ('gamma2S', 'O'), ('gamma2M', 'O'), ('gamma2E', 'O'), ('theta', 'O'), ('ripple', 'O')]
        mat_variables_to_extract = ['t', 'shank', 'cluster', 'aclu', 'qclu', 'cluinfo','x','y','speed','traj','lap']
        num_mat_variables = len(mat_variables_to_extract)
        flat_spikes_out_dict = dict()
        for i in np.arange(num_mat_variables):
            curr_var_name = mat_variables_to_extract[i]
            if curr_var_name == 'cluinfo':
                temp = flat_spikes_data[curr_var_name][0,0] # a Nx4 array
                temp = [tuple(temp[j,:]) for j in np.arange(np.shape(temp)[0])]
                flat_spikes_out_dict[curr_var_name] = temp
            else:
                flat_spikes_out_dict[curr_var_name] = flat_spikes_data[curr_var_name][0,0].flatten() # TODO: do we want .squeeze() instead of .flatten()??
        spikes_df = pd.DataFrame(flat_spikes_out_dict) # 1014937 rows × 11 columns
        spikes_df['cell_type'] = NeuronType.from_qclu_series(qclu_Series=spikes_df['qclu'])
        # add times in seconds both to the dict and the spikes_df under a new key:
        flat_spikes_out_dict['t_seconds'] = flat_spikes_out_dict['t'] * timestamp_scale_factor
        spikes_df['t_seconds'] = spikes_df['t'] * timestamp_scale_factor
        # spikes_df['qclu']
        spikes_df['flat_spike_idx'] = np.array(spikes_df.index)
        return spikes_df, flat_spikes_out_dict 

    @staticmethod
    def _default_spikeII_compute_laps_vars(session, spikes_df, time_variable_name='t_seconds'):
        """ 
        time_variable_name: (str) either 't' or 't_seconds', indicates which time variable to return in 'lap_start_stop_time'
        """
        # Get only the rows with a lap != -1:
        spikes_df = spikes_df[(spikes_df.lap != -1)] # 229887 rows × 13 columns
        # Group by the lap column:
        lap_grouped_spikes_df = spikes_df.groupby(['lap']) #  as_index=False keeps the original index
        laps_first_spike_instances = lap_grouped_spikes_df.first()
        laps_last_spike_instances = lap_grouped_spikes_df.last()

        lap_id = np.array(laps_first_spike_instances.index) # the lap_id (which serves to index the lap), like 1, 2, 3, 4, ...
        laps_spike_counts = np.array(lap_grouped_spikes_df.size().values) # number of spikes in each lap

        # print('lap_number: {}'.format(lap_number))
        # print('laps_spike_counts: {}'.format(laps_spike_counts))
        first_indicies = np.array(laps_first_spike_instances.t.index)
        num_laps = len(first_indicies)

        lap_start_stop_flat_idx = np.empty([num_laps, 2])
        lap_start_stop_flat_idx[:, 0] = np.array(laps_first_spike_instances.flat_spike_idx.values)
        lap_start_stop_flat_idx[:, 1] = np.array(laps_last_spike_instances.flat_spike_idx.values)
        # print('lap_start_stop_flat_idx: {}'.format(lap_start_stop_flat_idx))

        lap_start_stop_time = np.empty([num_laps, 2])
        lap_start_stop_time[:, 0] = np.array(laps_first_spike_instances[time_variable_name].values)
        lap_start_stop_time[:, 1] = np.array(laps_last_spike_instances[time_variable_name].values)
        # print('lap_start_stop_time: {}'.format(lap_start_stop_time))
        
        # Build output Laps object to add to session
        session.laps = Laps(lap_id, laps_spike_counts, lap_start_stop_flat_idx, lap_start_stop_time)
        
        # return lap_id, laps_spike_counts, lap_start_stop_flat_idx, lap_start_stop_time
        return session
        
    @staticmethod
    def __default_spikeII_compute_neurons(session, spikes_df, flat_spikes_out_dict, time_variable_name='t_seconds'):
        ## Get unique cell ids to enable grouping flattened results by cell:
        unique_cell_ids = np.unique(flat_spikes_out_dict['aclu'])
        flat_cell_ids = [int(cell_id) for cell_id in unique_cell_ids]
        num_unique_cell_ids = len(flat_cell_ids)
        # print('flat_cell_ids: {}'.format(flat_cell_ids))
        # Group by the aclu (cluster indicator) column
        cell_grouped_spikes_df = spikes_df.groupby(['aclu'])
        spiketrains = list()
        shank_ids = np.zeros([num_unique_cell_ids, ]) # (108,) Array of float64
        cell_quality = np.zeros([num_unique_cell_ids, ]) # (108,) Array of float64
        cell_type = list() # (108,) Array of float64

        for i in np.arange(num_unique_cell_ids):
            curr_cell_id = flat_cell_ids[i] # actual cell ID
            #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
            curr_cell_dataframe = cell_grouped_spikes_df.get_group(curr_cell_id)
            spiketrains.append(curr_cell_dataframe[time_variable_name].to_numpy())
            shank_ids[i] = curr_cell_dataframe['shank'].to_numpy()[0] # get the first shank identifier, which should be the same for all of this curr_cell_id
            cell_quality[i] = curr_cell_dataframe['qclu'].mean() # should be the same for all instances of curr_cell_id, but use mean just to make sure
            cell_type.append(curr_cell_dataframe['cell_type'].to_numpy()[0])

        spiketrains = np.array(spiketrains, dtype='object')
        t_stop = np.max(flat_spikes_out_dict[time_variable_name])
        flat_cell_ids = np.array(flat_cell_ids)
        cell_type = np.array(cell_type)
        session.neurons = Neurons(spiketrains, t_stop, t_start=0,
            sampling_rate=session.recinfo.dat_sampling_rate,
            neuron_ids=flat_cell_ids,
            neuron_type=cell_type,
            shank_ids=shank_ids
        )
        return session




    @staticmethod
    def _default_load_kamran_flat_spikes_mat_session_folder(args_dict):
        basepath = args_dict['basepath']
        session = args_dict['session_obj']
        # timestamp_scale_factor = (1/1E6)
        timestamp_scale_factor = (1/1E4)
        
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("")
        session.filePrefix = fp
        session.recinfo = NeuroscopeIO(xml_files[0])

        # if session.recinfo.eeg_filename.is_file():
        try:
            session.eegfile = BinarysignalIO(
                session.recinfo.eeg_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.eeg_sampling_rate,
            )
        except ValueError:
            print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(session.recinfo.eeg_filename))
            session.eegfile = None

        if session.recinfo.dat_filename.is_file():
            session.datfile = BinarysignalIO(
                session.recinfo.dat_filename,
                n_channels=session.recinfo.n_channels,
                sampling_rate=session.recinfo.dat_sampling_rate,
            )
        else:
            session.datfile = None
            
        session_name = basepath.parts[-1]
        print('\t basepath: {}\n\t session_name: {}'.format(basepath, session_name)) # session_name: 2006-6-08_14-26-15


        ## .spikeII.mat file:
        spikes_df, flat_spikes_out_dict = DataSessionLoader._load_kamran_spikeII_mat(session, timestamp_scale_factor=timestamp_scale_factor)
        
        # active_time_variable_name = 't' # default
        active_time_variable_name = 't_seconds' # use converted times (into seconds)
        
        # for debugging purposes, add spikes_df to the session
        session.spikes_df = spikes_df
        
        ## Laps:
        # lap_number, laps_spike_counts, lap_start_stop_flat_idx, lap_start_stop_time = DataSessionLoader._default_spikeII_compute_laps_vars(spikes_df, active_time_variable_name)
        session = DataSessionLoader._default_spikeII_compute_laps_vars(session, spikes_df, active_time_variable_name)
        
        ## Neurons (by Cell):
        session = DataSessionLoader.__default_spikeII_compute_neurons(session, spikes_df, flat_spikes_out_dict, active_time_variable_name)
          
        session.probegroup = ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))
        
        # *vt.mat file Position and Epoch:
        # session = DataSessionLoader.default_load_kamran_position_vt_mat(basepath, session_name, timestamp_scale_factor, spikes_df, session)
        
        # IIdata.mat file Position and Epoch:
        session = DataSessionLoader.default_load_kamran_IIdata_mat(basepath, session_name, session)
        
        # Load or compute linear positions if needed:
        try:
            session = DataSessionLoader._default_compute_linear_position_if_needed(session)
        except Exception as e:
            # raise e
            print('session.position linear positions could not be computed due to error {}. Skipping.'.format(e))
            session.position.computed_traces = np.full([1, session.position.traces.shape[1]], np.nan)
        else:
            # Successful!
            print('session.position linear positions computed!')
            pass

        # Common Extended properties:
        # session = DataSessionLoader.default_extended_postload(fp, session)
        
        session.is_loaded = True # indicate the session is loaded
        
        return session # returns the session when done


class DataSession(NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, TimeSlicableObjectProtocol):
    def __init__(self, config, filePrefix = None, recinfo = None,
                 eegfile = None, datfile = None,
                 neurons = None, probegroup = None, position = None, paradigm = None,
                 ripple = None, mua = None, laps= None):        
        self.config = config
        
        self.is_loaded = False
        self.filePrefix = filePrefix
        self.recinfo = recinfo
        
        self.eegfile = eegfile
        self.datfile = datfile
        
        self.neurons = neurons
        self.probegroup = probegroup
        self.position = position
        self.paradigm = paradigm
        self.ripple = ripple
        self.mua = mua
        self.laps = laps # core.laps.Laps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"
    #######################################################
    ## Passthru Accessor Properties:
    @property
    def is_resolved(self):
        return self.config.is_resolved
    @property
    def basepath(self):
        return self.config.basepath
    @property
    def session_name(self):
        return self.config.session_name
    @property
    def name(self):
        return self.session_name
    @property
    def resolved_files(self):
        return (self.config.resolved_required_files + self.config.resolved_optional_files)

    @property
    def position_sampling_rate(self):
        return self.position.sampling_rate

    @property
    def neuron_ids(self):
        return self.neurons.neuron_ids
    
    @property
    def n_neurons(self):
        return self.neurons.n_neurons
    # @property
    # def is_resolved(self):
    #     return self.config.is_resolved
    # @property
    # def is_resolved(self):
    #     return self.config.is_resolved
    # @property
    # def is_resolved(self):
    #     return self.config.is_resolved
    
    
    # @property
    # def is_loaded(self):
    #     """The epochs property is an alias for self.paradigm."""
    #     return self.paradigm
    
    @property
    def epochs(self):
        """The epochs property is an alias for self.paradigm."""
        return self.paradigm
    @epochs.setter
    def epochs(self, value):
        self.paradigm = value
        
    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
        active_epoch_times = [t_start, t_stop]
        print('Constraining to epoch with times (start: {}, end: {})'.format(active_epoch_times[0], active_epoch_times[1]))
        # make a copy of self:
        # should implement __deepcopy__() and __copy__()??
        copy_sess = DataSession.from_dict(self.to_dict())
        # update the copy_session's time_sliceable objects
        copy_sess.neurons = self.neurons.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_session_Neurons: Filter by pyramidal cells only, returns a core.
        copy_sess.position = self.position.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_pos: active_epoch_pos's .time and start/end are all valid        
        return copy_sess
    

    def get_neuron_type(self, query_neuron_type):
        """ filters self by the specified query_neuron_type, only returning neurons that match. """
        print('Constraining to units with type: {}'.format(query_neuron_type))
        # make a copy of self:
        copy_sess = DataSession.from_dict(self.to_dict())
        # update the copy_session's neurons objects
        copy_sess.neurons = self.neurons.get_neuron_type(query_neuron_type) # active_epoch_session_Neurons: Filter by pyramidal cells only, returns a core.
        return copy_sess
    
    

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids):
        """Implementors return a copy of themselves with neuron_ids equal to ids"""
        copy_sess = DataSession.from_dict(self.to_dict())
        copy_sess.neurons = self.neurons.get_by_id(ids)
        return copy_sess

  
    @staticmethod
    def from_dict(d: dict):
        return DataSession(d['config'], filePrefix = d['filePrefix'], recinfo = d['recinfo'],
                 eegfile = d['eegfile'], datfile = d['datfile'],
                 neurons = d['neurons'], probegroup = d.get('probegroup', None), position = d['position'], paradigm = d['paradigm'],
                 ripple = d.get('ripple', None), mua = d.get('mua', None))
        
        
    def to_dict(self, recurrsively=False):
        simple_dict = self.__dict__
        if recurrsively:
            simple_dict['paradigm'] = simple_dict['paradigm'].to_dict()
            simple_dict['position'] = simple_dict['position'].to_dict()
            simple_dict['neurons'] = simple_dict['neurons'].to_dict()        
        return simple_dict
        
    ## Linearize Position:
    @staticmethod
    def compute_linearized_position(session, epochLabelName='maze1', method='isomap'):
        # returns Position objects for active_epoch_pos and linear_pos
        from neuropy.utils import position_util
        active_epoch_times = session.epochs[epochLabelName] # array([11070, 13970], dtype=int64)
        acitve_epoch_timeslice_indicies = session.position.time_slice_indicies(active_epoch_times[0], active_epoch_times[1])
        active_epoch_pos = session.position.time_slice(active_epoch_times[0], active_epoch_times[1])
        linear_pos = position_util.linearize_position(active_epoch_pos, method=method)
        return acitve_epoch_timeslice_indicies, active_epoch_pos, linear_pos
    #  acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = compute_linearized_position(sess, 'maze1')
    #  acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = compute_linearized_position(sess, 'maze2')

    ## Ripple epochs
    #   To detect ripples one also needs probegroup.
    @staticmethod
    def compute_neurons_ripples(session):
        print('computing ripple epochs for session...\n')
        from neuropy.analyses import oscillations
        signal = session.eegfile.get_signal()
        ripple_epochs = oscillations.detect_ripple_epochs(signal, session.probegroup)
        ripple_epochs.filename = session.filePrefix.with_suffix('.ripple.npy')
        print('Saving ripple epochs results to {}...'.format(ripple_epochs.filename))
        ripple_epochs.save()
        print('done.\n')
        return ripple_epochs
    # sess.ripple = compute_neurons_ripples(sess)

    ## BinnedSpiketrain and Mua objects using Neurons
    @staticmethod
    def compute_neurons_mua(session):
        print('computing neurons mua for session...\n')
        mua = session.neurons.get_mua()
        mua.filename = session.filePrefix.with_suffix(".mua.npy")
        print('Saving mua results to {}...'.format(mua.filename))
        mua.save()
        print('done.\n')
        return mua    
    # sess.mua = compute_neurons_mua(sess) # Set the .mua field on the session object once complete

    @staticmethod
    def compute_pbe_epochs(session):
        from neuropy.analyses import detect_pbe_epochs
        print('computing PBE epochs for session...\n')
        smth_mua = session.mua.get_smoothed(sigma=0.02) # Get the smoothed mua from the session's mua
        pbe = detect_pbe_epochs(smth_mua)
        pbe.filename = session.filePrefix.with_suffix('.pbe.npy')
        print('Saving pbe results to {}...'.format(pbe.filename))
        pbe.save()
        print('done.\n')
        return pbe
    # sess.pbe = compute_pbe_epochs(sess)



# Helper function that processed the data in a given directory
def processDataSession(basedir='/Volumes/iNeo/Data/Bapun/Day5TwoNovel'):
    # sess = DataSession(basedir)
    curr_args_dict = dict()
    curr_args_dict['basepath'] = basedir
    curr_args_dict['session_obj'] = DataSession() # Create an empty session object
    sess = DataSessionLoader._default_load_bapun_npy_session_folder(curr_args_dict)
    return sess


## Main:
if __name__ == "__main__":
    # Now initiate the class
    # basedir = '/data/Working/Opto/Jackie671/Jackie_placestim_day2/Jackie_TRACK_2020-10-07_11-21-39'  # fill in here
    basedir = 'R:\data\Bapun\Day5TwoNovel'
    # basedir = '/Volumes/iNeo/Data/Bapun/Day5TwoNovel'
    sess = processDataSession(basedir)
    print(sess.recinfo)
    sess.epochs.to_dataframe()
    sess.neurons.get_all_spikes()
    sess.position.sampling_rate # 60
    
    pass
