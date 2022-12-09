from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
from neuropy.core.position import Position
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec, SessionConfig

# For specific load functions:
from neuropy.core import Mua, Epoch
from neuropy.core.epoch import NamedTimerange # required for DataSessionFormatBaseRegisteredClass.build_global_filter_config_function(.)
from neuropy.io import NeuroscopeIO, BinarysignalIO 
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.utils.dynamic_container import DynamicContainer, override_dict, overriding_dict_with, get_dict_subset
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter
from neuropy.utils.position_util import compute_position_grid_size
from neuropy.utils.result_context import IdentifyingContext


class DataSessionFormatRegistryHolder(type):
    """ a metaclass that automatically registers its conformers as a known loadable data session format.     
        
    Usage:
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
        from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
        from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
        from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
        from neuropy.core.session.Formats.Specific.HiroDataSessionFormat import HiroDataSessionFormatRegisteredClass

        DataSessionFormatRegistryHolder.get_registry()
        
    """
    REGISTRY: Dict[str, "DataSessionFormatRegistryHolder"] = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        """
            Here the name of the class is used as key but it could be any class
            parameter.
        """
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
    
    @classmethod
    def get_registry_data_session_type_class_name_dict(cls):
        """ returns a dict<str, DataSessionFormatBaseRegisteredClass> with keys corresponding to the registered short-names of the data_session_type (like 'kdiba', or 'bapun') and values of DataSessionFormatBaseRegisteredClass. """
        return {a_class._session_class_name:a_class for a_class_name, a_class in cls.get_registry().items() if a_class_name != 'DataSessionFormatBaseRegisteredClass'}
    
    
    
    @classmethod
    def get_registry_known_data_session_type_dict(cls, override_data_basepath=None):
        """ returns a dict<str, KnownDataSessionTypeProperties> with keys corresponding to the registered short-names of the data_session_type (like 'kdiba', or 'bapun') and values of KnownDataSessionTypeProperties. """
        return {a_class._session_class_name:a_class.get_known_data_session_type_properties(override_basepath=override_data_basepath) for a_class_name, a_class in cls.get_registry().items() if a_class_name != 'DataSessionFormatBaseRegisteredClass'}
    
    


class DataSessionFormatBaseRegisteredClass(metaclass=DataSessionFormatRegistryHolder):
    """
    Any class that will inherits from DataSessionFormatBaseRegisteredClass will be included
    inside the dict RegistryHolder.REGISTRY, the key being the name of the
    class and the associated value, the class itself.
    
    The user specifies a basepath, which is the path containing a list of files:
    
    📦Day5TwoNovel
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.eeg
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.mua.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.neurons.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.paradigm.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.pbe.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.position.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.probegroup.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.ripple.npy
     ┗ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.xml
    
    
    By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file:
        basedir: Path(r'R:\data\Bapun\Day5TwoNovel')
        session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
    
    From here, a list of known files to load from is determined:
        
    """
    _session_class_name = 'base'
    _session_default_relative_basedir = r'data\KDIBA\gor01\one\2006-6-07_11-26-53'
    _session_default_basedir = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53'
    
    _time_variable_name = None # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data

    @classmethod
    def get_session_format_name(cls):
        """ The name of the specific format (e.g. 'bapun', 'kdiba', etc) """
        return cls._session_class_name
    
    @classmethod
    def get_session_name(cls, basedir):
        """ MUST be overriden by implementor to return the session_name for this basedir, which determines the files to load. """
        raise NotImplementedError # innheritor must override

    @classmethod
    def get_session_spec(cls, session_name):
        """ MUST be overriden by implementor to return the a session_spec """
        raise NotImplementedError # innheritor must override
        
    @classmethod
    def build_global_epoch_filter_config_dict(cls, sess, global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None, debug_print=False):
        """ builds the 'global' filter for the entire session that includes by default the times from all other epochs in sess. 
        e.g. builds the 'maze' epoch from ['maze1', 'maze2'] epochs

        Usage:
            global_epoch_filter_fn_dict, global_named_timerange = build_global_epoch_filter_config_dict(sess, global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None, debug_print=True)
            global_epoch_filter_fn_dict

        """
        all_epoch_names = list(sess.epochs.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
        if global_epoch_name in all_epoch_names:
            global_epoch_name = f"{global_epoch_name}_GLOBAL"
            print('WARNING: name collision "{global_epoch_name}" already exists in all_epoch_names: {all_epoch_names}! Using {global_epoch_name} instead.')
        
        if first_included_epoch_name is not None:
            # global_start_end_times[0] = sess.epochs[first_included_epoch_name][0] # 'maze1'
            pass
        else:
            first_included_epoch_name = sess.epochs.get_unique_labels()[0]
            
        if last_included_epoch_name is not None:
            # global_start_end_times[1] = sess.epochs[last_included_epoch_name][1] # 'maze2'
            pass
        else:
            last_included_epoch_name = sess.epochs.get_unique_labels()[-1]
    
        # global_start_end_times = [epochs.t_start, epochs.t_stop]
        global_start_end_times = [sess.epochs[first_included_epoch_name][0], sess.epochs[last_included_epoch_name][1]]
        # global_start_end_times_fn = lambda x: [sess.epochs[first_included_epoch_name][0], sess.epochs[last_included_epoch_name][1]]
        
        global_named_timerange = NamedTimerange(name=global_epoch_name, start_end_times=global_start_end_times)
        # global_epoch_filter_fn = (lambda x: (x.filtered_by_epoch(NamedTimerange(name=global_epoch_name, start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])), NamedTimerange(name=global_epoch_name, start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])))
        if debug_print:
            print(f'global_named_timerange: {global_named_timerange}, first_included_epoch_name: {first_included_epoch_name}, last_included_epoch_name: {last_included_epoch_name}')
        global_epoch_filter_fn = (lambda x: (x.filtered_by_epoch(NamedTimerange(name=global_epoch_name, start_end_times=[x.epochs[(first_included_epoch_name or x.epochs.get_unique_labels()[0])][0], x.epochs[(last_included_epoch_name or x.epochs.get_unique_labels()[-1])][1]])), NamedTimerange(name=global_epoch_name, start_end_times=[x.epochs[(first_included_epoch_name or x.epochs.get_unique_labels()[0])][0], x.epochs[(last_included_epoch_name or x.epochs.get_unique_labels()[-1])][1]])))
        return {global_epoch_name: global_epoch_filter_fn}, global_named_timerange
    
    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_whitelist=None, filter_name_suffix=None, include_global_epoch=True):
        """ OPTIONALLY can be overriden by implementors to provide specific filter functions
        Inputs:
            epoch_name_whitelist: an optional list of names to restrict to, must already be valid epochs to filter by. e.g. ['maze1']
            filter_name_suffix: an optional string suffix to be added to the end of each filter_name. An example would be '_PYR'
            include_global_epoch: bool - If True, uses cls.build_global_epoch_filter_config_dict(...) to generate a global epoch that will be included in the filters
        """
        if epoch_name_whitelist is None:
            all_epoch_names = list(sess.epochs.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
            epoch_name_whitelist = all_epoch_names
            
        if filter_name_suffix is None:
            filter_name_suffix = ''

        if include_global_epoch:
            global_epoch_filter_fn_dict, global_named_timerange = cls.build_global_epoch_filter_config_dict(sess, global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None, debug_print=False)
        else:
            global_epoch_filter_fn_dict = {} # empty dict

        epoch_filter_configs_dict = {f'{an_epoch_name}{filter_name_suffix}':lambda a_sess, epoch_name=an_epoch_name: (a_sess.filtered_by_epoch(a_sess.epochs.get_named_timerange(epoch_name)), a_sess.epochs.get_named_timerange(epoch_name)) for an_epoch_name in epoch_name_whitelist}
        final_configs_dict = dict(epoch_filter_configs_dict, **global_epoch_filter_fn_dict)
        return  final_configs_dict

    @classmethod
    def build_default_computation_configs(cls, sess, **kwargs):
        """ OPTIONALLY can be overriden by implementors to provide specific filter functions """
        kwargs.setdefault('pf_params', PlacefieldComputationParameters(**override_dict({'speed_thresh': 10.0, 'grid_bin': cls.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), 'grid_bin_bounds': None, 'smooth': (2.0, 2.0), 'frate_thresh': 0.2, 'time_bin_size': 0.025, 'computation_epochs': None}, kwargs)))
        kwargs.setdefault('spike_analysis', DynamicContainer(**{'max_num_spikes_per_neuron': 20000,
                                                                 'kleinberg_parameters': DynamicContainer(**{'s': 2, 'gamma': 0.2}).override(kwargs),
                                                                 'use_progress_bar': False,
                                                                 'debug_print': False}).override(kwargs))        
        return [DynamicContainer(pf_params=kwargs['pf_params'], spike_analysis=kwargs['spike_analysis'])]
        # return [DynamicContainer(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=cls.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None),
        #                   spike_analysis=DynamicContainer(max_num_spikes_per_neuron=20000, kleinberg_parameters=DynamicContainer(s=2, gamma=0.2), use_progress_bar=False, debug_print=False))]
        # active_grid_bin = compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64))
        # active_session_computation_config.computation_epochs = None # set the placefield computation epochs to None, using all epochs.
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(128, 128)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(3.777, 1.043), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(32, 32)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #         PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #         PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(128, 128)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #        ]
  
  
    @classmethod
    def get_session(cls, basedir):
        _test_session = cls.build_session(Path(basedir))
        _test_session, loaded_file_record_list = cls.load_session(_test_session)
        return _test_session    
    
    @classmethod
    def find_session_name_from_sole_xml_file(cls, basedir, debug_print=False):
        """ By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file
        Example:
            basedir: Path(r'R:\data\Bapun\Day5TwoNovel')
            session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        """
        # Find the only .xml file to obtain the session name 
        xml_files = sorted(basedir.glob("*.xml"))
        assert len(xml_files) > 0, "Missing required .xml file!"
        assert len(xml_files) == 1, f"Found more than one .xml file. Found files: {xml_files}"
        file_prefix = xml_files[0].with_suffix("") # gets the session name (basically) without the .xml extension. (R:\data\Bapun\Day5TwoNovel\RatS-Day5TwoNovel-2020-12-04_07-55-09)   
        file_basename = xml_files[0].stem # file_basename: (RatS-Day5TwoNovel-2020-12-04_07-55-09)
        if debug_print:
            print('file_prefix: {}\nfile_basename: {}'.format(file_prefix, file_basename))
        return file_basename # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'


    @classmethod
    def get_session_basepath_to_context_parsing_keys(cls):
        """ Just a wrapper to access the cls._session_basepath_to_context_parsing_keys property
        Used only by `parse_session_basepath_to_context(.)`
        """
        return cls._session_basepath_to_context_parsing_keys



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
        data_index = tuple(map(str.casefold, dir_parts)).index('DATA'.casefold()) # .casefold is equivalent to .lower, but works for unicode characters
        post_data_root_dir_parts = dir_parts[data_index+1:] # ('KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')
        num_parts = len(post_data_root_dir_parts)
        context_keys = cls.get_session_basepath_to_context_parsing_keys()
        assert len(context_keys) == num_parts
        context_kwargs_dict = dict(zip(context_keys, post_data_root_dir_parts))
        curr_sess_ctx = IdentifyingContext(**context_kwargs_dict)
        # want to replace the 'format_name' with the one known for this session (e.g. 'KDIBA' vs. 'kdiba')
        format_name = cls.get_session_format_name() 
        curr_sess_ctx.format_name = format_name
        return curr_sess_ctx # IdentifyingContext<('KDIBA', 'gor01', 'one', '2006-6-07_11-26-53')>


    @classmethod
    def get_known_data_session_type_properties(cls, override_basepath=None):
        """ returns the KnownDataSessionTypeProperties for this class, which contains information about the process of loading the session."""
        if override_basepath is not None:
            basepath = override_basepath
        else:
            basepath = Path(cls._session_default_basedir)
        return KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: cls.get_session(basedir=a_base_dir)), basedir=basepath)
        
    @classmethod
    def build_session(cls, basedir):
        basedir = Path(basedir)
        session_name = cls.get_session_name(basedir) # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
        session_context = cls.parse_session_basepath_to_context(basedir) 
        session_spec = cls.get_session_spec(session_name)
        format_name = cls.get_session_format_name()
        session_config = SessionConfig(basedir, format_name=format_name, session_spec=session_spec, session_name=session_name, session_context=session_context)
        assert session_config.is_resolved, "active_sess_config could not be resolved!"
        session_obj = DataSession(session_config)
        return session_obj
    
    @classmethod
    def load_session(cls, session, debug_print=False):
        # .recinfo, .filePrefix, .eegfile, .datfile
        loaded_file_record_list = [] # Handled files list
        
        try:
            _test_xml_file_path, _test_xml_file_spec = list(session.config.resolved_required_filespecs_dict.items())[0]
            session = _test_xml_file_spec.session_load_callback(_test_xml_file_path, session)
            loaded_file_record_list.append(_test_xml_file_path)
        except IndexError as e:
            # No XML file can be found, so instead check for a dynamically provided rec_info
            session = cls._fallback_recinfo(None, session)
            # raise e
                
        # Now have access to proper session.recinfo.dat_filename and session.recinfo.eeg_filename:
        session.config.session_spec.optional_files.insert(0, SessionFileSpec('{}'+session.recinfo.dat_filename.suffix, session.recinfo.dat_filename.stem, 'The .dat binary data file', cls._load_datfile))
        session.config.session_spec.optional_files.insert(0, SessionFileSpec('{}'+session.recinfo.eeg_filename.suffix, session.recinfo.eeg_filename.stem, 'The .eeg binary data file', cls._load_eegfile))
        session.config.validate()
        
        _eeg_file_spec = session.config.resolved_optional_filespecs_dict[session.recinfo.eeg_filename]
        session = _eeg_file_spec.session_load_callback(session.recinfo.eeg_filename, session)
        loaded_file_record_list.append(session.recinfo.eeg_filename)
        
        _dat_file_spec = session.config.resolved_optional_filespecs_dict[session.recinfo.dat_filename]
        session = _dat_file_spec.session_load_callback(session.recinfo.dat_filename, session)
        loaded_file_record_list.append(session.recinfo.dat_filename)
        
        return session, loaded_file_record_list
        
    #######################################################
    ## Internal Methods:
    #######################################################
            
    @classmethod
    def compute_position_grid_bin_size(cls, x, y, num_bins=(64,64), debug_print=False):
        """ Compute Required Bin size given a desired number of bins in each dimension
        Usage:
            active_grid_bin = compute_position_grid_bin_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64)
        """
        out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(x, y, num_bins=num_bins)
        active_grid_bin = tuple(out_grid_bin_size)
        if debug_print:
            print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
        return active_grid_bin
    
    @classmethod
    def _default_compute_spike_interpolated_positions_if_needed(cls, session, spikes_df, time_variable_name='t_rel_seconds', force_recompute=True):     
        ## Positions:
        active_file_suffix = '.interpolated_spike_positions.npy'
        if not force_recompute:
            found_datafile = FlattenedSpiketrains.from_file(session.filePrefix.with_suffix(active_file_suffix))
        else:
            found_datafile = None
        if found_datafile is not None:
            print('\t Loading success: {}.'.format(active_file_suffix))
            session.flattened_spiketrains = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            if force_recompute:
                print(f'\t force_recompute is True! Forcing recomputation of {active_file_suffix}\n')
            else:
                print(f'\t Failure loading "{session.filePrefix.with_suffix(active_file_suffix)}". Must recompute.\n')
            with ProgressMessagePrinter('spikes_df', 'Computing', 'interpolate_spike_positions columns'):
                spikes_df = FlattenedSpiketrains.interpolate_spike_positions(spikes_df, session.position.time, session.position.x, session.position.y, position_linear_pos=session.position.linear_pos, position_speeds=session.position.speed, spike_timestamp_column_name=time_variable_name)
                session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=time_variable_name, t_start=0.0)
            
            session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix)
            # print('\t Saving updated interpolated spike position results to {}...'.format(session.flattened_spiketrains.filename), end='')
            with ProgressMessagePrinter(session.flattened_spiketrains.filename, '\t Saving', 'updated interpolated spike position results'):
                session.flattened_spiketrains.save()
            # print('\t done.\n')
    
        # return the session with the upadated member variables
        return session, spikes_df
    
    @classmethod
    def _default_add_spike_PBEs_if_needed(cls, session):
        with ProgressMessagePrinter('spikes_df', 'Computing', 'spikes_df PBEs column'):
            updated_spk_df = session.compute_spikes_PBEs()
        return session
    
    @classmethod
    def _default_add_spike_scISIs_if_needed(cls, session):
        with ProgressMessagePrinter('spikes_df', 'Computing', 'added spike scISI column'):
            updated_spk_df = session.spikes_df.spikes.add_same_cell_ISI_column()
        return session
    
    
    @classmethod
    def _default_compute_flattened_spikes(cls, session, timestamp_scale_factor=(1/1E4), spike_timestamp_column_name='t_seconds', progress_tracing=True):
        """ builds the session.flattened_spiketrains (and therefore spikes_df) from the session.neurons object. """
        spikes_df = FlattenedSpiketrains.build_spike_dataframe(session, timestamp_scale_factor=timestamp_scale_factor, spike_timestamp_column_name=spike_timestamp_column_name, progress_tracing=progress_tracing)
        print(f'spikes_df.columns: {spikes_df.columns}')
        session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=spike_timestamp_column_name, t_start=session.neurons.t_start) # FlattenedSpiketrains(spikes_df)
        print('\t Done!')
        return session
    
    @classmethod
    def _add_missing_spikes_df_columns(cls, spikes_df, neurons_obj):
        """ adds the 'fragile_linear_neuron_IDX' column to the spikes_df and updates the neurons_obj with a new reverse_cellID_index_map """
        spikes_df, neurons_obj._reverse_cellID_index_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        spikes_df['t'] = spikes_df[cls._time_variable_name] # add the 't' column required for visualization
 
    @classmethod
    def _default_extended_postload(cls, fp, session):
        # Computes Common Extended properties:
        ## Ripples:
        # Externally Computed Ripples (from 'ripple_df.pkl') file:
        # Load `ripple_df.pkl` previously saved:
        external_computed_ripple_df_filepath = session.basepath.joinpath('ripple_df.pkl')
        external_computed_ripple_df = pd.read_pickle(external_computed_ripple_df_filepath)
        # Add the required columns for Epoch(...):
        external_computed_ripple_df['label'] = [str(an_idx) for an_idx in external_computed_ripple_df.index]
        external_computed_ripple_df = external_computed_ripple_df.reset_index(drop=True)
        found_datafile = Epoch(external_computed_ripple_df) # Epoch from dataframe
        
        if found_datafile is not None:
            print('Loading success: {}.'.format(external_computed_ripple_df_filepath))
            session.ripple = found_datafile
            found_datafile.filename = external_computed_ripple_df_filepath

        else:
            ## try the '.ripple.npy' ripples:
            active_file_suffix = '.ripple.npy'
            found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
            if found_datafile is not None:
                print('Loading success: {}.'.format(active_file_suffix))
                session.ripple = found_datafile
                # ## TODO: overwrite the '.ripple.npy' version?
                # session.ripple.filename = session.filePrefix.with_suffix('.ripple.npy')
                # # print_file_progress_message(ripple_epochs.filename, 'Saving', 'ripple epochs')
                # with ProgressMessagePrinter(session.ripple.filename, 'Saving', 'ripple epochs'):
                #     session.ripple.save()
            else:
                # Otherwise both loads failed, perform the fallback computation:
                print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
                try:
                    session.ripple = DataSession.compute_neurons_ripples(session, save_on_compute=True)
                except (ValueError, AttributeError) as e:
                    print(f'Computation failed with error {e}. Skipping .ripple')
                    session.ripple = None

        ## MUA:
        active_file_suffix = '.mua.npy'
        found_datafile = Mua.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.mua = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            try:
                session.mua = DataSession.compute_neurons_mua(session, save_on_compute=True)
            except (ValueError, AttributeError) as e:
                print(f'Computation failed with error {e}. Skipping .mua')
                session.mua = None
                
        ## PBE Epochs:
        active_file_suffix = '.pbe.npy'
        found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
        if found_datafile is not None:
            print('Loading success: {}.'.format(active_file_suffix))
            session.pbe = found_datafile
        else:
            # Otherwise load failed, perform the fallback computation
            print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
            try:
                session.pbe = DataSession.compute_pbe_epochs(session, save_on_compute=True)
            except (ValueError, AttributeError) as e:
                print(f'Computation failed with error {e}. Skipping .pbe')
                session.pbe = None
        
        # add PBE information to spikes_df from session.pbe
        cls._default_add_spike_PBEs_if_needed(session)
        cls._default_add_spike_scISIs_if_needed(session)
        # return the session with the upadated member variables
        return session
    
    
    @classmethod
    def _fallback_recinfo(cls, filepath, session):
        """ called when the .xml-method fails. Implementor can override to provide a valid .recinfo and .filePrefix anyway. """
        raise NotImplementedError # innheritor MAY override
        session.filePrefix = filepath.with_suffix("") # gets the session name (basically) without the .xml extension.
        session.recinfo = DynamicContainer(**{
            "source_file": self.source_file,
            "channel_groups": self.channel_groups,
            "skipped_channels": self.skipped_channels,
            "discarded_channels": self.discarded_channels,
            "n_channels": self.n_channels,
            "dat_sampling_rate": self.dat_sampling_rate,
            "eeg_sampling_rate": self.eeg_sampling_rate,
        })
        return session
    
    @classmethod
    def _load_xml_file(cls, filepath, session):
        # .recinfo, .filePrefix:
        session.filePrefix = filepath.with_suffix("") # gets the session name (basically) without the .xml extension.
        session.recinfo = NeuroscopeIO(filepath)
        return session

    @classmethod
    def _load_eegfile(cls, filepath, session):
        # .eegfile
        try:
            session.eegfile = BinarysignalIO(filepath, n_channels=session.recinfo.n_channels, sampling_rate=session.recinfo.eeg_sampling_rate)
        except ValueError:
            print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(filepath))
            session.eegfile = None
        return session

    @classmethod
    def _load_datfile(cls, filepath, session):
        # .datfile
        if filepath.is_file():
            session.datfile = BinarysignalIO(filepath, n_channels=session.recinfo.n_channels, sampling_rate=session.recinfo.dat_sampling_rate)
        else:
            session.datfile = None   
        return session
