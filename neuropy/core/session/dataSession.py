import sys
from typing import Sequence, Union
from warnings import warn
import numpy as np
import pandas as pd
# from klepto.safe import lru_cache as memoized
from neuropy.utils.result_context import context_extraction as memoized

from pathlib import Path
from neuropy import core
from neuropy.core import neurons
from neuropy.core.epoch import Epoch, NamedTimerange
from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
from neuropy.core.laps import Laps
from neuropy.core.position import Position
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable
from copy import deepcopy


# Local imports:
## Core:
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta, print_file_progress_message
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
from neuropy.utils.mixins.panel import DataSessionPanelMixin

from neuropy.utils.efficient_interval_search import determine_event_interval_identity # numba acceleration


# Klepto caching/memoization
# from klepto.archives import sqltable_archive as sql_archive
from klepto.archives import file_archive, sql_archive, dict_archive
from klepto.keymaps import keymap, hashmap, stringmap

_context_keymap = stringmap(flat=False, sentinel='||')
# _context_keymap = keymap(flat=False, sentinel='||')
# _context_keymap = hashmap(sentinel='||') # signature not recoverable

#d = sql_archive('postgresql://user:pass@localhost/defaultdb', cached=False)
#d = sql_archive('mysql://user:pass@localhost/defaultdb', cached=False)
# _context_cache = sql_archive(cached=False)
_context_cache = dict_archive('_test_context_cache')
# _context_cache = dict_archive(cached=False)





class DataSession(DataSessionPanelMixin, NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, ConcatenationInitializable, TimeSlicableObjectProtocol):
    """ holds the collection of all data, both loaded and computed, related to an experimental recording session. Can contain multiple discontiguous time periods ('epochs') meaning it can represent the data collected over the course of an experiment for a single animal (across days), on a single day, etc.
    
    Provides methods for loading, accessing, and manipulating data such as neural spike trains, behavioral laps, etc.
        
    """
    def __init__(self, config, filePrefix = None, recinfo = None,
                 eegfile = None, datfile = None,
                 neurons = None, probegroup = None, position = None, paradigm = None,
                 ripple = None, mua = None, laps= None, flattened_spiketrains = None, pbe = None, **kwargs):       
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
        self.flattened_spiketrains = flattened_spiketrains # core.FlattenedSpiketrains
        self.pbe = pbe
        
        for an_additional_arg, arg_value in kwargs.items():
            setattr(self, an_additional_arg, arg_value) # allows specifying additional information as optional arguments
    
    def __repr__(self) -> str:
        if self.recinfo is None:
            return f"{self.__class__.__name__}(config: {self.config}): Not yet configured."
        else:
            if self.recinfo.source_file is None:
                return f"{self.__class__.__name__}(configured from manual recinfo: {self.recinfo})"
            else:
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


    # Neurons Properties _________________________________________________________________________________________________ #
    @property
    def neuron_ids(self):
        return self.neurons.neuron_ids
    
    @property
    def n_neurons(self):
        return self.neurons.n_neurons
    
    @property
    def spikes_df(self):
        return self.flattened_spiketrains.spikes_df
    
    # Epochs properties __________________________________________________________________________________________________ #
    @property
    def epochs(self):
        """The epochs property is an alias for self.paradigm."""
        return self.paradigm
    @epochs.setter
    def epochs(self, value):
        self.paradigm = value
    @property
    def t_start(self):
        return self.paradigm.t_start    
    @property
    def duration(self):
        return self.paradigm.duration
    @property
    def t_stop(self):
        return self.paradigm.t_stop
    
    @property
    def has_replays(self):
        """The has_replays property."""
        if not hasattr(self, 'replay'):
            return False
        else:
            return (self.replay is not None)
        
    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop, enable_debug=True):
        """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
        active_epoch_times = [t_start, t_stop]
        if enable_debug: 
            print('Constraining to epoch with times (start: {}, end: {})'.format(active_epoch_times[0], active_epoch_times[1]))
        # make a copy of self:
        # should implement __deepcopy__() and __copy__()??
        
        copy_sess = DataSession.from_dict(deepcopy(self.to_dict()))
        # copy_sess = DataSession.from_dict(self.to_dict())
        
        # Slices the epochs:
        copy_sess.epochs = copy_sess.epochs.time_slice(active_epoch_times[0], active_epoch_times[1])
        
        # update the copy_session's time_sliceable objects
        copy_sess.neurons = copy_sess.neurons.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_session_Neurons: Filter by pyramidal cells only, returns a core.
        copy_sess.position = copy_sess.position.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_pos: active_epoch_pos's .time and start/end are all valid
        copy_sess.flattened_spiketrains = copy_sess.flattened_spiketrains.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_pos: active_epoch_pos's .time and start/end are all valid  
        
        if copy_sess.ripple is not None:
            copy_sess.ripple = copy_sess.ripple.time_slice(active_epoch_times[0], active_epoch_times[1]) 
        if copy_sess.mua is not None:
            # copy_sess.mua = copy_sess.mua.time_slice(active_epoch_times[0], active_epoch_times[1])
            # TODO: mua needs to be time_sliced as well, but I'm not sure how to best do this. Might be easier just to remake it.
            copy_sess.mua = DataSession.compute_neurons_mua(copy_sess, save_on_compute=False)
        if copy_sess.pbe is not None:
            copy_sess.pbe = copy_sess.pbe.time_slice(active_epoch_times[0], active_epoch_times[1]) 
            
        if copy_sess.laps is not None:
            copy_sess.laps = copy_sess.laps.time_slice(active_epoch_times[0], active_epoch_times[1]) 
        
        if copy_sess.has_replays:            
            copy_sess.replay = copy_sess.replay.time_slicer.time_slice(active_epoch_times[0], active_epoch_times[1])

        return copy_sess
    
    def get_neuron_type(self, query_neuron_type):
        """ filters self by the specified query_neuron_type, only returning neurons that match. """
        print('Constraining to units with type: {}'.format(query_neuron_type))
        # make a copy of self:
        copy_sess = DataSession.from_dict(deepcopy(self.to_dict()))
        # update the copy_session's neurons objects
        copy_sess.neurons = copy_sess.neurons.get_neuron_type(query_neuron_type) # active_epoch_session_Neurons: Filter by pyramidal cells only, returns a core.
        copy_sess.flattened_spiketrains = copy_sess.flattened_spiketrains.get_neuron_type(query_neuron_type) # active_epoch_session_Neurons: Filter by pyramidal cells only, returns a core.
        return copy_sess
    
    def get_named_epoch_timerange(self, epoch_name):
        if isinstance(epoch_name, str):            
            # convert the epoch_name string to a NamedTimerange object, get its time from self.epochs
            active_custom_named_epoch = NamedTimerange(name=epoch_name, start_end_times=self.epochs[epoch_name])
            return active_custom_named_epoch
        return None
 
    

    ## filtered_by_* functions:
    def filtered_by_time_slice(self, t_start=None, t_stop=None):
        return self.time_slice(t_start, t_stop)
        
    def filtered_by_neuron_type(self, query_neuron_type):
        return self.get_neuron_type(query_neuron_type)
    
    def filtered_by_epoch(self, epoch_specifier):
        if isinstance(epoch_specifier, str):            
            # convert the epoch_name string to a NamedTimerange object, get its time from self.epochs
            active_custom_named_epoch = NamedTimerange(name=epoch_specifier, start_end_times=self.epochs[epoch_specifier])
            return self.filtered_by_named_timerange(active_custom_named_epoch)
        elif isinstance(epoch_specifier, core.epoch.NamedTimerange):
            return self.filtered_by_named_timerange(epoch_specifier)
        else:
            print('Type(epoch_specifier): {}'.format(type(epoch_specifier)))
            raise TypeError


    def filtered_by_named_timerange(self, custom_named_timerange_obj: NamedTimerange):
        return self.time_slice(custom_named_timerange_obj.t_start, custom_named_timerange_obj.t_stop)
    

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids):
        """Implementors return a copy of themselves with neuron_ids equal to ids"""
        # copy_sess = DataSession.from_dict(deepcopy(self.to_dict()))
        copy_sess = deepcopy(self)
        copy_sess.neurons = copy_sess.neurons.get_by_id(ids)
        copy_sess.flattened_spiketrains = copy_sess.flattened_spiketrains.get_by_id(ids)
        return copy_sess

    

    # Context and Description ____________________________________________________________________________________________ #
    def get_context(self):
        """ returns an IdentifyingContext for the session """
        return self.config.get_context()
    
    def get_description(self)->str:
        """ returns a simple text descriptor of the session
        Outputs:
            a str like 'sess_kdiba_2006-6-07_11-26-53'
        """
        return self.config.get_description()
    
    def __str__(self) -> str:
        return self.get_description()
    
    @staticmethod
    def from_dict(d: dict):
        config = d.pop('config')
        return DataSession(config, **d)
        
        # return DataSession(d['config'], filePrefix = d['filePrefix'], recinfo = d['recinfo'],
        #          eegfile = d['eegfile'], datfile = d['datfile'],
        #          neurons = d['neurons'], probegroup = d.get('probegroup', None), position = d['position'], paradigm = d['paradigm'],
        #          ripple = d.get('ripple', None), mua = d.get('mua', None), pbe = d.get('pbe', None),
        #          laps= d.get('laps', None),
        #          flattened_spiketrains = d.get('flattened_spiketrains', None))

    def to_dict(self, recurrsively=False):
        simple_dict = deepcopy(self.__dict__)
        if recurrsively:
            simple_dict['paradigm'] = simple_dict['paradigm'].to_dict()
            simple_dict['position'] = simple_dict['position'].to_dict()
            simple_dict['neurons'] = simple_dict['neurons'].to_dict() 
            # simple_dict['flattened_spiketrains'] = simple_dict['flattened_spiketrains'].to_dict() ## TODO: implement .to_dict() for FlattenedSpiketrains object to make this work
        return simple_dict
        
        
    def __sizeof__(self) -> int:
        """ Returns the approximate size in bytes for this object by getting the size of its dataframes. """
        return super().__sizeof__() + int(np.sum([sys.getsizeof(self.spikes_df), sys.getsizeof(self.epochs.to_dataframe()), sys.getsizeof(self.position.to_dataframe())]))

    # DataSessionPanelMixin:
    def panel_dataframes_overview(self, max_page_items=20):
        return DataSessionPanelMixin.panel_session_dataframes_overview(self, max_page_items=max_page_items)
    
    # ==================================================================================================================== #
    # Static Computation Helper Methods                                                                                    #
    # ==================================================================================================================== #
    
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
    def compute_neurons_ripples(session, save_on_compute=False):
        print('computing ripple epochs for session...\n')
        from neuropy.analyses import oscillations
        signal = session.eegfile.get_signal()
        ripple_epochs = oscillations.detect_ripple_epochs(signal, session.probegroup)
        if save_on_compute:
            ripple_epochs.filename = session.filePrefix.with_suffix('.ripple.npy')
            # print_file_progress_message(ripple_epochs.filename, 'Saving', 'ripple epochs')
            with ProgressMessagePrinter(ripple_epochs.filename, 'Saving', 'ripple epochs'):
                ripple_epochs.save()
        # print('done.')
        return ripple_epochs
    # sess.ripple = compute_neurons_ripples(sess)

    ## BinnedSpiketrain and Mua objects using Neurons
    @staticmethod
    def compute_neurons_mua(session, save_on_compute=False):
        print('computing neurons mua for session...\n')
        mua = session.neurons.get_mua()
        if save_on_compute:
            mua.filename = session.filePrefix.with_suffix(".mua.npy")
            # print('Saving mua results to {}...'.format(mua.filename), end=' ')
            with ProgressMessagePrinter(mua.filename, 'Saving', 'mua results'):
                mua.save()
        # print('done.')
        return mua    
    # sess.mua = compute_neurons_mua(sess) # Set the .mua field on the session object once complete

    @staticmethod
    def compute_pbe_epochs(session, save_on_compute=False):
        from neuropy.analyses import detect_pbe_epochs
        print('computing PBE epochs for session...\n')
        smth_mua = session.mua.get_smoothed(sigma=0.02) # Get the smoothed mua from the session's mua
        pbe = detect_pbe_epochs(smth_mua)
        if save_on_compute:
            pbe.filename = session.filePrefix.with_suffix('.pbe.npy')
            # print('Saving pbe results to {}...'.format(pbe.filename), end=' ')
            with ProgressMessagePrinter(pbe.filename, 'Saving', 'pbe results'):
                pbe.save()
        # print('done.')
        return pbe
    # sess.pbe = compute_pbe_epochs(sess)
    
    @staticmethod
    def compute_linear_position(session, debug_print=False):
        # compute linear positions:
        print('Computing linear positions for all active epochs for session...', end=' ')
        # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
        session.position.linear_pos = np.full_like(session.position.time, np.nan)
        for anEpochLabelName in session.epochs.labels:
            try:
                curr_active_epoch_timeslice_indicies, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, epochLabelName=anEpochLabelName, method='pca')
                if debug_print:
                    print('\t curr_active_epoch_timeslice_indicies: {}\n \t np.shape(curr_active_epoch_timeslice_indicies): {}'.format(curr_active_epoch_timeslice_indicies, np.shape(curr_active_epoch_timeslice_indicies)))
                
                session.position._data.loc[curr_active_epoch_timeslice_indicies, 'lin_pos'] = linearized_positions_maze1.x
            except ValueError as e:
                # A ValueError occurs when the positions are empty during a given epoch (which occurs during any non-maze Epoch, such as 'pre' or 'post'.
                if debug_print:
                    # print(f'\t skipping non-maze epoch "{anEpochLabelName}"')
                    warn(f'\t skipping non-maze epoch "{anEpochLabelName}" due to error: {e}')                

        return session.position
        

    # ## TODO: needs neuropy! Specifically: from neuropy.analyses import Pf1D, Pf2D, perform_compute_placefields, plot_all_placefields
    # @staticmethod
    # def compute_placefields_as_needed(active_session, computation_config=None, active_epoch_placefields1D = None, active_epoch_placefields2D = None, should_force_recompute_placefields=False, should_display_2D_plots=False):
    #     if computation_config is None:
    #         computation_config = PlacefieldComputationParameters(speed_thresh=9, grid_bin=2, smooth=0.5)
    #     active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_session.neurons, active_session.position, computation_config, active_epoch_placefields1D, active_epoch_placefields2D, should_force_recompute_placefields=True)
    #     # Plot the placefields computed and save them out to files:
    #     if should_display_2D_plots:
    #         ax_pf_1D, occupancy_fig, active_pf_2D_figures = plot_all_placefields(active_epoch_placefields1D, active_epoch_placefields2D, active_config.computation_config)
    #     else:
    #         print('skipping 2D placefield plots')
    #     return active_epoch_placefields1D, active_epoch_placefields2D


    @memoized(cache=_context_cache, keymap=_context_keymap, ignore=('self', 'save_on_compute', 'debug_print'))
    def perform_compute_estimated_replay_epochs(self, min_epoch_included_duration=0.06, max_epoch_included_duration=0.6, maximum_speed_thresh=2.0, min_inclusion_fr_active_thresh=2.0, min_num_unique_aclu_inclusions=3, save_on_compute=False, debug_print=False):
        """estimates replay epochs from PBE and Position data.

        Args:
            self (_type_): _description_
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            save_on_compute (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        return DataSession.compute_estimated_replay_epochs(self, min_epoch_included_duration=min_epoch_included_duration, max_epoch_included_duration=max_epoch_included_duration, maximum_speed_thresh=maximum_speed_thresh, min_inclusion_fr_active_thresh=min_inclusion_fr_active_thresh, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, save_on_compute=save_on_compute, debug_print=debug_print)
    

    ## Estimate Replay epochs from PBE and Position data.
    @staticmethod
    def compute_estimated_replay_epochs(a_session, min_epoch_included_duration=0.06, max_epoch_included_duration=0.6, maximum_speed_thresh=2.0, min_inclusion_fr_active_thresh=2.0, min_num_unique_aclu_inclusions=3, save_on_compute=False, debug_print=False):
        """estimates replay epochs from PBE and Position data.

        Args:
            a_session (_type_): _description_
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            save_on_compute (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import KnownFilterEpochs
        from neuropy.utils.efficient_interval_search import filter_epochs_by_speed
        from neuropy.utils.efficient_interval_search import filter_epochs_by_num_active_units

        print('computing estimated replay epochs for session...\n')
        filter_epochs = a_session.pbe # Epoch object
        filter_epoch_replacement_type = KnownFilterEpochs.PBE

        # filter_epochs = a_session.ripple # Epoch object
        # filter_epoch_replacement_type = KnownFilterEpochs.RIPPLE

        # active_identifying_session_ctx = a_session.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06' # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
        # active_context = active_identifying_session_ctx.adding_context(collision_prefix='fn', fn_name='long_short_firing_rate_indicies')
        print(f'\t using {filter_epoch_replacement_type} as surrogate replays...')
        # active_context = active_context.adding_context(collision_prefix='replay_surrogate', replays=filter_epoch_replacement_type.name)

        # `KnownFilterEpochs.perform_get_filter_epochs_df(...)` returns one of the pre-known types of epochs (e.g. PBE, Ripple, etc.) as an Epoch object.
        curr_replays = KnownFilterEpochs.perform_get_filter_epochs_df(sess=a_session, filter_epochs=filter_epochs, min_epoch_included_duration=None) # returns Epoch object, don't use min_epoch_included_duration here, we'll do it in the next step.


        
        ## Filter based on required overlap with Ripples:

        # active_filter_epochs = deepcopy(a_session.ripple) # epoch ripple object
        # if not isinstance(active_filter_epochs, pd.DataFrame):
            # active_filter_epochs = active_filter_epochs.to_dataframe()
        # active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
        # active_filter_epochs['label'] = active_filter_epochs.index.to_numpy() # integer ripple indexing

        ## Filter by overlapping ripple critiera first:
        # ripple_interval_obj = a_session.ripple.to_PortionInterval()
        # ripple_interval_obj

        # replays_Interval_obj = a_session.ripple.to_PortionInterval().intersection(curr_replays.to_PortionInterval())
        # replays_Interval_obj
        curr_replays = Epoch.from_PortionInterval(a_session.ripple.to_PortionInterval().intersection(curr_replays.to_PortionInterval()))

        # Filter by duration bounds:
        curr_replays = curr_replays.filtered_by_duration(min_duration=min_epoch_included_duration, max_duration=max_epoch_included_duration)
        # Filter *_replays_Interval by requiring them to be below the speed:
        if maximum_speed_thresh is not None:
            curr_replays, above_speed_threshold_intervals, below_speed_threshold_intervals = filter_epochs_by_speed(a_session.position.to_dataframe(), curr_replays, speed_thresh=maximum_speed_thresh, debug_print=debug_print)

        # 2023-02-10 - Trimming and Filtering Estimated Replay Epochs based on cell activity and pyramidal cell start/end times:
        if (min_inclusion_fr_active_thresh is not None) or (min_num_unique_aclu_inclusions is not None):
            active_spikes_df = a_session.spikes_df.spikes.sliced_by_neuron_type('pyr') # trim based on pyramidal cell activity only
            spike_trimmed_active_epochs, epoch_split_spike_dfs, all_aclus, dense_epoch_split_frs_mat, is_cell_active_in_epoch_mat = filter_epochs_by_num_active_units(active_spikes_df, curr_replays, min_inclusion_fr_active_thresh=min_inclusion_fr_active_thresh, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions) # TODO: seems wasteful considering we compute all these spikes_df metrics and refinements and then don't return them.
            curr_replays = spike_trimmed_active_epochs # use the spike_trimmed_active_epochs as the new curr_replays

        return curr_replays





    # ConcatenationInitializable protocol:
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]):
        print('!! WARNING: Session.concat(...) is not yet fully implemented, meaning the returned session is not fully valid. Continue with caution.')
        raise NotImplementedError
    
        new_neurons = neurons.Neurons.concat([aSession.neurons for aSession in objList])
        new_position = Position.concat([aSession.position for aSession in objList])
        new_flattened_spiketrains = FlattenedSpiketrains.concat([aSession.flattened_spiketrains for aSession in objList])
        
        # TODO: eegfile, datfile, recinfo, filePrefix should all be checked to ensure they're the same, and if they aren't, should be set to None
        # TODO: probegroup, paradigm should be checked to ensure that they're the same for all, and an exception should occur if they differ
        # TODO: ripple, mua, pbe, and maybe others should be concatenated themselves once that functionality is implemented.
        
        return cls(objList[0].config, filePrefix = objList[0].filePrefix, recinfo = objList[0].recinfo,
                 eegfile = objList[0].eegfile, datfile = objList[0].datfile,
                 neurons = new_neurons, probegroup = objList[0].probegroup, position = new_position, paradigm = objList[0].paradigm,
                 ripple = None, mua = None, pbe = None, laps= objList[0].laps, flattened_spiketrains = new_flattened_spiketrains)
    
    
    # def filtered_by_laps(self, lap_indices=None):
    #     """ Returns a copy of this session with all of its members filtered by the laps.
    #     """
    #     lap_specific_subsessions, lap_specific_dataframes, lap_spike_indicies, lap_spike_t_seconds = Laps.build_lap_specific_lists(self, include_empty_lists=True)

    #     if lap_indices is None:
    #         lap_indices = np.arange(1, len(lap_specific_subsessions)) # all laps by default, but exclude the 0 element since that's the -1 value
            
    #     print('filtering by laps: {}'.format(lap_indices))
    #     lap_specific_subsessions = [lap_specific_subsessions[i] for i in lap_indices] # filter by the desired number of laps 
    #     ## Effectively build the new session using only the lap-specific spiketimes:
    #     return DataSession.concat(lap_specific_subsessions)
    #     # raise NotImplementedError
    
    def split_by_laps(self):
        """ Returns a list containing separate copies of this session with all of its members filtered by the laps, for each lap
        """
        return Laps.build_lap_specific_lists(self, include_empty_lists=True) # when surrounded by deepcopy, this causes memory problems
    
    
    
    def filtered_by_laps(self, lap_indices=None):
        """ Returns a copy of this session with all of its members filtered by the laps.
        """
        lap_specific_subsessions = Laps.build_lap_specific_lists(self, include_empty_lists=True)
        if lap_indices is None:
            lap_indices = np.arange(1, len(lap_specific_subsessions)) # all laps by default, but exclude the 0 element since that's the -1 value
        print('filtering by laps: {}'.format(lap_indices))
        lap_specific_subsessions = [lap_specific_subsessions[i] for i in lap_indices] # filter by the desired number of laps 
        ## Effectively build the new session using only the lap-specific spiketimes:
        return DataSession.concat(lap_specific_subsessions)
        

    def compute_position_laps(self):
        """ Adds the 'lap' and the 'lap_dir' columns to the position dataframe:
        Usage:
            laps_position_traces, curr_position_df = compute_position_laps(sess) """
        curr_position_df = self.position.to_dataframe() # get the position dataframe from the session
        curr_laps_df = self.laps.to_dataframe()
        curr_position_df = DataSession.compute_laps_position_df(curr_position_df, curr_laps_df)
        
        # update:
        self.position._data['lap'] = curr_position_df['lap']
        self.position._data['lap_dir'] = curr_position_df['lap_dir']
        
        # lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in sess.laps.lap_id] # dataframes split for each ID:
        return curr_position_df

    @staticmethod
    def compute_laps_position_df(position_df, laps_df):
        """ Adds a 'lap' column to the position dataframe:
            Also adds a 'lap_dir' column, containing 0 if it's an outbound trial, 1 if it's an inbound trial, and -1 if it's neither.
        Usage:
            laps_position_traces, curr_position_df = compute_position_laps(sess) """
        position_df['lap'] = np.NaN # set all 'lap' column to NaN
        position_df['lap_dir'] = np.full_like(position_df['lap'], -1) # set all 'lap_dir' to -1

        for i in np.arange(len(laps_df['lap_id'])):
            curr_lap_id = laps_df.loc[i, 'lap_id']
            curr_lap_t_start, curr_lap_t_stop = laps_df.loc[i, 'start'], laps_df.loc[i, 'stop']
            # curr_lap_t_start, curr_lap_t_stop = self.laps.get_lap_times(i)
            # print('lap[{}]: ({}, {}): '.format(curr_lap_id, curr_lap_t_start, curr_lap_t_stop))
            curr_lap_position_df_is_included = position_df['t'].between(curr_lap_t_start, curr_lap_t_stop, inclusive='both') # returns a boolean array indicating inclusion in teh current lap
            position_df.loc[curr_lap_position_df_is_included, ['lap']] = curr_lap_id # set the 'lap' identifier on the object
            # curr_position_df.query('-0.5 <= t < 0.5')
        
        # update the lap_dir variable:
        position_df.loc[np.logical_not(np.isnan(position_df.lap.to_numpy())), 'lap_dir'] = np.mod(position_df.loc[np.logical_not(np.isnan(position_df.lap.to_numpy())), 'lap'], 2.0)
        
        # return the extracted traces and the updated curr_position_df
        return position_df
    
    
    def compute_spikes_PBEs(self):
        """ Adds the 'PBE_id' column to the spikes dataframe:
        Usage:
            updated_spikes_df = sess.compute_spikes_PBEs()"""
        curr_pbe_epoch = self.pbe # EPOCH object
        curr_pbe_epoch_df = curr_pbe_epoch.to_dataframe()
        # curr_spk_df = self.spikes_df.copy()
        # curr_spk_df = self.spikes_df
        curr_spk_df = DataSession.compute_PBEs_spikes_df(self.spikes_df, curr_pbe_epoch_df) # column is added to the self.spikes_df, so the return value doesn't matter
        
        # update: Not needed because the dataframe is updated in the DataSession.compute_PBE_spikes_df function.
        # self.neurons._data['PBE_id'] = curr_spk_df['PBE_id']
        # self.spikes_df['PBE_id'] = curr_spk_df['PBE_id']
        
        return self.spikes_df
    
    @staticmethod
    def compute_PBEs_spikes_df(spk_df, pbe_epoch_df):
        """ Adds a 'PBE_id' column to the spikes_df:
        Usage:
            spk_df = compute_PBEs_spikes_df(sess) """
        
        # no_interval_fill_value = np.nan
        no_interval_fill_value = -1
        spk_df['PBE_id'] = no_interval_fill_value # set all 'spk_df' column to NaN

        # make sure the labels are just the PBE index:
        pbe_epoch_df['label'] = pbe_epoch_df.index
        pbe_epoch_df['label'] = pbe_epoch_df['label'].astype(str)
        curr_time_variable_name = spk_df.spikes.time_variable_name

        try:
            spk_times_arr = spk_df[curr_time_variable_name].copy().to_numpy() # get the timestamps column using the time_variable_name property. It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data:
        except KeyError as e:
            # curr_time_variable_name (spk_df.spikes.time_variable_name) is invalid for some reason. 
            # raise "curr_time_variable_name is invalid for some reason!"
            
            proposed_updated_time_variable_name = 't_seconds'
            print(f'encounter KeyError {e} when attempting to access spk_df using its spk_df.spikes.time_variable_name variable. Original spk_df.spikes.time_variable_name: "{spk_df.spikes.time_variable_name}". Changing it to "{proposed_updated_time_variable_name}" and proceeding forward')
            # if 't_rel_seconds' is invalid, try the other one:
            spk_df.spikes.set_time_variable_name(proposed_updated_time_variable_name)
            # after the change, try again to get the spike times array:
            spk_times_arr = spk_df[proposed_updated_time_variable_name].copy().to_numpy() # get the timestamps column using the time_variable_name property. It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data:    
            

        pbe_start_stop_arr = pbe_epoch_df[['start','stop']].to_numpy()
        # pbe_identity_label = pbe_epoch_df['label'].to_numpy()
        pbe_identity_label = pbe_epoch_df.index.to_numpy() # currently using the index instead of the label.
        spike_pbe_identity_arr = determine_event_interval_identity(spk_times_arr, pbe_start_stop_arr, pbe_identity_label, no_interval_fill_value=no_interval_fill_value)
        # Set the PBE_id of the spikes dataframe:
        spk_df.loc[:, 'PBE_id'] = spike_pbe_identity_arr
        # spk_df['PBE_id'] = spike_pbe_identity_arr
        # return the extracted traces and the updated curr_position_df
        return spk_df

        
# # Helper function that processed the data in a given directory
# def processDataSession(basedir='/Volumes/iNeo/Data/Bapun/Day5TwoNovel'):
#     # sess = DataSession(basedir)
#     curr_args_dict = dict()
#     curr_args_dict['basepath'] = basedir
#     curr_args_dict['session_obj'] = DataSession() # Create an empty session object
#     sess = DataSessionLoader._default_load_bapun_npy_session_folder(curr_args_dict)
#     return sess


# ## Main:
# if __name__ == "__main__":
#     # Now initiate the class
#     # basedir = '/data/Working/Opto/Jackie671/Jackie_placestim_day2/Jackie_TRACK_2020-10-07_11-21-39'  # fill in here
#     basedir = 'R:\data\Bapun\Day5TwoNovel'
#     # basedir = '/Volumes/iNeo/Data/Bapun/Day5TwoNovel'
#     sess = processDataSession(basedir)
#     print(sess.recinfo)
#     sess.epochs.to_dataframe()
#     sess.neurons.get_all_spikes()
#     sess.position.sampling_rate # 60
    
#     pass


