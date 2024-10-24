from copy import deepcopy
from typing import Optional
import numpy as np
import pandas as pd
from neuropy.utils.efficient_interval_search import OverlappingIntervalsFallbackBehavior, determine_event_interval_identity, determine_event_interval_is_included # numba acceleration


class StartStopTimesMixin:
    def safe_start_stop_times(self, t_start, t_stop):
        """ Returns t_start and t_stop while ensuring the values passed in aren't None.
        Usage:
             t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        """
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        return t_start, t_stop

class TimeSlicableIndiciesMixin(StartStopTimesMixin):
    def time_slice_indicies(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        return (self.time > t_start) & (self.time < t_stop)
    
class TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
        raise NotImplementedError

class TimeSlicedMixin:
    """ Used in Pho's more recent Pandas DataFrame-based core classes """
    
    @property
    def time_variable_name(self):
        raise NotImplementedError

    def time_sliced(self, t_start=None, t_stop=None):
        """ 
        Implementors have a list of event times that will be used to determine inclusion/exclusion criteria.
        
        returns a copy of the spikes dataframe filtered such that only elements within the time ranges specified by t_start[i]:t_stop[i] (inclusive) are included. """
        # wrap the inputs in lists if they are scalars
        if np.isscalar(t_start):
            t_start = np.array([t_start])
        if np.isscalar(t_stop):
            t_stop = np.array([t_stop])
        
        starts = t_start
        stops = t_stop        
        # print(f'time_sliced(...): np.shape(starts): {np.shape(starts)}, np.shape(stops): {np.shape(stops)}')
        assert np.shape(starts) == np.shape(stops), f"starts and stops must be the same shape, but np.shape(starts): {np.shape(starts)} and np.shape(stops): {np.shape(stops)}"
        
        # New numba accelerated (compiled) version:
        start_stop_times_arr = np.hstack((np.atleast_2d(starts).T, np.atleast_2d(stops).T)) # atleast_2d ensures that each array is represented as a column, so start_stop_times_arr is at least of shape (1, 2)
        # print(f'time_sliced(...): np.shape(start_stop_times_arr): {np.shape(start_stop_times_arr)}')
        # print(f'np.shape(start_stop_times_arr): {np.shape(start_stop_times_arr)}')
        inclusion_mask = determine_event_interval_is_included(self._obj[self.time_variable_name].to_numpy(), start_stop_times_arr)
        # once all slices have been computed and the inclusion_mask is complete, use it to mask the output dataframe
        return self._obj.loc[inclusion_mask, :].copy()


class TimeColumnAliasesProtocol:
    """ allows time columns to be access by aliases for interoperatability """
    _time_column_name_synonyms = {"start":{'begin','start_t'},
        "stop":['end','stop_t'],
        "label":['name', 'id', 'flat_replay_idx']
    }

    @classmethod
    def find_first_extant_suitable_columns_name(cls, df: pd.DataFrame, col_connonical_name:str='start', required_columns_synonym_dict: Optional[dict]=None, should_raise_exception_on_fail:bool=False) -> Optional[str]:
        """ if the required columns (as specified in _time_column_name_synonyms's keys are missing, search for synonyms and replace the synonym columns with the preferred column name.

        Usage:
            from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

            start_col_name: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(df, col_connonical_name='start', required_columns_synonym_dict={"start":{'begin','start_t','ripple_start_t'}, "stop":['end','stop_t']}, should_raise_exception_on_fail=False)

        """
        if required_columns_synonym_dict is None:
            required_columns_synonym_dict = cls._time_column_name_synonyms
        if not isinstance(df, pd.DataFrame):
            df = df.to_dataframe()
            
        if col_connonical_name in df.columns:
            return col_connonical_name
            
        ## otherwise try synonyms for that column
        assert col_connonical_name in required_columns_synonym_dict, f"col_connonical_name: '{col_connonical_name}' is missing from required_columns_synonym_dict: {required_columns_synonym_dict}"
        synonym_columns_list = required_columns_synonym_dict[col_connonical_name]
        
        # try to rename based on synonyms
        for a_synonym in synonym_columns_list:
            if a_synonym in df.columns:
                return a_synonym # return the found column synonym
                    
        ## must be in there by the time that you're done.
        if should_raise_exception_on_fail:
            raise AttributeError(f"Failed to find synonym for the col_connonical_name: '{col_connonical_name}'.")
        else:
            return None


    @classmethod
    def renaming_synonym_columns_if_needed(cls, df: pd.DataFrame, required_columns_synonym_dict: Optional[dict]=None) -> pd.DataFrame:
        """ if the required columns (as specified in _time_column_name_synonyms's keys are missing, search for synonyms and replace the synonym columns with the preferred column name.

        Usage:
            obj = cls.renaming_synonym_columns_if_needed(obj, required_columns_synonym_dict={"start":{'begin','start_t'}, "stop":['end','stop_t']})

        """
        if required_columns_synonym_dict is None:
            required_columns_synonym_dict = cls._time_column_name_synonyms
        
        if not isinstance(df, pd.DataFrame):
            df = df.to_dataframe()
        
        for preferred_column_name, replacement_set in required_columns_synonym_dict.items():
            if preferred_column_name not in df.columns:
                # try to rename based on synonyms
                for a_synonym in replacement_set:
                    if a_synonym in df.columns:
                        df = df.rename({a_synonym: preferred_column_name}, axis="columns") # rename the synonym column to preferred_column_name
                ## must be in there by the time that you're done.
                if preferred_column_name not in df.columns:
                    raise AttributeError(f"Must have '{preferred_column_name}' column.")
        return df # important! Must return the modified obj to be assigned (since its columns were altered by renaming





@pd.api.extensions.register_dataframe_accessor("time_slicer")
class TimeSliceAccessor(TimeColumnAliasesProtocol, TimeSlicableObjectProtocol):
    """ Allows general epochs represented as Pandas DataFrames to be easily time-sliced and manipulated along with their accompanying data without making a custom class. """

    def __init__(self, pandas_obj):
        pandas_obj = self.renaming_synonym_columns_if_needed(pandas_obj, required_columns_synonym_dict={"start":{'begin','start_t'}, "stop":['end','stop_t']}) # @IgnoreException 
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @classmethod
    def _validate(cls, obj):
        """ verify there are the appropriate time columns to slice on """
        if "start" not in obj.columns or "stop" not in obj.columns:
            raise AttributeError("Must have temporal data columns named 'start' and 'stop' that represent the start and ends of the epochs.")

    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start=None, t_stop=None):
        """ Implementors return a copy of themselves with each of their members sliced at the specified indicies """
        # t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        
        # Approach copied from Laps object's time_slice(...) function
        included_df = deepcopy(self._obj)
        included_indicies = (((self._obj.start >= t_start) & (self._obj.start <= t_stop)) & ((self._obj.stop >= t_start) & (self._obj.stop <= t_stop)))
        included_df = included_df[included_indicies].reset_index(drop=True)
        return included_df
    
            

# ==================================================================================================================== #
# General Spike Identities from Epochs                                                                                 #
# ==================================================================================================================== #
def _compute_spike_arbitrary_provided_epoch_ids(spk_df, provided_epochs_df, epoch_label_column_name=None, no_interval_fill_value=np.nan, override_time_variable_name=None, overlap_behavior=OverlappingIntervalsFallbackBehavior.ASSERT_FAIL, debug_print=False):
    """ Computes the appropriate IDs from provided_epochs_df for each spikes to be added as an identities column to spikes_df
    
    overlap_behavior: OverlappingIntervalsFallbackBehavior - If ASSERT_FAIL, an AssertionError will be thrown in the case that any of the intervals in provided_epochs_df overlap each other. Otherwise, if FALLBACK_TO_SLOW_SEARCH, a much slower search will be performed that will still work.
    
    Example:
        # np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
        spike_pbe_identity_arr # Elapsed Time (seconds) = 90.92654037475586, 93.46184754371643, 90.16610431671143 
    """
    # spk_times_arr = spk_df.t_seconds.to_numpy()
    active_time_variable_name: str = (override_time_variable_name or spk_df.spikes.time_variable_name) # by default use spk_df.spikes.time_variable_name, but an optional override can be provided (to ensure compatibility with PBEs)
    spk_times_arr = spk_df[active_time_variable_name].to_numpy()
    curr_epochs_start_stop_arr = provided_epochs_df[['start','stop']].to_numpy()
    if epoch_label_column_name is None:
        curr_epoch_identity_labels = provided_epochs_df.index.to_numpy() # currently using the index instead of the label.
    else:
        assert epoch_label_column_name in provided_epochs_df.columns, f"if epoch_label_column_name is specified (not None) than the column {epoch_label_column_name} must exist in the provided_epochs_df, but provided_epochs_df.columns: {list(provided_epochs_df.columns)}!"
        curr_epoch_identity_labels = provided_epochs_df[epoch_label_column_name].to_numpy()
        
    if debug_print:
        print(f'np.shape(spk_times_arr): {np.shape(spk_times_arr)}, p.shape(curr_epochs_start_stop_arr): {np.shape(curr_epochs_start_stop_arr)}, p.shape(curr_epoch_identity_labels): {np.shape(curr_epoch_identity_labels)}')
    spike_epoch_identity_arr = determine_event_interval_identity(spk_times_arr, curr_epochs_start_stop_arr, curr_epoch_identity_labels, no_interval_fill_value=no_interval_fill_value, overlap_behavior=overlap_behavior)
    return spike_epoch_identity_arr


def add_epochs_id_identity(spk_df, epochs_df, epoch_id_key_name='temp_epoch_id', epoch_label_column_name='label', override_time_variable_name=None, no_interval_fill_value=np.nan, overlap_behavior=OverlappingIntervalsFallbackBehavior.ASSERT_FAIL):
    """ Adds the epoch IDs to each spike in spikes_df as a column named epoch_id_key_name
    
    NOTE: you can use this for non-spikes dataframes by providing `override_time_variable_name='t'`

    Example:
        # add the active_epoch's id to each spike in active_spikes_df to make filtering and grouping easier and more efficient:
        
        from neuropy.utils.mixins.time_slicing import add_epochs_id_identity
        
        active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=active_epochs.to_dataframe(), epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name=None, override_time_variable_name='t_rel_seconds', no_interval_fill_value=-1) # uses new add_epochs_id_identity

        # Get all aclus and epoch_idxs used throughout the entire spikes_df:
        all_aclus = active_spikes_df['aclu'].unique()
        all_probe_epoch_ids = active_spikes_df['Probe_Epoch_id'].unique()

        selected_spikes = active_spikes_df.groupby(['Probe_Epoch_id', 'aclu'])[active_spikes_df.spikes.time_variable_name].first() # first spikes
        

        # np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
        spike_pbe_identity_arr # Elapsed Time (seconds) = 90.92654037475586, 93.46184754371643, 90.16610431671143 , 89.04321789741516
    """
    spike_epoch_identity_arr = _compute_spike_arbitrary_provided_epoch_ids(spk_df, epochs_df, epoch_label_column_name=epoch_label_column_name, override_time_variable_name=override_time_variable_name, no_interval_fill_value=no_interval_fill_value, overlap_behavior=overlap_behavior)
    spk_df[epoch_id_key_name] = spike_epoch_identity_arr
    return spk_df


# ==================================================================================================================== #
# Spike PBE Specific Columns                                                                                           #
# ==================================================================================================================== #
def add_PBE_identity(spk_df, pbe_epoch_df, no_interval_fill_value=np.nan, overlap_behavior=OverlappingIntervalsFallbackBehavior.ASSERT_FAIL):
    """ Adds the PBE identity to the spikes_df
    Example:
        # np.shape(spk_times_arr): (16318817,), p.shape(pbe_start_stop_arr): (10960, 2), p.shape(pbe_identity_label): (10960,)
        spike_pbe_identity_arr # Elapsed Time (seconds) = 90.92654037475586, 93.46184754371643, 90.16610431671143 , 89.04321789741516
    """
    spk_df = add_epochs_id_identity(spk_df, epochs_df=pbe_epoch_df, epoch_id_key_name='PBE_id', epoch_label_column_name=None, override_time_variable_name='t_seconds', no_interval_fill_value=no_interval_fill_value, overlap_behavior=overlap_behavior) # uses new add_epochs_id_identity method which is general
    return spk_df
