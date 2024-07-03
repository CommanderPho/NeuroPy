"""
This type stub file was generated by pyright.
"""

import pandas as pd
import portion as P
from enum import Enum

class OverlappingIntervalsFallbackBehavior(Enum):
    """Describes the behavior of the search when the provided epochs overlap each other.
        overlap_behavior: OverlappingIntervalsFallbackBehavior - If ASSERT_FAIL, an AssertionError will be thrown in the case that any of the intervals in provided_epochs_df overlap each other. Otherwise, if FALLBACK_TO_SLOW_SEARCH, a much slower search will be performed that will still work.
    """
    ASSERT_FAIL = ...
    FALLBACK_TO_SLOW_SEARCH = ...


def verify_non_overlapping(start_stop_times_arr):
    """Returns True if no members of the start_stop_times_arr overlap each other.

    Args:
        start_stop_times_arr (_type_): An N x 2 numpy array of start, stop times

    Returns:
        bool: Returns true if all members are non-overlapping
        
    Example:
        are_all_non_overlapping = verify_non_overlapping(pbe_epoch_df[['start','stop']].to_numpy())
        are_all_non_overlapping

    """
    ...

def get_non_overlapping_epochs(start_stop_times_arr):
    """Gets the indicies of any epochs that DON'T overlap one another.
    
    Args:
        start_stop_times_arr (_type_): An N x 2 numpy array of start, stop times
        
    Example:        
        from neuropy.utils.efficient_interval_search import drop_overlapping
        start_stop_times_arr = any_lap_specific_epochs.to_dataframe()[['start','stop']].to_numpy() # note that this returns one less than the number of epochs.
        non_overlapping_start_stop_times_arr = get_non_overlapping_epochs(start_stop_times_arr)
        non_overlapping_start_stop_times_arr
    """
    ...

def drop_overlapping(start_stop_times_arr):
    """Drops the overlapping epochs

    Args:
        start_stop_times_arr (_type_): An N x 2 numpy array of start, stop times
        
    Example:        
        from neuropy.utils.efficient_interval_search import drop_overlapping
        start_stop_times_arr = any_lap_specific_epochs.to_dataframe()[['start','stop']].to_numpy() # note that this returns one less than the number of epochs.
        non_overlapping_start_stop_times_arr = drop_overlapping(start_stop_times_arr)
        non_overlapping_start_stop_times_arr
    """
    ...

def get_overlapping_indicies(start_stop_times_arr):
    """Gets the indicies of any epochs that DO overlap one another.
    
    Args:
        start_stop_times_arr (_type_): An N x 2 numpy array of start, stop times
        
    Example:        
        from neuropy.utils.efficient_interval_search import get_overlapping_indicies
        curr_laps_obj = deepcopy(sess.laps)
        start_stop_times_arr = np.vstack([curr_laps_obj.starts, curr_laps_obj.stops]).T # (80, 2)
        all_overlapping_lap_indicies = get_overlapping_indicies(start_stop_times_arr)
        all_overlapping_lap_indicies
    """
    ...

def debug_overlapping_epochs(epochs_df):
    """
    from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, drop_overlapping, get_overlapping_indicies, OverlappingIntervalsFallbackBehavior
    curr_epochs_obj = deepcopy(sess.ripple)
    debug_overlapping_epochs(curr_epochs_obj.to_dataframe())
    
    """
    ...

def determine_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels=..., no_interval_fill_value=..., overlap_behavior=...): # -> tuple[Any, list[Any]]:
    """ Given a list of event times (`times_arr`) and a separate list of epoch start_stop_times (`start_stop_times_arr`), adds a
    Usage:
        from neuropy.utils.efficient_interval_search import determine_event_interval_identity

    """
    ...

def determine_unsorted_event_interval_identity(times_arr, start_stop_times_arr, period_identity_labels, no_interval_fill_value=..., overlap_behavior=...): # -> tuple[Any, list[Any]]:
    ...

def determine_event_interval_is_included(times_arr, start_stop_times_arr):
    ...

def deduplicate_epochs(epochs_df, agressive_deduplicate: bool = ...):
    """ Attempts to remove literal duplicate ('start', 'stop') entries in the epochs_df. Does not do anything about overlap if the epochs don't match
    returns the non-duplicated epochs in epochs_df.

    Usage:
        from neuropy.utils.efficient_interval_search import deduplicate_epochs
        curr_epochs_df = deduplicate_epochs(epochs_df, agressive_deduplicate=True)

    """
    ...

def convert_PortionInterval_to_epochs_df(intervals: P.Interval) -> pd.DataFrame:
    """ 
    Usage:
        epochs_df = convert_PortionInterval_to_epochs_df(long_replays_intervals)
    """
    ...

def convert_PortionInterval_to_Epoch_obj(interval: P.Interval): # -> Epoch:
    """ build an Epoch object version
    Usage:
        combined_epoch_obj = convert_PortionInterval_to_Epoch_obj(long_replays_intervals)
    """
    ...

def convert_Epoch_obj_to_PortionInterval_obj(epoch_obj, **P_Interval_kwargs) -> P.Interval:
    """ build an Interval object version
    Usage:
        combined_interval_obj = convert_Epoch_obj_to_PortionInterval_obj(long_replays_intervals)
    """
    ...

def filter_epochs_by_speed(speed_df, *epoch_args, speed_thresh=..., debug_print=...): # -> tuple[*tuple[Epoch, ...], Any, Any]:
    """ Filter *_replays_Interval by requiring them to be below the speed 
    *epoch_args = long_replays, short_replays, global_replays

    Usage:
        from neuropy.utils.efficient_interval_search import filter_epochs_by_speed
        speed_thresh = 2.0
        speed_df = global_session.position.to_dataframe()
        long_replays, short_replays, global_replays, above_speed_threshold_intervals, below_speed_threshold_intervals = filter_epochs_by_speed(speed_df, long_replays, short_replays, global_replays, speed_thresh=speed_thresh, debug_print=True)
    """
    ...

def trim_epochs_to_first_last_spikes(active_spikes_df, active_epochs, min_num_unique_aclu_inclusions=...): # -> tuple[Epoch, list[Any]]:
    """ 2022-02-16 - Trim the active_epochs to the first and last spike times for each epoch.

    Usage:
        from neuropy.utils.efficient_interval_search import trim_epochs_to_first_last_spikes
        spike_trimmed_active_epochs, epoch_split_spike_dfs = trim_epochs_to_first_last_spikes(active_spikes_df, active_epochs, min_num_unique_aclu_inclusions=1)
    """
    ...

def filter_epochs_by_num_active_units(active_spikes_df, active_epochs, min_inclusion_fr_active_thresh: float = ..., min_num_unique_aclu_inclusions=..., include_intermediate_computations: bool = ...): # -> tuple[Epoch, tuple[list[Any], Any, Any, Any, Any] | None]:
    """ Filter active_epochs by requiring them to have at least `min_num_unique_aclu_inclusions` active units as determined by filtering active_spikes_df.
    Inputs:
        active_spikes_df: a spike_df with only active units
        min_inclusion_fr_active_thresh: the minimum firing rate (Hz) for a unit during a given epoch for it to be considered active for that epoch
        min_num_unique_aclu_inclusions: the minimum number of unique active units that must be present in the epoch for it to be considered active


    # Calls `trim_epochs_to_first_last_spikes`
    
    Outputs:
        epoch_split_spike_dfs: !!PITFALL: note the number of these is per original epochs, not the post-filtered number. To get the post-filtered number for any of these values, do the following:
            epoch_split_spike_dfs = [df for i, df in enumerate(epoch_split_spike_dfs) if is_epoch_sufficiently_active[i]] # filter the list `epoch_split_spike_dfs` as well, takes some time    
        is_cell_active_in_epoch_mat: !!PITFALL: note the number of these is per original epochs, not the post-filtered number
        is_epoch_sufficiently_active: !!PITFALL: note the number of these is per original epochs, not the post-filtered number
        
    """
    ...

