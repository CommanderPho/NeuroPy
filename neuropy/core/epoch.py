from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Callable, Any
from importlib import metadata
import warnings
from warnings import warn
import numpy as np
import nptyping as ND
from nptyping import NDArray
import pandas as pd
import portion as P # Required for interval search: portion~=2.3.0
import heapq # for `assign_overlap_y_offset`

from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable, DataFrameInitializable
from .datawriter import DataWriter
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicedMixin, TimeColumnAliasesProtocol
from neuropy.utils.mixins.metadata_helpers import DataframeMetadataProtocol, MetadataAccessor
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, deduplicate_epochs # for EpochsAccessor's .get_non_overlapping_df()
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin





# Compatibility wrappers - these delegate to EpochHelpers for backward compatibility
# These can be deprecated in the future. New code should use EpochHelpers directly.
def find_data_indicies_from_epoch_times(a_df: pd.DataFrame, epoch_times: NDArray, t_column_names=None, atol:float=1e-3, not_found_action='skip_index', debug_print=False) -> NDArray:
    """DEPRECATED: Use EpochHelpers.find_data_indicies_from_epoch_times instead.
    
    This is a compatibility wrapper that delegates to EpochHelpers.find_data_indicies_from_epoch_times.
    """
    return EpochHelpers.find_data_indicies_from_epoch_times(a_df, epoch_times, t_column_names=t_column_names, atol=atol, not_found_action=not_found_action, debug_print=debug_print)

def find_epoch_times_to_data_indicies_map(a_df: pd.DataFrame, epoch_times: NDArray, t_column_names=None, atol:float=1e-3, not_found_action='skip_index', debug_print=False) -> Dict[Union[float, Tuple[float, float]], Union[int, NDArray]]:
    """DEPRECATED: Use EpochHelpers.find_epoch_times_to_data_indicies_map instead.
    
    This is a compatibility wrapper that delegates to EpochHelpers.find_epoch_times_to_data_indicies_map.
    """
    return EpochHelpers.find_epoch_times_to_data_indicies_map(a_df, epoch_times, t_column_names=t_column_names, atol=atol, not_found_action=not_found_action, debug_print=debug_print)

def find_epochs_overlapping_other_epochs(epochs_df: pd.DataFrame, epochs_df_required_to_overlap: pd.DataFrame):
    """DEPRECATED: Use EpochHelpers.find_epochs_overlapping_other_epochs instead.
    
    This is a compatibility wrapper that delegates to EpochHelpers.find_epochs_overlapping_other_epochs.
    """
    return EpochHelpers.find_epochs_overlapping_other_epochs(epochs_df, epochs_df_required_to_overlap)

def sample_random_period_from_epoch(epoch_start: float, epoch_stop: float, training_data_portion: float, *additional_lap_columns, debug_print=False, debug_override_training_start_t=None):
    """DEPRECATED: Use EpochHelpers.sample_random_period_from_epoch instead.
    
    This is a compatibility wrapper that delegates to EpochHelpers.sample_random_period_from_epoch.
    """
    return EpochHelpers.sample_random_period_from_epoch(epoch_start, epoch_stop, training_data_portion, *additional_lap_columns, debug_print=debug_print, debug_override_training_start_t=debug_override_training_start_t)

def split_epochs_into_training_and_test(epochs_df: pd.DataFrame, training_data_portion: float=5.0/6.0, group_column_name: str='lap_id', additional_epoch_identity_column_names=['label', 'lap_id', 'lap_dir'], debug_print: bool = False):
    """DEPRECATED: Use EpochHelpers.split_epochs_into_training_and_test instead.
    
    This is a compatibility wrapper that delegates to EpochHelpers.split_epochs_into_training_and_test.
    """
    return EpochHelpers.split_epochs_into_training_and_test(epochs_df, training_data_portion=training_data_portion, group_column_name=group_column_name, additional_epoch_identity_column_names=additional_epoch_identity_column_names, debug_print=debug_print)

def subdivide_epochs(df: pd.DataFrame, subdivide_bin_size: float, start_col='start', stop_col='stop') -> pd.DataFrame:
    """DEPRECATED: Use EpochHelpers.subdivide_epochs instead.
    
    This is a compatibility wrapper that delegates to EpochHelpers.subdivide_epochs.
    """
    return EpochHelpers.subdivide_epochs(df, subdivide_bin_size, start_col=start_col, stop_col=stop_col)


class EpochHelpers:
    """ top-level static helpers for epochs

    from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch, EpochHelpers


    """
    @classmethod
    def assign_overlap_y_offset(cls, df: pd.DataFrame, start_col: str = 'start', stop_col: str = 'stop', out_col: str = 'overlap_y_offset') -> pd.DataFrame:
        """Assign an integer y-offset so overlapping epochs don't share the same band.
        Used by: 'pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.EpochsEditorItem'

        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch, EpochHelpers

            curr_paradigm_df = EpochHelpers.assign_overlap_y_offset(df=curr_paradigm_df)
            curr_paradigm_df

        """
        if df.empty:
            df[out_col] = []
            return df

        sorted_df = df.sort_values(by=[start_col, stop_col])
        starts = sorted_df[start_col].to_numpy()
        stops = sorted_df[stop_col].to_numpy()
        idxs = sorted_df.index.to_numpy()

        active = []
        free_levels = []
        next_level = 0
        level_by_index = {}

        for start, stop, idx in zip(starts, stops, idxs):
            while active and active[0][0] < start:
                finished_stop, finished_level = heapq.heappop(active)
                heapq.heappush(free_levels, finished_level)

            if free_levels:
                level = heapq.heappop(free_levels)
            else:
                level = next_level
                next_level += 1

            level_by_index[idx] = level
            heapq.heappush(active, (stop, level))

        df[out_col] = pd.Series(level_by_index)
        return df


    @classmethod
    def find_data_indicies_from_epoch_times(cls, a_df: pd.DataFrame, epoch_times: NDArray, t_column_names=None, atol:float=1e-3, not_found_action='skip_index', debug_print=False) -> NDArray:
        """ returns the matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times


        skip_index: drops indicies that can't be found, meaning that the number of returned indicies might be less than len(epoch_times)


        Returns: (S, ) array of data indicies corresponding to the times.

        Uses:
            from neuropy.core.epoch import EpochHelpers

            selection_start_stop_times = deepcopy(active_epochs_df[['start', 'stop']].to_numpy())
            print(f'np.shape(selection_start_stop_times): {np.shape(selection_start_stop_times)}')

            test_epochs_data_df: pd.DataFrame = deepcopy(ripple_simple_pf_pearson_merged_df)
            print(f'np.shape(test_epochs_data_df): {np.shape(test_epochs_data_df)}')

            # 2D_search (for both start, end times):
            found_data_indicies = EpochHelpers.find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=selection_start_stop_times)
            print(f'np.shape(found_data_indicies): {np.shape(found_data_indicies)}')

            # 1D_search (only for start times):
            found_data_indicies_1D_search = EpochHelpers.find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=np.squeeze(selection_start_stop_times[:, 0]))
            print(f'np.shape(found_data_indicies_1D_search): {np.shape(found_data_indicies_1D_search)}')
            found_data_indicies_1D_search

            assert np.array_equal(found_data_indicies, found_data_indicies_1D_search)
        

        - [X] FIXED 2024-03-04 19:55 - This function was peviously incorrect and could return multiple matches for each passed time due to the tolerance.
            
        """
        def _subfn_find_epoch_times(epoch_slices_df: pd.DataFrame, epoch_times: NDArray, active_t_column_names=['start','stop'], ndim:int=2) -> NDArray:
            """Loop through each pair of epoch_times and find the closest start and end time
            
            Captures: atol, debug_print, not_found_action

            """
            assert len(active_t_column_names) == ndim, f"ndim: {ndim}, active_t_column_names: {active_t_column_names}"
            assert not_found_action in ['skip_index', 'full_nan']

            if (ndim == 0):
                epoch_times = np.atleast_1d(epoch_times)
                ndim = 1

            indices = []
            if (ndim == 1):
                for start_time in epoch_times:
                    # Find the index with the closest start time
                    start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin() # idxmin returns a .loc index apparently?

                    ## Numpy-only version:
                    # start_index: NDArray = np.argmin(epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().to_numpy())

                    # start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin() 
                    selected_index = start_index
                    
                    ## End if
                    ## Check the tolerance
                    assert selected_index is not None
                    was_index_found: bool = True # true by default

                    if atol is not None:
                        
                        # Can convert to an actual integer index like this:
                        # selected_integer_position_index = epoch_slices_df.index.get_loc(selected_index) # to match with .iloc do this
                        # selected_index_diff = epoch_slices_df.iloc[selected_integer_position_index].sub(start_time)

                        ## See how the selecteded index's values diff from the search values
                        selected_index_diff = epoch_slices_df.loc[selected_index].sub(start_time) # IndexError: single positional indexer is out-of-bounds

                        ## Check against tolerance:
                        exceeds_tolerance: bool = np.any((selected_index_diff.abs() > atol))
                        if exceeds_tolerance:
                            if debug_print:
                                print(f'WARN: CLOSEST FOUND INDEX EXCEEDS TOLERANCE (atol={atol}):\n\tsearch_time: {start_time}, closest: {epoch_slices_df.loc[selected_index].to_numpy()}, {selected_index_diff}. No matching index was found.')
                            selected_index = np.nan
                            was_index_found = False

                    if was_index_found:
                        # index was found
                        indices.append(selected_index)
                    else:
                        ## index not found:
                        if not_found_action == 'skip_index':
                            # skip without adding this index. This means the the output array will be smaller than the epoch_times
                            pass
                        elif (not_found_action == 'full_nan'):
                            ## append the nan anyway
                            indices.append(selected_index)
                        else:
                            raise NotImplementedError(f"not_found_action: {not_found_action}")


                # end for
                        
            elif (ndim == 2):
                for start_time, end_time in epoch_times:
                    # Find the index with the closest start time
                    start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin()
                    # Find the index with the closest end time
                    end_index = epoch_slices_df[active_t_column_names[1]].sub(end_time).abs().idxmin()
                    
                    was_index_found: bool = True # true by default
                    
                    selected_index = None
                    # If the start and end indices are the same, we have a match
                    if (start_index == end_index):
                        ## Good, this is how it should be, they correspond to the same (single) row:
                        selected_index = start_index
                    else:
                        ## MODE: CLOSEST START
                        if debug_print:
                            print(f'WARNING: CLOSEST START INDEX: {start_index} is not equal to the closest END index: {end_index}. Using start index.')
                        selected_index = start_index

                        # ## MODE: CLOSEST START OR STOP
                        # start_diff = epoch_slices_df.iloc[start_index].sub([start_time, end_time]).abs().sum()
                        # end_diff = epoch_slices_df.iloc[end_index].sub([start_time, end_time]).abs().sum()
                        # # If not, find which one is closer overall (by comparing the sum of absolute differences to start_time and end_time)
                        # selected_index = (start_index if start_diff <= end_diff else end_index)

                    ## End if
                    ## Check the tolerance
                    assert selected_index is not None
                    if atol is not None:
                        ## See how the selecteded index's values diff from the search values
                        selected_index_diff = epoch_slices_df.loc[selected_index].sub([start_time, end_time]) # .loc[selected_index] method supposedly compatibile with .idxmin()
                        # selected_index_diff = epoch_slices_df.iloc[selected_index].sub([start_time, end_time]) #.abs() #.sum() # IndexError: single positional indexer is out-of-bounds -- selected_index: 319. SHIT. Confirmed it corresponds to df.Index == 319, which is at .iloc[134]
                        exceeds_tolerance: bool = np.any((selected_index_diff.abs() > atol))
                        if exceeds_tolerance:
                            if debug_print:
                                print(f'WARN: CLOSEST FOUND INDEX EXCEEDS TOLERANCE (atol={atol}):\n\tsearch_time: [{start_time}, {end_time}], closest: [{epoch_slices_df.loc[selected_index].to_numpy()}], diff: [{selected_index_diff.to_numpy()}]. No matching index was found.')
                            selected_index = np.nan
                            was_index_found = False

                    if was_index_found:
                        # index was found
                        indices.append(selected_index)
                    else:
                        ## index not found:
                        if not_found_action == 'skip_index':
                            # skip without adding this index. This means the the output array will be smaller than the epoch_times
                            pass
                        elif (not_found_action == 'full_nan'):
                            ## append the nan anyway
                            indices.append(selected_index)
                        else:
                            raise NotImplementedError(f"not_found_action: {not_found_action}")
                    # end for
            else:
                raise NotImplementedError(f"ndim: {ndim}")
            
            # Return the indices as an ndarray
            return np.array(indices)


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        assert not_found_action in ['skip_index', 'full_nan']

        if ((np.ndim(epoch_times) == 2) and (np.shape(epoch_times)[1] == 2)):
            if t_column_names is None:
                t_column_names = ['start', 'stop']
            assert (len(t_column_names) == 2), f"len(t_column_names): {len(t_column_names)} != 2)"
            num_query_times: int = np.shape(epoch_times)[0]
        elif (np.ndim(epoch_times) == 1):
            # start times only
            if t_column_names is None:
                t_column_names = ['start',]
            if len(t_column_names) > 1:
                t_column_names = [t_column_names[0],]
            num_query_times: int = len(epoch_times)


        elif (np.ndim(epoch_times) == 0):
            # single start time only
            if t_column_names is None:
                t_column_names = ['start',]
            if len(t_column_names) > 1:
                t_column_names = [t_column_names[0],]
            epoch_times = np.atleast_1d(epoch_times)
            num_query_times: int = 1

        else:
            raise NotImplementedError(f"np.ndim(epoch_times): {np.ndim(epoch_times)}, np.shape(epoch_times)[1]: {np.shape(epoch_times)[1]}")

        # start, stop epoch times:
        epoch_slices_df = a_df[t_column_names]

        found_data_indicies = _subfn_find_epoch_times(epoch_slices_df=epoch_slices_df, epoch_times=epoch_times, active_t_column_names=t_column_names, ndim=len(t_column_names))
        if not_found_action == 'skip_index':
            # skip without adding this index. This means the the output found_data_indicies might be smaller than the num_query_times
            assert (len(found_data_indicies) <= num_query_times), f"num_query_times: {num_query_times}, len(found_data_indicies): {len(found_data_indicies)}"
        elif (not_found_action == 'full_nan'):
            assert (len(found_data_indicies) == num_query_times), f"num_query_times: {num_query_times}, len(found_data_indicies): {len(found_data_indicies)}"
        else:
            raise NotImplementedError(f"not_found_action: {not_found_action}")

        return found_data_indicies

    @classmethod
    def find_epoch_times_to_data_indicies_map(cls, a_df: pd.DataFrame, epoch_times: NDArray, t_column_names=None, atol:float=1e-3, not_found_action='skip_index', debug_print=False) -> Dict[Union[float, Tuple[float, float]], Union[int, NDArray]]:
        """ returns the matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times


        skip_index: drops indicies that can't be found, meaning that the number of returned indicies might be less than len(epoch_times)


        Returns: (S, ) array of data indicies corresponding to the times.

        Uses:
            from neuropy.core.epoch import EpochHelpers
            
            selection_start_stop_times = deepcopy(active_epochs_df[['start', 'stop']].to_numpy())
            print(f'np.shape(selection_start_stop_times): {np.shape(selection_start_stop_times)}')

            test_epochs_data_df: pd.DataFrame = deepcopy(ripple_simple_pf_pearson_merged_df)
            print(f'np.shape(test_epochs_data_df): {np.shape(test_epochs_data_df)}')

            # 2D_search (for both start, end times):
            found_data_indicies = EpochHelpers.find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=selection_start_stop_times)
            print(f'np.shape(found_data_indicies): {np.shape(found_data_indicies)}')

            # 1D_search (only for start times):
            found_data_indicies_1D_search = EpochHelpers.find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=np.squeeze(selection_start_stop_times[:, 0]))
            print(f'np.shape(found_data_indicies_1D_search): {np.shape(found_data_indicies_1D_search)}')
            found_data_indicies_1D_search

            assert np.array_equal(found_data_indicies, found_data_indicies_1D_search)
        

        - [X] FIXED 2024-03-04 19:55 - This function was peviously incorrect and could return multiple matches for each passed time due to the tolerance.
            
        """
        def _subfn_find_epoch_times_map(epoch_slices_df: pd.DataFrame, epoch_times: NDArray, active_t_column_names=['start','stop'], ndim:int=2):
            """Loop through each pair of epoch_times and find the closest start and end time
            
            Captures: atol, debug_print, not_found_action

            """
            assert len(active_t_column_names) == ndim, f"ndim: {ndim}, active_t_column_names: {active_t_column_names}"
            assert not_found_action in ['skip_index', 'full_nan']

            if (ndim == 0):
                epoch_times = np.atleast_1d(epoch_times)
                ndim = 1

            indices = []
            epoch_time_to_index_map = {}
            if (ndim == 1):
                for start_time in epoch_times:
                    # Find the index with the closest start time
                    start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin() # idxmin returns a .loc index apparently?

                    ## Numpy-only version:
                    # start_index: NDArray = np.argmin(epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().to_numpy())

                    # start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin() 
                    selected_index = start_index
                    
                    ## End if
                    ## Check the tolerance
                    assert selected_index is not None
                    was_index_found: bool = True # true by default

                    if atol is not None:
                        
                        # Can convert to an actual integer index like this:
                        # selected_integer_position_index = epoch_slices_df.index.get_loc(selected_index) # to match with .iloc do this
                        # selected_index_diff = epoch_slices_df.iloc[selected_integer_position_index].sub(start_time)

                        ## See how the selecteded index's values diff from the search values
                        selected_index_diff = epoch_slices_df.loc[selected_index].sub(start_time) # IndexError: single positional indexer is out-of-bounds

                        ## Check against tolerance:
                        exceeds_tolerance: bool = np.any((selected_index_diff.abs() > atol))
                        if exceeds_tolerance:
                            if debug_print:
                                print(f'WARN: CLOSEST FOUND INDEX EXCEEDS TOLERANCE (atol={atol}):\n\tsearch_time: {start_time}, closest: {epoch_slices_df.loc[selected_index].to_numpy()}, {selected_index_diff}. No matching index was found.')
                            selected_index = np.nan
                            was_index_found = False

                    if was_index_found:
                        # index was found
                        indices.append(selected_index)
                        if isinstance(selected_index, (list, tuple, NDArray)):
                            ## if it's a list, tuple, NDArray, etc
                            if len(selected_index) > 0:
                                if (not isinstance(selected_index, NDArray)):
                                    selected_index = NDArray(selected_index) ## convert to NDArray
                                
                        epoch_time_to_index_map[start_time] = selected_index
                        
                    else:
                        ## index not found:
                        if not_found_action == 'skip_index':
                            # skip without adding this index. This means the the output array will be smaller than the epoch_times
                            pass
                        elif (not_found_action == 'full_nan'):
                            ## append the nan anyway
                            indices.append(selected_index)
                            if isinstance(selected_index, (list, tuple, NDArray)):
                                ## if it's a list, tuple, NDArray, etc
                                if len(selected_index) > 0:
                                    if (not isinstance(selected_index, NDArray)):
                                        selected_index = NDArray(selected_index) ## convert to NDArray
                                
                            epoch_time_to_index_map[start_time] = selected_index
                        else:
                            raise NotImplementedError(f"not_found_action: {not_found_action}")


                # end for
                        
            elif (ndim == 2):
                for start_time, end_time in epoch_times:
                    # Find the index with the closest start time
                    start_index = epoch_slices_df[active_t_column_names[0]].sub(start_time).abs().idxmin()
                    # Find the index with the closest end time
                    end_index = epoch_slices_df[active_t_column_names[1]].sub(end_time).abs().idxmin()
                    
                    was_index_found: bool = True # true by default
                    
                    selected_index = None
                    # If the start and end indices are the same, we have a match
                    if (start_index == end_index):
                        ## Good, this is how it should be, they correspond to the same (single) row:
                        selected_index = start_index
                    else:
                        ## MODE: CLOSEST START
                        if debug_print:
                            print(f'WARNING: CLOSEST START INDEX: {start_index} is not equal to the closest END index: {end_index}. Using start index.')
                        selected_index = start_index

                        # ## MODE: CLOSEST START OR STOP
                        # start_diff = epoch_slices_df.iloc[start_index].sub([start_time, end_time]).abs().sum()
                        # end_diff = epoch_slices_df.iloc[end_index].sub([start_time, end_time]).abs().sum()
                        # # If not, find which one is closer overall (by comparing the sum of absolute differences to start_time and end_time)
                        # selected_index = (start_index if start_diff <= end_diff else end_index)

                    ## End if
                    ## Check the tolerance
                    assert selected_index is not None
                    if atol is not None:
                        ## See how the selecteded index's values diff from the search values
                        selected_index_diff = epoch_slices_df.loc[selected_index].sub([start_time, end_time]) # .loc[selected_index] method supposedly compatibile with .idxmin()
                        # selected_index_diff = epoch_slices_df.iloc[selected_index].sub([start_time, end_time]) #.abs() #.sum() # IndexError: single positional indexer is out-of-bounds -- selected_index: 319. SHIT. Confirmed it corresponds to df.Index == 319, which is at .iloc[134]
                        exceeds_tolerance: bool = np.any((selected_index_diff.abs() > atol))
                        if exceeds_tolerance:
                            if debug_print:
                                print(f'WARN: CLOSEST FOUND INDEX EXCEEDS TOLERANCE (atol={atol}):\n\tsearch_time: [{start_time}, {end_time}], closest: [{epoch_slices_df.loc[selected_index].to_numpy()}], diff: [{selected_index_diff.to_numpy()}]. No matching index was found.')
                            selected_index = np.nan
                            was_index_found = False
                            
                    # if (not isinstance(selected_index, (float, int))):


                    if was_index_found:
                        # index was found
                        indices.append(selected_index)
                        if isinstance(selected_index, (list, tuple, NDArray)):
                            ## if it's a list, tuple, NDArray, etc
                            if len(selected_index) > 0:
                                if (not isinstance(selected_index, NDArray)):
                                    selected_index = NDArray(selected_index) ## convert to NDArray
                        epoch_time_to_index_map[(start_time, end_time,)] = selected_index
                    else:
                        ## index not found:
                        if not_found_action == 'skip_index':
                            # skip without adding this index. This means the the output array will be smaller than the epoch_times
                            pass
                        elif (not_found_action == 'full_nan'):
                            ## append the nan anyway
                            indices.append(selected_index)
                            if isinstance(selected_index, (list, tuple, NDArray)):
                                ## if it's a list, tuple, NDArray, etc
                                if len(selected_index) > 0:
                                    if (not isinstance(selected_index, NDArray)):
                                        selected_index = NDArray(selected_index) ## convert to NDArray
        
                            epoch_time_to_index_map[(start_time, end_time,)] = selected_index
                        else:
                            raise NotImplementedError(f"not_found_action: {not_found_action}")
                    # end for
            else:
                raise NotImplementedError(f"ndim: {ndim}")
            
            # Return the indices as an ndarray
            return epoch_time_to_index_map, np.array(indices)


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        assert not_found_action in ['skip_index', 'full_nan']

        if ((np.ndim(epoch_times) == 2) and (np.shape(epoch_times)[1] == 2)):
            if t_column_names is None:
                t_column_names = ['start', 'stop']
            assert (len(t_column_names) == 2), f"len(t_column_names): {len(t_column_names)} != 2)"
            num_query_times: int = np.shape(epoch_times)[0]
        elif (np.ndim(epoch_times) == 1):
            # start times only
            if t_column_names is None:
                t_column_names = ['start',]
            if len(t_column_names) > 1:
                t_column_names = [t_column_names[0],]
            num_query_times: int = len(epoch_times)


        elif (np.ndim(epoch_times) == 0):
            # single start time only
            if t_column_names is None:
                t_column_names = ['start',]
            if len(t_column_names) > 1:
                t_column_names = [t_column_names[0],]
            epoch_times = np.atleast_1d(epoch_times)
            num_query_times: int = 1

        else:
            raise NotImplementedError

        # start, stop epoch times:
        epoch_slices_df = a_df[t_column_names]

        epoch_time_to_index_map, found_data_indicies = _subfn_find_epoch_times_map(epoch_slices_df=epoch_slices_df, epoch_times=epoch_times, active_t_column_names=t_column_names, ndim=len(t_column_names))
        if not_found_action == 'skip_index':
            # skip without adding this index. This means the the output found_data_indicies might be smaller than the num_query_times
            assert (len(found_data_indicies) <= num_query_times), f"num_query_times: {num_query_times}, len(found_data_indicies): {len(found_data_indicies)}"
        elif (not_found_action == 'full_nan'):
            assert (len(found_data_indicies) == num_query_times), f"num_query_times: {num_query_times}, len(found_data_indicies): {len(found_data_indicies)}"
        else:
            raise NotImplementedError(f"not_found_action: {not_found_action}")

        # , found_data_indicies
        
        return epoch_time_to_index_map


    @classmethod
    def find_epochs_overlapping_other_epochs(cls, epochs_df: pd.DataFrame, epochs_df_required_to_overlap: pd.DataFrame):
        """ 
        For example, you might wonder which epochs occur during laps:

            from neuropy.core.epoch import EpochHelpers

            ## INPUTS: time_bin_containers, global_laps
            left_edges = deepcopy(time_bin_containers.left_edges)
            right_edges = deepcopy(time_bin_containers.right_edges)
            continuous_time_binned_computation_epochs_df: pd.DataFrame = pd.DataFrame({'start': left_edges, 'stop': right_edges, 'label': np.arange(len(left_edges))})
            is_included: NDArray = EpochHelpers.find_epochs_overlapping_other_epochs(epochs_df=continuous_time_binned_computation_epochs_df, epochs_df_required_to_overlap=deepcopy(global_laps))
            continuous_time_binned_computation_epochs_df['is_in_laps'] = is_included
            continuous_time_binned_computation_epochs_df
            continuous_time_binned_computation_epochs_included_only_df = continuous_time_binned_computation_epochs_df[continuous_time_binned_computation_epochs_df['is_in_laps']].drop(columns=['is_in_laps'])
            continuous_time_binned_computation_epochs_included_only_df

        """
        ## INPUTS: continuous_time_binned_computation_epochs, laps
        epochs_df: pd.DataFrame = ensure_dataframe(epochs_df)
        epochs_df_required_to_overlap: pd.DataFrame = ensure_dataframe(epochs_df_required_to_overlap)
        epochs_df_required_to_overlap_portion: P.Interval = epochs_df_required_to_overlap.epochs.to_PortionInterval()
        continuous_time_binned_computation_epochs_portion_intervals: List[P.Interval] = [P.closedopen(a_row.start, a_row.stop) for a_row in epochs_df[['start', 'stop']].itertuples()]
        is_included: NDArray = np.array([an_interval.overlaps(epochs_df_required_to_overlap_portion) for an_interval in continuous_time_binned_computation_epochs_portion_intervals])
        return is_included


    @classmethod
    def sample_random_period_from_epoch(cls, epoch_start: float, epoch_stop: float, training_data_portion: float, *additional_lap_columns, debug_print=False, debug_override_training_start_t=None):
        """ randomly sample a portion of each lap. Draw a random period of duration (duration[i] * training_data_portion) from the lap.

        """
        total_epoch_duration: float = (epoch_stop - epoch_start)
        training_duration: float = total_epoch_duration * training_data_portion
        test_duration: float = total_epoch_duration - training_duration

        ## new method:
        # I'd like to randomly choose a test_start_t period from any time during the interval.

        # TRAINING data split mode:
        if debug_override_training_start_t is not None:
            print(f'debug_override_training_start_t: {debug_override_training_start_t} provided, so not generating random number.')
            training_start_t = debug_override_training_start_t
        else:
            training_start_t = np.random.uniform(epoch_start, epoch_stop)
        
        training_end_t = (training_start_t + training_duration)
        
        if debug_print:
            print(f'training_start_t: {training_start_t}, training_end_t: {training_end_t}') # , training_wrap_duration: {training_wrap_duration}

        if training_end_t > epoch_stop:
            # Wrap around if training_end_t is beyond the period (wrap required):
            # CASE: [train[0], test[0], train[1]] - train[1] = (train
            # Calculate how much time should wrap to the beginning
            wrap_duration = training_end_t - epoch_stop
            
            # Define the training periods
            train_period_1 = (training_start_t, epoch_stop, *additional_lap_columns) # training spans to the end of the lap
            train_period_2 = (epoch_start, (epoch_start + wrap_duration), *additional_lap_columns) ## new period is crated for training at start of lap
            
            # Return both training periods
            train_outputs = [train_period_1, train_period_2]
        else:
            # all other cases have only one train interval (train[0])
            train_outputs = [(training_start_t, training_end_t, *additional_lap_columns)]


        train_outputs.sort(key=lambda i: (i[0], i[1])) # sort by low first, then by high if the low keys tie
        return train_outputs


    @classmethod
    def split_epochs_into_training_and_test(cls, epochs_df: pd.DataFrame, training_data_portion: float=5.0/6.0, group_column_name: str='lap_id', additional_epoch_identity_column_names=['label', 'lap_id', 'lap_dir'], debug_print: bool = False):
        """ Splits laps into separate training and test sections
        
        Usage:

            from neuropy.core.epoch import EpochHelpers

            ### Get the laps to train on
            training_data_portion: float = 5.0/6.0
            test_data_portion: float = 1.0 - training_data_portion # test data portion is 1/6 of the total duration

            print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')

            laps_df: pd.DataFrame = deepcopy(global_any_laps_epochs_obj.to_dataframe())

            laps_training_df, laps_test_df = EpochHelpers.split_epochs_into_training_and_test(laps_df=laps_df, training_data_portion=training_data_portion, debug_print=False)

            laps_df
            laps_training_df
            laps_test_df 

        """
        from neuropy.core.epoch import Epoch, ensure_dataframe

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # Randomly sample a portion of each lap. Draw a random period of duration (duration[i] * training_data_portion) from the lap.
        train_rows = []
        test_rows = []

        for epoch_id, group in epochs_df.groupby(group_column_name):
            lap_start = group['start'].min()
            lap_stop = group['stop'].max()
            curr_lap_duration: float = lap_stop - lap_start
            if debug_print:
                print(f'lap_id: {epoch_id} - group: {group}')
            curr_additional_lap_column_values = [group[a_col].to_numpy()[0] for a_col in additional_epoch_identity_column_names]
            if debug_print:
                print(f'\tcurr_additional_lap_column_values: {curr_additional_lap_column_values}')
            # Get the random training start and stop times for the lap.
            # Define your period as an interval
            curr_lap_period = P.closed(lap_start, lap_stop)
            epoch_start_stop_tuple_list = cls.sample_random_period_from_epoch(lap_start, lap_stop, training_data_portion, *curr_additional_lap_column_values)

            a_combined_intervals = P.empty()
            for an_epoch_start_stop_tuple in epoch_start_stop_tuple_list:
                a_combined_intervals = a_combined_intervals.union(P.closed(an_epoch_start_stop_tuple[0], an_epoch_start_stop_tuple[1]))
                train_rows.append(an_epoch_start_stop_tuple)
            
            # Calculate the difference between the period and the combined interval
            complement_intervals = curr_lap_period.difference(a_combined_intervals)
            _temp_test_epochs_df = EpochsAccessor.from_PortionInterval(complement_intervals)
            _temp_test_epochs_df[additional_epoch_identity_column_names] = curr_additional_lap_column_values ## add in the additional columns
            test_rows.append(_temp_test_epochs_df)

            ## VALIDATE:
            a_train_durations = [(an_epoch_start_stop_tuple[1]-an_epoch_start_stop_tuple[0]) for an_epoch_start_stop_tuple in epoch_start_stop_tuple_list]
            all_train_durations: float = np.sum(a_train_durations)
            all_test_durations: float = _temp_test_epochs_df['duration'].sum()
            assert np.isclose(curr_lap_duration, (all_train_durations+all_test_durations)), f"(all_train_durations: {all_train_durations} + all_test_durations: {all_test_durations}) should equal curr_lap_duration: {curr_lap_duration}, but instead it equals {(all_train_durations+all_test_durations)}"


        ## INPUTS: laps_df, laps_df

        # train_rows
        # Convert to DataFrame and reset indices
        epochs_training_df = pd.DataFrame(train_rows, columns=['start', 'stop', *additional_epoch_identity_column_names])
        epochs_training_df['duration'] = epochs_training_df['stop'] - epochs_training_df['start']


        # Convert to DataFrame and reset indices
        epochs_test_df = pd.concat(test_rows)
        epochs_training_df.reset_index(drop=True, inplace=True)
        epochs_test_df.reset_index(drop=True, inplace=True)

        # assert np.shape(laps_test_df)[0] == np.shape(laps_df)[0], f"np.shape(laps_test_df)[0]: {np.shape(laps_test_df)[0]} != np.shape(laps_df)[0]: {np.shape(laps_df)[0]}"

        ## OUTPUTS: epochs_training_df, epochs_test_df
        return epochs_training_df, epochs_test_df


    @classmethod
    def subdivide_epochs(cls, df: pd.DataFrame, subdivide_bin_size: float, start_col='start', stop_col='stop') -> pd.DataFrame:
        """ splits each epoch into equally sized chunks determined by subidivide_bin_size.
        
        # Example usage
            from neuropy.core.epoch import EpochHelpers, ensure_dataframe

            df: pd.DataFrame = ensure_dataframe(deepcopy(long_LR_epochs_obj)) 
            df['epoch_type'] = 'lap'
            df['interval_type_id'] = 666

            subdivide_bin_size = 0.100  # Specify the size of each sub-epoch in seconds
            subdivided_df: pd.DataFrame = EpochHelpers.subdivide_epochs(df, subdivide_bin_size)
            print(subdivided_df)

        """
        sub_epochs = []

        # extra_column_names = list(set(df.columns) - set([start_col, stop_col, 'label', 'duration', 'lap_id', 'lap_dir', 'epoch_t_bin_idx', 'epoch_num_t_bins']))
        extra_column_names = list(set(df.columns) - set([start_col, stop_col, 'label', 'duration'])) # ['lap_id', 'lap_dir', 'epoch_t_bin_idx', 'epoch_num_t_bins']
        # print(f'extra_column_names: {extra_column_names}')

        for index, row in df.iterrows():
            start = row[start_col]
            stop = row[stop_col]
            
            duration = stop - start
            num_bins: int = int(duration / subdivide_bin_size) + 1
            
            for i in range(num_bins):
                sub_start = start + (i * subdivide_bin_size)
                sub_stop = min(start + (i + 1) * subdivide_bin_size, stop)
                sub_duration = sub_stop - sub_start
                sub_epochs.append({
                    start_col: sub_start,
                    stop_col: sub_stop,
                    'label': row['label'],
                    'duration': sub_duration,
                    # 'lap_id': row['lap_id'],
                    # 'lap_dir': row['lap_dir'],
                    # 'epoch_t_bin_idx': i,
                    # 'epoch_num_t_bins': num_bins,
                    **{k:row[k] for k in extra_column_names},
                })
        
        sub_epochs_df = pd.DataFrame(sub_epochs)
        return sub_epochs_df


    @classmethod
    def init_epochs_df_from_time_bin_edges(cls, time_bin_edges: NDArray, EPSILON_GAP_SIZE_SEC: Optional[float] = None) -> pd.DataFrame:
        """ converts a series of sequential time_bin_edges (time_window_edges) representing the edges of adjacent time windows to an epoch-like ('start', 'stop')-df by constructing epochs between each time point.
        
        EPSILON_GAP_SIZE_SEC: float = 1e-9 :: time (in seconds) to subtract from start time so the produced epochs don't techinically overlap.
                
        Usage:
        
            from neuropy.core.epoch import EpochHelpers, ensure_dataframe
        
            pos_df, time_windows = pos_df.time_point_event.adding_fixed_length_chunk_columns(subdivide_bin_size=5.0)
            subdivided_epochs_df: pd.DataFrame = EpochHelpers.init_epochs_df_from_time_bin_edges(time_bin_edges=time_windows)
            subdivided_epochs_df

        """
        if len(time_bin_edges) < 1:
            print(f'warn: empty time_bin_edges!')
            return pd.DataFrame({})

        # assert time_col_name in subdivided_epochs_df, f"time_col_name: {time_col_name} not in df.columns: {list(subdivided_epochs_df.columns)}"
        subdivided_epochs_df: pd.DataFrame = pd.DataFrame({'start': time_bin_edges[:-1], 'stop': time_bin_edges[1:]})
        subdivided_epochs_df['duration'] = subdivided_epochs_df['stop'] - subdivided_epochs_df['start']
        subdivided_epochs_df['label'] = subdivided_epochs_df.index.astype('uint64') #['stop'] - subdivided_epochs_df['start']
        
        if (EPSILON_GAP_SIZE_SEC is not None):
            ## ensure all the gaps are greater than the gap size if we';re going to use it.
            min_epoch_duration: float = np.nanmin(np.abs(subdivided_epochs_df['duration'].to_numpy())) # 0.008333336005307501
            assert min_epoch_duration > EPSILON_GAP_SIZE_SEC, f"min_epoch_duration: {min_epoch_duration} must be > EPSILON_GAP_SIZE_SEC: {EPSILON_GAP_SIZE_SEC} if EPSILON_GAP_SIZE_SEC is provided, but it is not."
            subdivided_epochs_df['stop'] = subdivided_epochs_df['stop'] - EPSILON_GAP_SIZE_SEC ## subtract off the EPSILON_GAP_SIZE_SEC if not None
            subdivided_epochs_df['duration'] = subdivided_epochs_df['stop'] - subdivided_epochs_df['start'] ## update duration 

        return subdivided_epochs_df



""" 
from neuropy.core.epoch import NamedTimerange, EpochsAccessor, Epoch

"""
class NamedTimerange(SimplePrintable, metaclass=OrderedMeta):
    """ A simple named period of time with a known start and end time """
    def __init__(self, name, start_end_times):
        self.name = name
        self.start_end_times = start_end_times
        
    @property
    def t_start(self):
        return self.start_end_times[0]
    
    @t_start.setter
    def t_start(self, t):
        self.start_end_times[0] = t

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.start_end_times[1]
    
    @t_stop.setter
    def t_stop(self, t):
        self.start_end_times[1] = t
    
    
    def to_Epoch(self):
        return Epoch(pd.DataFrame({'start': [self.t_start], 'stop': [self.t_stop], 'label':[self.name]}))
        

@pd.api.extensions.register_dataframe_accessor("epochs")
class EpochsAccessor(TimeColumnAliasesProtocol, TimeSlicedMixin, StartStopTimesMixin, TimeSlicableObjectProtocol, DataframeMetadataProtocol):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals """
    
    _time_column_name_synonyms = {"start":{'begin','start_t'},
            "stop":['end','stop_t'],
            "label":['name', 'id', 'flat_replay_idx','lap_id']
        }
    

    _required_column_names = ['start', 'stop', 'label', 'duration']


    # Define constants for numerical precision
    EPSILON_OVERLAP_COMPARE_TOL_SEC: float = 1e-15  # Tolerance for numerical comparisons
    EPSILON_GAP_SIZE_SEC: float = 1e-9  # Amount to subtract from stop times

    def __init__(self, pandas_obj):
        # pandas_obj = self.renaming_synonym_columns_if_needed(pandas_obj, required_columns_synonym_dict=self._time_column_name_synonyms, fail_on_missing_columns=False)  #@IgnoreException 
        pandas_obj = self.renaming_synonym_columns_if_needed(pandas_obj, required_columns_synonym_dict={k:v for k, v in self._time_column_name_synonyms.items() if k in ['start', 'stop']}, fail_on_missing_columns=True)  #@IgnoreException         
        pandas_obj = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj = self._obj.sort_values(by=["start"]) # sorts all values in ascending order
        # Optional: If the 'label' column of the dataframe is empty, should populate it with the index (after sorting) as a string.
        # self._obj['label'] = self._obj.index
        self._obj["label"] = self._obj["label"].astype("str")
        # Optional: Add 'duration' column:
        self._obj["duration"] = self._obj["stop"] - self._obj["start"]
        # Optional: check for and remove overlaps
        self._obj = self.renaming_synonym_columns_if_needed(self._obj, required_columns_synonym_dict=self._time_column_name_synonyms, fail_on_missing_columns=True)  #@IgnoreException 
        


    @classmethod
    def init_from_time_bin_edges(cls, time_bin_edges: NDArray, EPSILON_GAP_SIZE_SEC: Optional[float] = None) -> pd.DataFrame:
        """ converts a series of sequential time_bin_edges (time_window_edges) representing the edges of adjacent time windows to an epoch-like ('start', 'stop')-df by constructing epochs between each time point.
        
        EPSILON_GAP_SIZE_SEC: float = 1e-9 :: time (in seconds) to subtract from start time so the produced epochs don't techinically overlap.
                
        Usage:
        
            from neuropy.core.epoch import EpochsAccessor
        
            pos_df, time_windows = pos_df.time_point_event.adding_fixed_length_chunk_columns(subdivide_bin_size=5.0)
            subdivided_epochs_df: pd.DataFrame = EpochsAccessor.init_epochs_df_from_time_bin_edges(time_bin_edges=time_windows)
            subdivided_epochs_df

        """
        return EpochHelpers.init_epochs_df_from_time_bin_edges(time_bin_edges=time_bin_edges, EPSILON_GAP_SIZE_SEC=EPSILON_GAP_SIZE_SEC)


    @classmethod
    def _validate(cls, obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        return obj # important! Must return the modified obj to be assigned (since its columns were altered by renaming

    @property
    def is_valid(self):
        """ The dataframe is valid (because it passed _validate(...) in __init__(...) so just return True."""
        return True

    @property
    def starts(self):
        return self._obj.start.values

    @property
    def midtimes(self): # -> NDArray
        """ since each epoch is given by a (start, stop) time, the midtimes are the center of this epoch. """
        return self._obj.start.values + ((self._obj.stop.values - self._obj.start.values)/2.0)

    @property
    def stops(self):
        return self._obj.stop.values
    
    @property
    def t_start(self):
        return self.starts[0]
    @t_start.setter
    def t_start(self, t):
        include_indicies = np.argwhere(t < self.stops)
        if (np.size(include_indicies) == 0):
            # this proposed t_start is after any contained epochs, so the returned object would be empty
            print('Error: this proposed t_start ({}) is after any contained epochs, so the returned object would be empty'.format(t))
            raise ValueError
        first_include_index = include_indicies[0]
        
        if (first_include_index > 0):
            # drop the epochs preceeding the first_include_index:
            drop_indicies = np.arange(first_include_index)
            print('drop_indicies: {}'.format(drop_indicies))
            raise NotImplementedError # doesn't yet drop the indicies before the first_include_index
        self._obj.loc[first_include_index, ('start')] = t # exclude the first short period where the animal isn't on the maze yet

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.stops[-1]

    @property
    def durations(self):
        return self.stops - self.starts

    @property
    def n_epochs(self):
        return len(self.starts)

    @property
    def labels(self):
        return self._obj.label.values


    @property
    def extra_data_column_names(self):
        """Any additional columns in the dataframe beyond those that exist by default. """
        return list(set(self._obj.columns) - set(self._required_column_names))
        
    @property
    def extra_data_dataframe(self) -> pd.DataFrame:
        """The subset of the dataframe containing additional information in its columns beyond that what is required. """
        return self._obj[self.extra_data_column_names]

    def as_array(self) -> NDArray:
        return self._obj[["start", "stop"]].to_numpy()

    def get_unique_labels(self):
        # return np.unique(self.labels) # this is sorted, which is mostly WRONG
        return self._obj.label.unique()
    
    def rebuild_labels_column(self, reset_df_index: bool=True):
        """ resets the index (by default) and then rebuilds the labels. """
        if reset_df_index:
            self._obj = self._obj.reset_index(drop=True)
        self._obj["label"] = self._obj.index.astype("str")
        return self._obj

    def get_start_stop_tuples_list(self):
        """ returns a list of (start, stop) tuples. """
        return list(zip(self.starts, self.stops))

    def get_valid_df(self) -> pd.DataFrame:
        """ gets a validated copy of the dataframe. Looks better than doing `epochs_df.epochs._obj` """
        return self._obj.copy()

    # @classmethod
    # def _mergable(cls, a, b):
    #     """ 
    #     NOT YET IMPLEMENTED. Based off of Portion's mergable operation with intent to extend to Epochs.

    #     """
    #     # a - a single period in time
    #     # b - a single potentially overlapping period in time
    #     a_start, a_end = a # unwrap
    #     b_start, b_end = b # unwrap
    #     ## Check their lower bounds first

    #     ## Check their upper bounds for overlap

    
    # ==================================================================================================================== #
    # Handling overlapping                                                                                                 #
    # ==================================================================================================================== #

    def is_gapless_overlap_df(self) -> bool:
        """ checks whether all of epochs are back-to-back with no intermediate overlaps, returning true if they are 
        """
        # Early exit for empty or single-element dataframes
        if len(self._obj) <= 1:
            return True
        
        # The dataframe is already sorted in __init__ method, so we can use it directly
        # Get numpy arrays of starts and stops for fast operations
        starts = self._obj['start'].values
        stops = self._obj['stop'].values
        
        # Vectorized check for back-to-back epochs:
        # stops[:-1] = all stop times except the last one
        # starts[1:] = all start times except the first one
        # If the epochs are back-to-back, these should be equal
        return np.array_equal(stops[:-1], starts[1:])

    def get_non_overlapping_df(self, debug_print=False) -> pd.DataFrame:
        """ 
        Returns a dataframe with overlapping epochs removed.
        
        The algorithm:
        1. Identifies and handles gapless epoch pairs by applying a small epsilon offset
        2. For any remaining overlaps, uses PortionInterval to resolve them
        
        Parameters:
        -----------
        debug_print : bool, optional
            If True, prints debug information about the processing. Default is False.
            
        Returns:
        --------
        pd.DataFrame
            A dataframe with no overlapping epochs


        NOTE: Code generated with aid of AI - Claude 3.7 - 2025-03-10 09:23
        """
        # Quick return for empty or single-element dataframes
        if len(self._obj) <= 1:
            return self._obj.copy()
        
        # Preserve DataFrame metadata/attributes if present
        df_metadata = None
        if hasattr(self._obj, 'attrs') and self._obj.attrs is not None:
            from copy import deepcopy
            df_metadata = deepcopy(self._obj.attrs)
                
        # Create a copy of the dataframe to modify
        modified_df = self._obj.copy()
        # modified_df = modified_df.sort_values(by=["start"]).reset_index(drop=True) ## this will mis-sort the global epoch
        modified_df = modified_df.sort_values(by=["stop", "start"]).reset_index(drop=True) ## this should preserve the global epoch at the end property while keeping the other in the right place
        
        # Find pairs where stop[i] == start[i+1] (gapless)
        stops = modified_df['stop'].values[:-1]  # all except last
        next_starts = modified_df['start'].values[1:]  # all except first
        # Create a boolean mask for gapless pairs - use vectorized comparison
        gapless_mask = np.isclose(stops, next_starts, rtol=self.__class__.EPSILON_OVERLAP_COMPARE_TOL_SEC, atol=self.__class__.EPSILON_OVERLAP_COMPARE_TOL_SEC)
        
        # Only proceed with gapless handling if gapless pairs exist
        if np.any(gapless_mask):
            # Get indices of epochs that end at exactly the start of the next epoch
            gapless_indices = np.where(gapless_mask)[0]
            
            # Apply epsilon to the stop times of gapless epochs - use vectorized operation
            modified_df.iloc[gapless_indices, modified_df.columns.get_loc('stop')] -= self.__class__.EPSILON_GAP_SIZE_SEC
            
            # Only recalculate duration if it exists - use vectorized operation
            if 'duration' in modified_df.columns:
                modified_df['duration'] = modified_df['stop'] - modified_df['start']
            
            if debug_print:
                print(f'Applied gapless optimization to {len(gapless_indices)} epoch pairs')
        
        # Check if any epochs still overlap after the gapless handling
        # This allows us to skip the PortionInterval processing if possible
        if len(modified_df) <= 1:
            result_df = modified_df
        else:
            # Sort the dataframe by start time (should already be sorted, but ensure it)
            # modified_df = modified_df.sort_values(by=["start"]).reset_index(drop=True)
            modified_df = modified_df.sort_values(by=["stop", "start"]).reset_index(drop=True) ## this should preserve the global epoch at the end property while keeping the other in the right place
            # Check if any epochs overlap after gapless handling
            # This is a fast check that can avoid the more expensive PortionInterval processing
            remaining_stops = modified_df['stop'].values[:-1]
            remaining_next_starts = modified_df['start'].values[1:]
            any_overlaps = np.any(remaining_stops > remaining_next_starts)
            # Only use PortionInterval if there are still overlaps
            if any_overlaps:
                from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df, _convert_start_end_tuples_list_to_PortionInterval
                # Set up PortionInterval kwargs based on codebase patterns
                P_Interval_kwargs = {'merge_on_adjacent': False, 'enable_auto_simplification': True}
                # Get a list of extra columns that aren't part of the standard epochs columns
                extra_columns = [col for col in modified_df.columns if col not in ['start', 'stop', 'label', 'duration']]
                # Convert modified dataframe to PortionInterval
                _intermediate_portions_interval = _convert_start_end_tuples_list_to_PortionInterval(zip(modified_df['start'].values, modified_df['stop'].values), **P_Interval_kwargs)
                # Convert back to dataframe, resolving any remaining overlaps
                result_df = convert_PortionInterval_to_epochs_df(_intermediate_portions_interval)
                # Handle extra columns if possible
                if extra_columns:
                    # Find correspondences between old and new rows if possible
                    # This is a simple approach - more complex matching might be needed for some cases
                    if debug_print and len(result_df) != len(modified_df):
                        print(f"Warning: Row count changed from {len(modified_df)} to {len(result_df)}. Extra columns may not be preserved correctly.")
                    
                    # Try to preserve extra column data using the epochs.find_data_indicies_from_epoch_times method if available
                    try:
                        if hasattr(result_df, 'epochs') and hasattr(result_df.epochs, 'find_data_indicies_from_epoch_times'):
                            epoch_times = result_df[['start', 'stop']].to_numpy()
                            indices = EpochHelpers.find_data_indicies_from_epoch_times(modified_df, epoch_times, atol=self.__class__.EPSILON_GAP_SIZE_SEC*10)
                            # Copy extra columns from original dataframe where indices were found
                            for col in extra_columns:
                                if len(indices) > 0:
                                    result_df[col] = modified_df.iloc[indices][col].reset_index(drop=True)
                        else:
                            # Fallback for simpler cases where row count doesn't change
                            if len(result_df) == len(modified_df):
                                for col in extra_columns:
                                    result_df[col] = modified_df[col].values
                    except Exception as e:
                        if debug_print:
                            print(f"Error preserving extra columns: {e}")
    
                if debug_print:
                    before_num_rows = self.n_epochs
                    after_num_rows = np.shape(result_df)[0]
                    changed_num_rows = after_num_rows - before_num_rows
                    print(f'Dataframe Changed from {before_num_rows} -> {after_num_rows} ({changed_num_rows = })')
            else:
                # No overlaps remain, so we can return the modified dataframe directly
                if debug_print:
                    print(f'No overlaps found after gapless processing, skipping PortionInterval')

                result_df = modified_df
        
        # Apply metadata to the result before returning
        if df_metadata is not None and hasattr(result_df, 'attrs'):
            result_df.attrs = df_metadata
        
        return result_df


    def get_epochs_longer_than(self, minimum_duration: float, debug_print=False) -> pd.DataFrame:
        """ returns a copy of the dataframe contining only epochs longer than the specified minimum_duration. """
        active_filter_epochs = self.get_valid_df()
        if debug_print:
            before_num_rows = np.shape(active_filter_epochs)[0]
        if 'duration' not in active_filter_epochs.columns:
            active_filter_epochs['duration'] = active_filter_epochs['stop'] - active_filter_epochs['start']
        if debug_print:
            filtered_epochs = active_filter_epochs[active_filter_epochs['duration'] >= minimum_duration]
            after_num_rows = np.shape(filtered_epochs)[0]
            changed_num_rows = after_num_rows - before_num_rows
            print(f'Dataframe Changed from {before_num_rows} -> {after_num_rows} ({changed_num_rows = })')
            return filtered_epochs
        else:
            return active_filter_epochs[active_filter_epochs['duration'] >= minimum_duration]


    def merge_adjacent_epochs_within(self, max_merge_duration: float, omit_unspecified_column_merge_rules: bool=True, **column_merge_rules) -> pd.DataFrame:
        """Merge consecutive epochs whose separation is less than or equal to ``max_separation``.

        
        omit_unspecified_column_merge_rules: bool = True, if True, only columns explicitly specified in `column_merge_rules` are used, no values are inferred, otherwise, values are intellegently inferred.
        
        
            column_merge_rules : dict[str, str | callable], optional
        Per-column merge policy. Values may be:
            - predefined string strategies ('first','last','mean','sum','min',
              'max','concat','unique_concat','list','set','any','all')
            - callable: f(pd.Series) -> scalar
            
            
            - 'require_same' : return the value only if all values in the merge block
                 are identical (NaNs count as equal); otherwise raise.

                 
            
        Rules:
        - Epochs are treated in start-time order.
        - If the gap (next.start - current.stop) <= max_separation (+ small numeric tolerance), they are merged.
        - Overlapping epochs (negative gap) also merge.
        - ``label`` handling:
            - 'first' (default): keep the first epoch's label in a merged run
            - 'concat': concatenate unique labels with '+' in encounter order

        Returns a new dataframe with columns ['start','stop','label','duration'].
        
        Does not modify in place.
        """
        # ---------------------------
        # Merge strategy resolution
        # ---------------------------
        
        def _subfn_resolve_strategy(values: pd.Series, col: str, omit_unspecified_column_merge_rules: bool):
            """ Resolve merge strategy for a column.
            captures: column_merge_rules 

            Order of precedence:
            1. Explicit user rule
            2. Known column defaults
            3. Dtype-based smart defaults
            4. Fallback: 'omit'
            """
            # 1. Explicit override
            if column_merge_rules and col in column_merge_rules:
                return column_merge_rules[col]

            # 2. Known semantic columns
            if col == 'start':
                return 'min'
            if col == 'stop':
                return 'max'
            if col == 'label':
                return 'concat'
            
            if not omit_unspecified_column_merge_rules:
                # Infer smart Dtype-based defaults
                if pd.api.types.is_bool_dtype(values):
                    return 'any'
                if pd.api.types.is_numeric_dtype(values):
                    return 'mean'
                if pd.api.types.is_object_dtype(values):
                    return 'first'

            # 4. Explicit fallback to omitting
            return 'omit'


        def _subfn_apply_strategy(values: pd.Series, strategy: Union[Callable, str]):
            """ applys the strategy to the particular column 
            """
            if callable(strategy):
                return strategy(values)
            
            if strategy in {'mean', 'sum', 'min', 'max'}:
                if not pd.api.types.is_numeric_dtype(values):
                    raise TypeError(f"Merge strategy '{strategy}' requires numeric dtype for column '{values.name}'")
            elif strategy in {'any', 'all'}:
                if not pd.api.types.is_bool_dtype(values):
                    raise TypeError(f"Merge strategy '{strategy}' requires boolean dtype for column '{values.name}'")
                
            if strategy == 'first':
                return values.iloc[0]
            if strategy == 'last':
                return values.iloc[-1]
            if strategy == 'mean':
                return values.mean()
            if strategy == 'sum':
                return values.sum()
            if strategy == 'min':
                return values.min()
            if strategy == 'max':
                return values.max()
            if strategy == 'concat':
                return '+'.join(map(str, values))
            if strategy == 'unique_concat':
                return '+'.join(dict.fromkeys(map(str, values)))
            if strategy == 'list':
                return list(values)
            if strategy == 'set':
                return set(values)
            if strategy == 'any':
                return values.any()
            if strategy == 'all':
                return values.all()

            if strategy == 'require_same':
                # Treat NaNs as equal
                if values.isna().all():
                    return values.iloc[0]

                non_na = values.dropna()

                if non_na.nunique(dropna=False) != 1:
                    raise ValueError(
                        f"Merge strategy 'require_same' violated for column '{values.name}': \n"
                        f"unique={np.unique(list(values)).tolist()}\n"
                        f"values={list(values)}"
                    )

                return non_na.iloc[0]



            raise ValueError(f"Unknown merge strategy: {strategy}")


        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        assert max_merge_duration is not None and max_merge_duration >= 0.0, f"max_separation must be >= 0.0, got {max_merge_duration}"

        # Quick return for trivial sizes
        if self.n_epochs <= 1:
            result_df = self.get_valid_df()
            if hasattr(self._obj, 'attrs') and (self._obj.attrs is not None):
                result_df.attrs = deepcopy(self._obj.attrs)
            return result_df

        #TODO 2025-10-21 09:31: - [ ] ChatGPT-5 Implementation:
        df = self.get_valid_df().sort_values('start').reset_index(drop=True)
        
        # ---------------------------
        # Merge runs
        # ---------------------------
        merged_rows = []
        run_start_idx = 0

        for i in range(1, len(df)):
            gap = df.loc[i, 'start'] - df.loc[i - 1, 'stop']
            if gap > max_merge_duration:
                merged_rows.append((run_start_idx, i - 1))
                run_start_idx = i

        merged_rows.append((run_start_idx, len(df) - 1))

        # ---------------------------
        # Build merged dataframe
        # ---------------------------
        merged_records = []
        for start_idx, stop_idx in merged_rows:
            block = df.iloc[start_idx:stop_idx + 1]
            record = {}
            for col in df.columns:
                strategy = _subfn_resolve_strategy(block[col], col, omit_unspecified_column_merge_rules=omit_unspecified_column_merge_rules)
                if strategy == 'omit':
                    continue
                record[col] = _subfn_apply_strategy(block[col], strategy)

            merged_records.append(record)

        merged_df = pd.DataFrame.from_records(merged_records)
        merged_df['duration'] = merged_df['stop'] - merged_df['start']

        # ==================================================================================================================================================================================================================================================================================== #
        # Pho pre-2026-01-23 logic                                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        # merged = []
        # current_start = df.iloc[0]['start']
        # current_stop = df.iloc[0]['stop']
        # current_label = df.iloc[0]['label']

        # for i in range(1, len(df)):
        #     next_start = df.iloc[i]['start']
        #     next_stop = df.iloc[i]['stop']
        #     next_label = df.iloc[i]['label']
        #     gap = next_start - current_stop

        #     if gap <= max_merge_duration:
        #         # Merge with current run
        #         current_stop = max(current_stop, next_stop)
        #         current_label = f"{current_label}+{next_label}"
        #     else:
        #         # Push the previous run
        #         merged.append((current_start, current_stop, current_label))
        #         current_start, current_stop, current_label = next_start, next_stop, next_label

        # merged.append((current_start, current_stop, current_label))

        # merged_df = pd.DataFrame(merged, columns=['start', 'stop', 'label'])
        # merged_df['duration'] = merged_df['stop'] - merged_df['start']
        # # merged_df['label'] = merged_df.index.astype('str')
        # merged_df['label'] = merged_df['label'].astype('str')
        

        if hasattr(self._obj, 'attrs') and self._obj.attrs is not None:
            merged_df.attrs = deepcopy(self._obj.attrs)

        return merged_df


        # # raise NotImplementedError(f'This looses all other columss when merging!!')
        # # Work on a validated, sorted copy
        # df = self.get_valid_df().sort_values(by=["start"]).reset_index(drop=True)

        # merged_rows: List[Dict[str, Union[float, str]]] = []

        # curr_start = float(df.loc[0, 'start'])
        # curr_stop = float(df.loc[0, 'stop'])
        # curr_labels: List[str] = [str(df.loc[0, 'label'])]

        # tol = self.__class__.EPSILON_OVERLAP_COMPARE_TOL_SEC

        # for i in range(1, len(df)):
        #     next_start = float(df.loc[i, 'start'])
        #     next_stop = float(df.loc[i, 'stop'])
        #     next_label = str(df.loc[i, 'label'])

        #     gap = next_start - curr_stop
        #     if gap <= (max_separation + tol):
        #         # Merge into current run
        #         if next_stop > curr_stop:
        #             curr_stop = next_stop
        #         if label_merge_mode == 'concat':
        #             if (len(curr_labels) == 0) or (next_label != curr_labels[-1]):
        #                 # keep encounter order, avoid immediate duplicates
        #                 if next_label not in curr_labels:
        #                     curr_labels.append(next_label)
        #         # 'first' keeps original label
        #     else:
        #         # Finalize current run
        #         if label_merge_mode == 'concat':
        #             out_label = '+'.join(curr_labels)
        #         else:
        #             out_label = curr_labels[0]
        #         merged_rows.append({'start': curr_start, 'stop': curr_stop, 'label': out_label})

        #         # Start new run
        #         curr_start = next_start
        #         curr_stop = next_stop
        #         curr_labels = [next_label]

        # # Finalize last run
        # if label_merge_mode == 'concat':
        #     out_label = '+'.join(curr_labels)
        # else:
        #     out_label = curr_labels[0]
        # merged_rows.append({'start': curr_start, 'stop': curr_stop, 'label': out_label})

        # result_df = pd.DataFrame(merged_rows, columns=['start', 'stop', 'label'])
        # # Ensure dtypes
        # result_df[['start', 'stop']] = result_df[['start', 'stop']].astype(float)
        # result_df['label'] = result_df['label'].astype('str')
        # result_df['duration'] = result_df['stop'] - result_df['start']

        # if debug_print:
        #     before_num_rows = self.n_epochs
        #     after_num_rows = np.shape(result_df)[0]
        #     changed_num_rows = after_num_rows - before_num_rows
        #     print(f'Merged adjacent epochs within {max_separation} s: {before_num_rows} -> {after_num_rows} ({changed_num_rows = })')

        # if copy_metadata and hasattr(self._obj, 'attrs') and (self._obj.attrs is not None):
        #     from copy import deepcopy
        #     result_df.attrs = deepcopy(self._obj.attrs)

        # return result_df

    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop) -> pd.DataFrame:
        """ trim the epochs down to the provided time range
        
        """
        # TODO time_slice should also include partial epochs falling in between the timepoints
        df = self._obj.copy() 
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        df = df[(df["start"] >= t_start) & (df["start"] < t_stop)].reset_index(drop=True) # 2023-11-13 - changed to `(df["start"] < t_stop)` from `(df["start"] <= t_stop)` because in the equals case the resulting included interval would be zero duration.
        return df
        
    def label_slice(self, label) -> pd.DataFrame:
        if isinstance(label, (list, NDArray)):
            df = self._obj[np.isin(self._obj["label"], label)].reset_index(drop=True)
        else:
            assert isinstance(label, str), "label must be string"
            df = self._obj[self._obj["label"] == label].reset_index(drop=True)
        return df

    def find_data_indicies_from_epoch_times(self, epoch_times: NDArray, atol:float=1e-3, t_column_names=None) -> NDArray:
        """ returns the matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        Uses:
            self.plots_data.epoch_slices
        
        - [X] FIXED 2024-03-04 19:55 - This function was peviously incorrect and could return multiple matches for each passed time due to the tolerance.

        """
        # find_data_indicies_from_epoch_times(a_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t',])
        return EpochHelpers.find_data_indicies_from_epoch_times(self._obj, epoch_times, t_column_names=t_column_names, atol=atol, debug_print=False)
    

    def find_epoch_times_to_data_indicies_map(self, epoch_times: NDArray, atol:float=1e-3, t_column_names=None) -> Dict[Union[float, Tuple[float, float]], Union[int, NDArray]]:
        """ returns the a Dict[Union[float, Tuple[float, float]], Union[int, NDArray]] matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        Usage:
            epoch_time_to_index_map = deepcopy(dbgr.active_decoder_decoded_epochs_result).filter_epochs.epochs.find_epoch_times_to_data_indicies_map(epoch_times=[epoch_start_time, ])
        
        """
        return EpochHelpers.find_epoch_times_to_data_indicies_map(self._obj, epoch_times, t_column_names=t_column_names, atol=atol, debug_print=False)
    

            
    def matching_epoch_times_slice(self, epoch_times: NDArray, atol:float=1e-3, t_column_names=None) -> pd.DataFrame:
        """ slices the dataframe to return only the rows that match the epoch_times with some tolerance.
        
        Internally calls self.find_data_indicies_from_epoch_times(...)

        """
        # , not_found_action='skip_index'
        found_data_indicies = self._obj.epochs.find_data_indicies_from_epoch_times(epoch_times=epoch_times, atol=atol, t_column_names=t_column_names)
        # df = self._obj.iloc[found_data_indicies].copy().reset_index(drop=True)
        df = self._obj.loc[found_data_indicies].copy().reset_index(drop=True)
        return df

    def filtered_by_duration(self, min_duration=None, max_duration=None):
        return self._obj[(self.durations >= (min_duration or 0.0)) & (self.durations <= (max_duration or np.inf))].reset_index(drop=True)
        
    # Requires Optional `portion` library
    # import portion as P # Required for interval search: portion~=2.3.0
    @classmethod
    def from_PortionInterval(cls, portion_interval):
        from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df
        return convert_PortionInterval_to_epochs_df(portion_interval)

    def to_PortionInterval(self): # -> "P.Interval"
        from neuropy.utils.efficient_interval_search import _convert_start_end_tuples_list_to_PortionInterval
        return _convert_start_end_tuples_list_to_PortionInterval(zip(self.starts, self.stops))


    def get_in_between(self, copy_metadata:bool=False) -> pd.DataFrame:
        """ Returns the periods between each pair of consecutive epochs
        
        Usage:
            inter_lap_epoch_df: pd.DataFrame = laps_df.epochs.get_in_between()
            inter_lap_epoch_df
            
        """
        epochs_df: pd.DataFrame = self.get_non_overlapping_df()
        df_metadata = None
        if (self._obj.attrs is not None) and copy_metadata:
            df_metadata = deepcopy(self._obj.attrs)
        
        inter_lap_epochs = []
        prev_lap = None
        for a_lap in epochs_df.itertuples():
            if prev_lap is not None:
                inter_lap_epochs.append([float(prev_lap.stop), float(a_lap.start), str(prev_lap.label), str(a_lap.label)])
            prev_lap = a_lap

        inter_epoch_df: pd.DataFrame = pd.DataFrame(np.array(inter_lap_epochs), columns=['start', 'stop', 'precceding_epoch_label', 'following_epoch_label'])
        inter_epoch_df['start'] = inter_epoch_df['start'].astype('float')
        inter_epoch_df['stop'] = inter_epoch_df['stop'].astype('float')
        # Optional: If the 'label' column of the dataframe is empty, should populate it with the index (after sorting) as a string.
        inter_epoch_df['label'] = inter_epoch_df.index
        inter_epoch_df["label"] = inter_epoch_df["label"].astype("str")
        # Optional: Add 'duration' column:
        inter_epoch_df["duration"] = inter_epoch_df["stop"] - inter_epoch_df["start"]
        if df_metadata is not None:
            inter_epoch_df.attrs = df_metadata


        return inter_epoch_df   
        
        
    def modify_each_epoch_by(self, additive_factor: float = 0.0, multiplicative_factor: float=1.0, final_output_minimum_epoch_duration:float=0.0, copy_metadata:bool=False) -> pd.DataFrame:
        """ gets a copy of the epochs where each epoch is modified by the provided scale factors.
        
        if safe_contract_epochs is not 0.0, a value of 0.0 is used to safely
        Usage:
            inter_lap_epoch_df: pd.DataFrame = inter_lap_epoch_df.epochs.modify_each_epoch_by(additive_factor=-0.008, final_output_minimum_epoch_duration=0.040)
            inter_lap_epoch_df
            
        """
        epochs_df: pd.DataFrame = self.get_non_overlapping_df()
        df_metadata = None
        if (self._obj.attrs is not None) and copy_metadata:
            df_metadata = deepcopy(self._obj.attrs)
        

        # smallest_allowed_epoch_duration: float = (2.0 * safe_contract_epochs) + 0.001 # add another ms just to be safe
        # epochs_df = epochs_df.epochs.get_epochs_longer_than(minimum_duration=smallest_allowed_epoch_duration)
        
        if additive_factor != 0.0:
            epochs_df['start'] = epochs_df['start'] - additive_factor 
            epochs_df['stop'] = epochs_df['stop'] + additive_factor
            epochs_df['duration'] = epochs_df['stop'] - epochs_df['start']
                        
        if multiplicative_factor != 1.0:
            epochs_df['duration'] = epochs_df['stop'] - epochs_df['start'] ## ensure duration is up-to-date
            epochs_df['half_duration'] = epochs_df['duration'] * 0.5
            epochs_df['t_mid'] = epochs_df['start'] + epochs_df['half_duration'] ## compute midpoint, which will be unchanged by the multipliciatve_factor  
            epochs_df['duration'] = multiplicative_factor * epochs_df['duration'] ## update the duration
            epochs_df['half_duration'] = epochs_df['duration'] * 0.5 ## updated half-duration
            # epochs_df['t_mid'] = epochs_df['start'] + epochs_df['half_duration'] ## compute updated midpoint
            epochs_df['start'] = epochs_df['t_mid'] - epochs_df['half_duration'] 
            epochs_df['stop'] = epochs_df['t_mid'] + epochs_df['half_duration'] 
            ## drop temporary columns
            epochs_df = epochs_df.drop(columns=['half_duration', 't_mid'], inplace=False)
            
        if final_output_minimum_epoch_duration is not None:
            epochs_df = epochs_df.epochs.get_epochs_longer_than(minimum_duration=final_output_minimum_epoch_duration) ## post-hoc filtering

        if df_metadata is not None:
            epochs_df.attrs = df_metadata

        return epochs_df   
        
    def subtracting(self, other_epochs_df: pd.DataFrame, skip_get_non_overlapping:bool=False) -> pd.DataFrame:
        """ gets a copy of the epochs after subtracting the epochs provided in `other_epochs_df`.
        
        Usage:
            global_epoch_only_non_PBE_epoch_df: pd.DataFrame = global_epoch_only_df.epochs.subtracting(PBE_df)
            global_epoch_only_non_PBE_epoch_df= global_epoch_only_non_PBE_epoch_df.epochs.modify_each_epoch_by(additive_factor=-0.008, final_output_minimum_epoch_duration=0.040)
            global_epoch_only_non_PBE_epoch_df
            
        """
        if not skip_get_non_overlapping:
            epochs_df: pd.DataFrame = self.get_non_overlapping_df()
        else:
            epochs_df: pd.DataFrame = ensure_dataframe(self._obj)
        epochs_Portion: P.Interval = epochs_df.epochs.to_PortionInterval()
        other_epochs_Porition: P.Interval = other_epochs_df.epochs.to_PortionInterval()
        return EpochsAccessor.from_PortionInterval(epochs_Portion.difference(other_epochs_Porition))
        

    def intersecting(self, other_epochs_df: pd.DataFrame, skip_get_non_overlapping:bool=False, enable_splitting_on_partial_intersect: bool=True) -> pd.DataFrame:
        """ gets a copy of the epochs after intersecting with the epochs provided in `other_epochs_df`, getting only those portions which overlap `other_epochs_df`.
        
        Usage:
            global_epoch_only_non_PBE_epoch_df: pd.DataFrame = global_epoch_only_df.epochs.intersecting(PBE_df)
            global_epoch_only_non_PBE_epoch_df= global_epoch_only_non_PBE_epoch_df.epochs.modify_each_epoch_by(additive_factor=-0.008, final_output_minimum_epoch_duration=0.040)
            global_epoch_only_non_PBE_epoch_df
            
        """
        is_in_key: str = 'is_overlapping_intersecting_INTERNAL'
        if not skip_get_non_overlapping:
            epochs_df: pd.DataFrame = self.get_non_overlapping_df()
        else:
            epochs_df: pd.DataFrame = ensure_dataframe(self._obj)

        other_epochs_df = ensure_dataframe(other_epochs_df)
        
        if enable_splitting_on_partial_intersect:
            """ split continuously by intersection with the epochs in `other_epochs_df`. If only a fragment of an interval matches one in `other_epochs_df`, only the overlapping segment is included. """
            epochs_Portion: P.Interval = epochs_df.epochs.to_PortionInterval()
            other_epochs_Porition: P.Interval = other_epochs_df.epochs.to_PortionInterval()
            return EpochsAccessor.from_PortionInterval(epochs_Portion.intersection(other_epochs_Porition))
        
        else:
            """ does not split any existing epoch intervals, only chooses whether to include/exclude them in the resultant series based on whether they overlap (at all, even a little bit) with any epochs in `other_epochs_df` """
            is_included: NDArray = EpochHelpers.find_epochs_overlapping_other_epochs(epochs_df=epochs_df, epochs_df_required_to_overlap=other_epochs_df)
            # is_included: NDArray = EpochHelpers.find_epochs_overlapping_other_epochs(epochs_df=continuous_time_binned_computation_epochs_df, epochs_df_required_to_overlap=deepcopy(global_laps))
            epochs_df[is_in_key] = is_included
            return epochs_df[epochs_df[is_in_key]].drop(columns=[is_in_key], inplace=False)


    def split_into_training_and_test(self, training_data_portion: float=5.0/6.0, group_column_name: str='label', additional_epoch_identity_column_names:List[str]=['label'], skip_get_non_overlapping:bool=False, debug_print: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Splits laps into separate training and test sections
        
        Usage:

            ### Get the laps to train on
            training_data_portion: float = 5.0/6.0
            test_data_portion: float = 1.0 - training_data_portion # test data portion is 1/6 of the total duration

            print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')

            a_laps_df: pd.DataFrame = ensure_dataframe(deepcopy(a_config['pf_params'].computation_epochs))
            a_laps_training_df, a_laps_test_df = a_laps_df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name ='lap_id', additional_epoch_identity_column_names=['label', 'lap_id', 'lap_dir'], skip_get_non_overlapping=False, debug_print=False) # a_laps_training_df, a_laps_test_df both comeback good here.

            laps_df
            laps_training_df
            laps_test_df 
            
        Usage 2:
            training_data_portion: float = 5.0/6.0
            a_new_training_df, a_new_test_df = global_epoch_only_non_PBE_epoch_df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name ='label', additional_epoch_identity_column_names=['label'], skip_get_non_overlapping=False, debug_print=False) # a_laps_training_df, a_laps_test_df both comeback good here.

            a_new_training_df
            a_new_test_df

    

        """
        if not skip_get_non_overlapping:
            epochs_df: pd.DataFrame = self.get_non_overlapping_df()
        else:
            epochs_df: pd.DataFrame = self.get_valid_df()        

        return EpochHelpers.split_epochs_into_training_and_test(epochs_df=epochs_df, training_data_portion=training_data_portion, group_column_name=group_column_name, additional_epoch_identity_column_names=additional_epoch_identity_column_names, debug_print=debug_print)

    
    # Column Adding/Updating Methods _____________________________________________________________________________________ #
    def adding_active_aclus_information(self, spikes_df: pd.DataFrame, epoch_id_key_name: str = 'Probe_Epoch_id', add_unique_aclus_list_column: bool=False) -> pd.DataFrame:
        """ 
        adds the columns: ['unique_active_aclus', 'n_unique_aclus'] 

        Usage:

            active_epochs_df = active_epochs_df.epochs.adding_active_aclus_information(spikes_df=active_spikes_df, epoch_id_key_name='Probe_Epoch_id', add_unique_aclus_list_column=True)

        """
        active_epochs_df: pd.DataFrame = self._obj.epochs.get_valid_df()
        
        # Ensures the appropriate columns are added to spikes_df:
        # spikes_df = spikes_df.spikes.adding_epochs_identity_column(epochs_df=active_epochs_df, epoch_id_key_name=epoch_id_key_name, epoch_label_column_name='label', override_time_variable_name='t_rel_seconds',
        #     should_replace_existing_column=False, drop_non_epoch_spikes=True)
        assert epoch_id_key_name in spikes_df, f"""ERROR: epoch_id_key_name: '{epoch_id_key_name}' is not in spikes_df.columns: {spikes_df.columns}. Did you add the columns to spikes_df via:
## INPUTS: curr_active_pipeline, epochs_df
active_spikes_df: pd.DataFrame = get_proper_global_spikes_df(curr_active_pipeline).spikes.adding_epochs_identity_column(epochs_df=epochs_df, epoch_id_key_name='replay_id', epoch_label_column_name='label', override_time_variable_name='t_rel_seconds',
    should_replace_existing_column=False, drop_non_epoch_spikes=True) ## gets the proper spikes_df, and then adds the epoch_id columns (named 'replay_id') for the epochs provided in `epochs_df`
epochs_df = epochs_df.epochs.adding_active_aclus_information(spikes_df=active_spikes_df, epoch_id_key_name='replay_id', add_unique_aclus_list_column=True)
epochs_df
        """
        unique_values = np.unique(spikes_df[epoch_id_key_name]) # array([ 0,  1,  2,  3,  4,  7, 11, 12, 13, 14])
        grouped_df = spikes_df.groupby([epoch_id_key_name]) #  Groups on the specified column.
        epoch_unique_aclus_dict = {aValue:grouped_df.get_group(aValue).aclu.unique() for aValue in unique_values} # dataframes split for each unique value in the column

        # Convert label column in `active_epochs_df` to same dtype as the unique_values that were found
        active_epochs_df.label = active_epochs_df.label.astype(unique_values.dtype) ## WARNING: without this line it returns all np.nan results in the created columns!
        if add_unique_aclus_list_column:
            active_epochs_df['unique_active_aclus'] = active_epochs_df.label.map(epoch_unique_aclus_dict)
        epoch_num_unique_aclus_dict = {k:len(v) for k,v in epoch_unique_aclus_dict.items()}
        active_epochs_df['n_unique_aclus'] = active_epochs_df.label.map(epoch_num_unique_aclus_dict)
        active_epochs_df['n_unique_aclus'] = active_epochs_df['n_unique_aclus'].fillna(0).astype(int) # fill NaN values with 0s and convert to int
        return active_epochs_df


    
    @classmethod
    def add_maze_id_if_needed(cls, epochs_df: pd.DataFrame, t_start:Optional[float]=None, t_delta:Optional[float]=None, t_end:Optional[float]=None, active_maze_epochs_df: Optional[pd.DataFrame]=None, replace_existing:bool=True, labels_column_name:str='label', start_time_col_name: str='start', end_time_col_name: str='stop', no_interval_fill_value: Union[str, int] = '') -> pd.DataFrame:
        """ 2024-01-17 - adds the 'maze_id' column if it doesn't exist

        Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on
        
        WARNING: does NOT modify in place!

        Adds Columns: ['maze_id']
        Usage:
            from neuropy.core.session.dataSession import Laps

            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            laps_obj: Laps = curr_active_pipeline.sess.laps
            laps_df = laps_obj.to_dataframe()
            laps_df = laps_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            laps_df

        """
        from neuropy.utils.mixins.time_slicing import add_fully_overlapping_epochs_id_identity_to_epochs
        
        # epochs_df = epochs_df.epochs.to_dataframe()
        epochs_df[[labels_column_name]] = epochs_df[[labels_column_name]].astype('int')
        is_missing_column: bool = ('maze_id' not in epochs_df.columns)
        if (is_missing_column or replace_existing):
            # Create the maze_id column:
            if active_maze_epochs_df is not None:
                epochs_df['maze_id'] = '' # all empty string to start
                epochs_df = add_fully_overlapping_epochs_id_identity_to_epochs(query_child_epochs = epochs_df, potential_fully_enclosing_epochs_df = active_maze_epochs_df, epoch_id_key_name = 'maze_id', epoch_label_column_name=labels_column_name, start_time_col_name=start_time_col_name, end_time_col_name=end_time_col_name, no_interval_fill_value=no_interval_fill_value)
                ## These will be string values
            else:
                epochs_df['maze_id'] = np.full_like(epochs_df[labels_column_name].to_numpy(), -1) # all -1 to start
                epochs_df.loc[(np.logical_and((epochs_df[start_time_col_name].to_numpy() >= t_start), (epochs_df[end_time_col_name].to_numpy() <= t_delta))), 'maze_id'] = 0 # first epoch
                epochs_df.loc[(np.logical_and((epochs_df[start_time_col_name].to_numpy() >= t_delta), (epochs_df[end_time_col_name].to_numpy() <= t_end))), 'maze_id'] = 1 # second epoch, post delta
                epochs_df['maze_id'] = epochs_df['maze_id'].astype('int') # note the single vs. double brakets in the two cases. Not sure if it makes a difference or not
                
        else:
            # already exists and we shouldn't overwrite it:
            if (active_maze_epochs_df is None) and (t_delta is not None):
                ## only do this in the kdiba mode
                epochs_df[['maze_id']] = epochs_df[['maze_id']].astype('int') # note the single vs. double brakets in the two cases. Not sure if it makes a difference or not

        return epochs_df
            

    def adding_maze_id_if_needed(self, t_start:Optional[float]=None, t_delta:Optional[float]=None, t_end:Optional[float]=None, active_maze_epochs_df: Optional[pd.DataFrame]=None, replace_existing:bool=True, labels_column_name:str='label') -> pd.DataFrame:
        """ 2024-01-17 - adds the 'maze_id' column if it doesn't exist

        Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on
        
        WARNING: does NOT modify in place!

        Adds Columns: ['maze_id']
        Usage:
            from neuropy.core.session.dataSession import Laps

            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            laps_obj: Laps = curr_active_pipeline.sess.laps
            laps_df = laps_obj.to_dataframe()
            laps_df = laps_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            laps_df

        """
        epochs_df: pd.DataFrame = self._obj.epochs.get_valid_df()
        return self.add_maze_id_if_needed(epochs_df=epochs_df, t_start=t_start, t_delta=t_delta, t_end=t_end, active_maze_epochs_df=active_maze_epochs_df, replace_existing=replace_existing, labels_column_name=labels_column_name, start_time_col_name='start', end_time_col_name='stop')
    

    def adding_global_epoch_row(self, global_epoch_name='maze_GLOBAL', first_included_epoch_name=None, last_included_epoch_name=None, included_epoch_names=None, inplace: bool=False) -> pd.DataFrame:
        """ builds the 'global' epoch row for the entire session that includes by default the times from all other epochs in epochs_df. 
        e.g. builds the 'maze' epoch from ['maze1', 'maze2'] epochs
        
        Based off of `neuropy.core.session.Formats.BaseDataSessionFormats.DataSessionFormatBaseRegisteredClass.build_global_epoch_filter_config_dict` on 2025-01-15 15:56 
        
        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

            maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            maze_epochs_df = maze_epochs_df.epochs.adding_global_epoch_row()
            maze_epochs_df

        NOTE: Undoing this can be done with
            modified_df = curr_active_pipeline.sess.paradigm.adding_global_epoch_row().to_dataframe()
            modified_df = modified_df[np.logical_not(np.isin(modified_df['label'], ('maze', 'maze_GLOBAL')))]
            curr_active_pipeline.sess.paradigm._df = modified_df
        """
        all_epoch_names = list(self.get_unique_labels()) # all_epoch_names # ['maze1', 'maze2']
        if global_epoch_name in all_epoch_names:
            global_epoch_name = f"{global_epoch_name}_GLOBAL"
            print(f'WARNING: name collision "{global_epoch_name}" already exists in all_epoch_names: {all_epoch_names}! Using {global_epoch_name} instead.')
            if (global_epoch_name in all_epoch_names):
                print(F'\t\tDOUBLE-WARNING: already had the _GLOBAL suffixed one too! Skipping and returning unaltered!')
                return self._obj
                

        if (included_epoch_names is not None) and (len(included_epoch_names) >= 2):
            ## valid
            raise NotImplementedError(f'this mode is not yet implemented!')
            
        else:        
            if first_included_epoch_name is not None:
                # global_start_end_times[0] = sess.epochs[first_included_epoch_name][0] # 'maze1'
                pass
            else:
                first_included_epoch_name = self.get_unique_labels()[0]
                

            if last_included_epoch_name is not None:
                # global_start_end_times[1] = sess.epochs[last_included_epoch_name][1] # 'maze2'
                pass
            else:
                last_included_epoch_name = self.get_unique_labels()[-1]
        
        maze_epochs_df =  pd.DataFrame({'start': [*self.starts, self._obj.iloc[(self.labels == first_included_epoch_name)]['start'].tolist()[0]], 
                                        'stop': [*self.stops, self._obj.iloc[(self.labels == last_included_epoch_name)]['stop'].tolist()[0]],
        # 'stop': [*self.stops, self._obj[self.labels == last_included_epoch_name][1]],
        'label': [*self.labels, global_epoch_name],
        }) # .epochs.get_valid_df()    
        maze_epochs_df[['start', 'stop']] = maze_epochs_df[['start', 'stop']].astype(float) 
        maze_epochs_df['duration'] = maze_epochs_df['stop'] - maze_epochs_df['start']

        if inplace:
            self._obj = maze_epochs_df
            
        return maze_epochs_df # self._obj


    # @function_attributes(short_name=None, tags=['epochs', 'combine' 'global'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-08 16:19', related_items=[])
    def adding_concatenated_epoch(self, epochs_to_create_global_from_names = ['roam', 'sprinkle'], created_epoch_name: str = 'maze') -> pd.DataFrame:
        """ Creates an additional epoch with the desired merged epoch name in the epochs_df

        Usage:        
            from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch, EpochsAccessor

            epochs_df = ensure_dataframe(curr_active_pipeline.sess.epochs)
            epochs_df = epochs_df.epochs.adding_concatenated_epoch(epochs_to_create_global_from_names=['roam', 'sprinkle'], created_epoch_name='maze')


        """
        assert created_epoch_name not in self._obj['label'], f"self._obj['label']: {self._obj['label']} already contains the desired created epoch name: '{created_epoch_name}'"
        assert np.all(np.isin(epochs_to_create_global_from_names, self._obj['label'])), f"missing epochs: epochs_to_create_global_from_names: {epochs_to_create_global_from_names},\n\tACtual epoch names:_{self._obj['label'].to_list()}"

        filtered_epochs_df = self._obj[np.isin(self._obj['label'], epochs_to_create_global_from_names)] ## grab only the epochs of interest
        ## Create the concatenated epoch from the desired epochs:
        concatenated_epoch = {'start': filtered_epochs_df['start'].min(), 'stop': filtered_epochs_df['stop'].max(), 'label': created_epoch_name, 'duration': (filtered_epochs_df['stop'].max() - filtered_epochs_df['start'].min())}
        self._obj.loc[-1] = concatenated_epoch  # adding a row
        self._obj = self._obj.reset_index(drop=True, inplace=False)
        
        return self._obj



    def getting_metadata(self) -> Dict:
        """ gets the dataframe's `df.attrs` dictionary metadata
        """
        ## Add the metadata:
        if self._obj.attrs is None:
            return {}
        else:
            return self._obj.attrs


    def adding_or_updating_metadata(self, **metadata_update_kwargs) -> pd.DataFrame:
        """ updates the dataframe's `df.attrs` dictionary metadata, building it as a new dict if it doesn't yet exist
         
        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

            maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            maze_epochs_df = maze_epochs_df.epochs.adding_or_updating_metadata(train_test_period='train')
            maze_epochs_df

        """
        ## Add the metadata:
        if self._obj.attrs is None:
            self._obj.attrs = {} # create a new metadata dict on the dataframe
        self._obj.attrs.update(**metadata_update_kwargs)
        return self._obj



    # ==================================================================================================================== #
    # `Epoch` object / pd.DataFrame exchangeability                                                                         #
    # ==================================================================================================================== #
    def to_dataframe(self) -> pd.DataFrame:
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        return self._obj.copy()

    def to_Epoch(self) -> "Epoch":
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        return Epoch(self._obj.copy())


class Epoch(HDFMixin, StartStopTimesMixin, TimeSlicableObjectProtocol, DataFrameRepresentable, DataFrameInitializable, DataWriter):
    """ An Epoch object holds one ore more periods of time (marked by start/end timestamps) along with their corresponding metadata.
    from neuropy.core.epoch import Epoch
    
    """
    def __init__(self, epochs: pd.DataFrame, metadata=None) -> None:
        """[summary]
        Args:
            epochs (pd.DataFrame): Each column is a pd.Series(["start", "stop", "label"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        if not isinstance(epochs, pd.DataFrame):
            _epochs_metadata = getattr(epochs, 'metadata', None)
            metadata = metadata or _epochs_metadata
            epochs = epochs.to_dataframe() # try to convert to dataframe if the object is an Epoch or other compatible object
            assert isinstance(epochs, pd.DataFrame)
        super().__init__(metadata=metadata)
        self._df = epochs.epochs.get_valid_df() # gets already sorted appropriately and everything. epochs.epochs uses the DataFrame accesor
        self._check_epochs(self._df) # check anyway
        # self._check_epochs(epochs) # check anyway


    @classmethod
    def init_from_start_stops_df(cls, starts_stops_df: pd.DataFrame, **kwargs):
        if 'label' not in starts_stops_df:
            starts_stops_df['label'] = deepcopy(starts_stops_df.index.astype('str'))
        if 'duration' not in starts_stops_df:
            starts_stops_df['duration'] = starts_stops_df['stop'] - starts_stops_df['start']
        return cls(epochs=starts_stops_df.epochs.get_valid_df(), **kwargs)
        
        

    @property
    def starts(self):
        return self._df.epochs.starts

    @property
    def stops(self):
        return self._df.epochs.stops
    
    @property
    def t_start(self):
        return self.starts[0]
    @t_start.setter
    def t_start(self, t):
        self._df.epochs.t_start = t

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def t_stop(self):
        return self.stops[-1]

    @property
    def durations(self):
        return self.stops - self.starts

    @property
    def midtimes(self): # NDArray
        """ since each epoch is given by a (start, stop) time, the midtimes are the center of this epoch. """
        return self._df.epochs.midtimes


    @property
    def n_epochs(self):
        return self._df.epochs.n_epochs
    @property
    def labels(self):
        return self._df.epochs.labels

    def get_unique_labels(self):
        # return np.unique(self.labels)
        return self._df.label.unique()
    
    def get_named_timerange(self, epoch_name):
        return NamedTimerange(name=epoch_name, start_end_times=self[epoch_name])


    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """metadata compatibility"""
        self._metadata = metadata

        
    @property
    def epochs(self) -> "EpochsAccessor":
        """ a passthrough accessor to the Pandas dataframe `EpochsAccessor` to allow complete pass-thru compatibility with either Epoch or pd.DataFrame versions of epochs.
        Instead of testing whether it's an `Epoch` object or pd.DataFrame and then converting back and forth, should just be able to pretend it's a dataframe for the most part and use the `some_epochs.epochs.*` properties and methods.
        """
        return self._df.epochs


    def _check_epochs(self, epochs):
        assert isinstance(epochs, pd.DataFrame)
        # epochs.epochs.
        assert (
            pd.Series(["start", "stop", "label"]).isin(epochs.columns).all()
        ), f"Epoch dataframe should at least have columns with names: start, stop, label but it only has the columns: {list(epochs.columns)}"

    def __repr__(self) -> str:
        # return f"{len(self.starts)} epochs"
        # return f"{len(self.starts)} epochs\n{self.as_array().__repr__()}\n"
        # """
        # 1 epochs
        # array([[7125, 11745]])
        # """
        # return self.to_dataframe()[['label', 'start', 'stop']].values
        # return self.to_dataframe()[['label', 'start', 'stop']].itertuples(index=False)
        # column_names: List[str] = ['label', 'start', 'stop']
        # records = [tuple(v) for v in self.to_dataframe()[column_names].itertuples(index=False)]
        # return f"pd.DataFrame.from_records({records!r}, columns={column_names!r})"  ## return whole initialization code
        # """
        # array([['roam', 7125.0, 9590.999999],
        # ['sprinkle', 9591.0, 11745.0]], dtype=object)
        # """
        return f"{type(self).__name__}({self.str_for_concise_display()})"



    def _repr_pretty_(self, p, cycle=False):
        """ The cycle parameter will be true if the representation recurses - e.g. if you put a container inside itself. """
        # p.text(self.__repr__() if not cycle else '...')
        p.text(self.to_dataframe().__repr__() if not cycle else '...')

    def __str__(self) -> str:
        return f"{len(self.starts)} epochs\n{self.as_array().__repr__()}\n"
    
    def __len__(self):
        """ allows using `len(epochs_obj)` and getting the number of epochs. """
        return len(self.starts)

    def str_for_concise_display(self) -> str:
        """ returns a minimally descriptive string like: '60 epochs in (17.9, 524.1)' that doesn't print all the array elements only the number of epochs and the first and last. """
        return f"{len(self.starts)} epochs in ({self.starts[0]:.1f}, {self.stops[-1]:.1f})" # "60 epochs in (17.9, 524.1)"

    def str_for_filename(self) -> str:
        return f"Epoch[{len(self.starts)}]({self.starts[0]:.1f}-{self.stops[-1]:.1f})" #


    @property
    def __array_interface__(self):
        """ wraps the internal dataframe's `__array_interface__` which Pandas uses to provide numpy with information about dataframes such as np.shape(a_df) info.
        Allows np.shape(an_epoch_obj) to work.

        """
        # Get the numpy array's __array_interface__ from the DataFrame's values
        # The .to_numpy() method explicitly converts the DataFrame to a NumPy array
        return self._df.to_numpy().__array_interface__
        # return self._df.__array_interface__


    def __getitem__(self, slice_):
        """ Allows pass-thru indexing like it were a numpy array.

        2024-03-07 Potentially more dangerous than helpful.

        having issue whith this being called with pd.Dataframe columns (when assuming a pd.DataFrame epochs format but actually an Epoch object)

        IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
               Occurs because `_slice == ['lap_id']` which doesn't pass the first check because it's a list of strings not a string itself
        Example:
            Error line `laps_df[['lap_id']] = laps_df[['lap_id']].astype('int')`
        """
        if isinstance(slice_, str):
            indices = np.where(self.labels == slice_)[0]
            if len(indices) > 1:
                return np.vstack((self.starts[indices], self.stops[indices])).T
            else:
                return np.array([self.starts[indices], self.stops[indices]]).squeeze()
        elif ((slice_ is not None) and (len(slice_) > 0) and isinstance(slice_[0], str)): # TypeError: object of type 'int' has no len()
            # a list of strings, probably meant to use a dataframe indexing method
            # having issue whith this being called with pd.Dataframe columns (when assuming a pd.DataFrame epochs format but actually an Epoch object)
            # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
            #     Occurs because `_slice == ['lap_id']` which doesn't pass the first check because it's a list of strings not a string itself
            # Example:
            #     Error line `laps_df[['lap_id']] = laps_df[['lap_id']].astype('int')`                
            raise IndexError(f"PHO: you're probably trying to treat the epochs as if they are in the pd.DataFrame format but they are an Epoch object! Use `actual_laps_df = incorrectly_assumed_laps_df.epochs.to_dataframe()` to convert.")
            
        else:
            return np.vstack((self.starts[slice_], self.stops[slice_])).T


    def adding_global_epoch_row(self, global_epoch_name='maze_GLOBAL', first_included_epoch_name=None, last_included_epoch_name=None, included_epoch_names=None, inplace: bool=False) -> "Epoch":
        """ builds the 'global' epoch row for the entire session that includes by default the times from all other epochs in epochs_df. 
        e.g. builds the 'maze' epoch from ['maze1', 'maze2'] epochs
        
        Based off of `neuropy.core.session.Formats.BaseDataSessionFormats.DataSessionFormatBaseRegisteredClass.build_global_epoch_filter_config_dict` on 2025-01-15 15:56 
        
        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

            maze_epochs_obj = ensure_Epoch(deepcopy(curr_active_pipeline.sess.epochs).to_dataframe())
            maze_epochs_obj.adding_global_epoch_row(global_epoch_name='maze', first_included_epoch_name=None, last_included_epoch_name=None)
            maze_epochs_obj
            
        """ 
        updated_df = self._df.epochs.adding_global_epoch_row(global_epoch_name=global_epoch_name, first_included_epoch_name=first_included_epoch_name, last_included_epoch_name=last_included_epoch_name, included_epoch_names=included_epoch_names, inplace=inplace)
        if inplace:        
            self._df = updated_df
            return self
        else:
            ## make a copy
            _epoch_copy: "Epoch" = deepcopy(self)
            _epoch_copy._df = deepcopy(updated_df)
            # return Epoch(deepcopy(maze_epochs_df), metadata=self.metadata)
            return _epoch_copy
        

        
    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        return Epoch(epochs=self._df.epochs.time_slice(t_start, t_stop), metadata=self.metadata)
        
    def label_slice(self, label):
        return Epoch(epochs=self._df.epochs.label_slice(label), metadata=self.metadata)

    def boolean_indicies_slice(self, boolean_indicies):
        return Epoch(epochs=self._df[boolean_indicies], metadata=self.metadata)

    def filtered_by_duration(self, min_duration=None, max_duration=None):
        return Epoch(epochs=self._df.epochs.filtered_by_duration(min_duration, max_duration), metadata=self.metadata)

    @classmethod
    def filter_epochs(cls, curr_epochs: Union[pd.DataFrame, "Epoch"], pos_df:Optional[pd.DataFrame]=None, spikes_df:pd.DataFrame=None, require_intersecting_epoch:"Epoch"=None, min_epoch_included_duration=0.06, max_epoch_included_duration=0.6,
        maximum_speed_thresh=2.0, min_inclusion_fr_active_thresh=2.0, min_num_unique_aclu_inclusions=3, debug_print=False) -> "Epoch":
        """filters the provided replay epochs by specified constraints.

        Args:
            curr_epochs (Epoch): the epochs to filter on
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            max_epoch_included_duration (float, optional): all epochs longer than max_epoch_included_duration will be excluded from analysis. Defaults to 0.6.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            min_inclusion_fr_active_thresh: minimum firing rate (in Hz) for a unit to be considered "active" for inclusion.
            min_num_unique_aclu_inclusions: minimum number of unique active cells that must be included in an epoch to have it included.

            save_on_compute (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            Epoch: the filtered epochs as an Epoch object

        NOTE: this really is a general method that works for any Epoch object or Epoch-type dataframe to filter it.

        TODO 2023-04-11 - This really belongs in the Epoch class or the epoch dataframe accessor. 

        """
        from neuropy.utils.efficient_interval_search import filter_epochs_by_speed
        from neuropy.utils.efficient_interval_search import filter_epochs_by_num_active_units

        if not isinstance(curr_epochs, pd.DataFrame):
            curr_epochs = curr_epochs.to_dataframe() # .get_valid_df() # convert to pd.DataFrame to start
    
        assert isinstance(curr_epochs, pd.DataFrame), f'curr_replays must be a pd.DataFrame or Epoch object, but is {type(curr_epochs)}'
        # Ensure the dataframe representation has the required columns. TODO: is this needed?
        if not 'stop' in curr_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            curr_epochs['stop'] = curr_epochs['end'].copy()
        if not 'label' in curr_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            curr_epochs['label'] = curr_epochs['flat_replay_idx'].copy()
        # must convert back from pd.DataFrame to Epoch object to use the Epoch methods
        curr_epochs = cls(curr_epochs)

        ## Use the existing replay epochs from the session but ensure they look valid:

        ## Filter based on required overlap with Ripples:
        if require_intersecting_epoch is not None:
            curr_epochs = cls.from_PortionInterval(require_intersecting_epoch.to_PortionInterval().intersection(curr_epochs.to_PortionInterval()))
        else:
            curr_epochs = cls.from_PortionInterval(curr_epochs.to_PortionInterval()) # just do this to ensure non-overlapping

        if curr_epochs.n_epochs == 0:
            warn(f'curr_epochs already empty prior to any filtering')

        # Filter by duration bounds:
        if (min_epoch_included_duration is not None) or (max_epoch_included_duration is not None):
            curr_epochs = curr_epochs.filtered_by_duration(min_duration=min_epoch_included_duration, max_duration=max_epoch_included_duration)

        # Filter *_replays_Interval by requiring them to be below the speed:
        if maximum_speed_thresh is not None:
            assert pos_df is not None, "must provide pos_df if filtering by speed"
            if curr_epochs.n_epochs > 0:
                curr_epochs, above_speed_threshold_intervals, below_speed_threshold_intervals = filter_epochs_by_speed(pos_df, curr_epochs, speed_thresh=maximum_speed_thresh, debug_print=debug_print)
            else:
                warn(f'curr_epochs already empty prior to filtering by speed')

        # 2023-02-10 - Trimming and Filtering Estimated Replay Epochs based on cell activity and pyramidal cell start/end times:
        if (min_inclusion_fr_active_thresh is not None) or (min_num_unique_aclu_inclusions is not None):
            assert spikes_df is not None, "must provide spikes_df if filtering by active units"
            active_spikes_df = spikes_df.spikes.sliced_by_neuron_type('pyr') # trim based on pyramidal cell activity only
            if curr_epochs.n_epochs > 0:
                curr_epochs, _extra_outputs = filter_epochs_by_num_active_units(active_spikes_df, curr_epochs, min_inclusion_fr_active_thresh=min_inclusion_fr_active_thresh, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, include_intermediate_computations=False) # TODO: seems wasteful considering we compute all these spikes_df metrics and refinements and then don't return them.
            else:
                warn(f'curr_epochs already empty prior to filtering by firing rate or minimum active units')
                
        return curr_epochs

    def to_dict(self, recurrsively=False):
        d = {"epochs": self._df, "metadata": self._metadata}
        return d
    
    @staticmethod
    def from_dict(d: dict):
        return Epoch(d["epochs"], metadata=d["metadata"])

    ## TODO: refactor these methods into the 'epochs' pd.DataFrame accessor above and then wrap them:
    def fill_blank(self, method="from_left"):
        ep_starts = self.epochs["start"].values
        ep_stops = self.epochs["stop"].values
        ep_durations = self.epochs["duration"].values
        ep_labels = self.epochs["label"].values

        mask = (ep_starts[:-1] + ep_durations[:-1]) < ep_starts[1:]
        (inds,) = np.nonzero(mask)

        if method == "from_left":
            for ind in inds:
                ep_durations[ind] = ep_starts[ind + 1] - ep_starts[ind]

        elif method == "from_right":
            for ind in inds:
                gap = ep_starts[ind + 1] - (ep_starts[ind] + ep_durations[ind])
                ep_starts[ind + 1] -= gap
                ep_durations[ind + 1] += gap

        elif method == "from_nearest":
            for ind in inds:
                gap = ep_starts[ind + 1] - (ep_starts[ind] + ep_durations[ind])
                ep_durations[ind] += gap / 2.0
                ep_starts[ind + 1] -= gap / 2.0
                ep_durations[ind + 1] += gap / 2.0

        self.epochs["start"] = ep_starts
        self.epochs["stop"] = ep_starts + ep_durations
        self.epochs["duration"] = ep_durations

    def delete_in_between(self, t1, t2):
        epochs_df = self.to_dataframe()[["start", "stop", "label"]]
        # delete epochs if they are within t1, t2
        epochs_df = epochs_df[~((epochs_df["start"] >= t1) & (epochs_df["stop"] <= t2))]

        # truncate stop if start is less than t1 but stop is within t1,t2
        epochs_df.loc[
            (epochs_df["start"] < t1)
            & (t1 < epochs_df["stop"])
            & (epochs_df["stop"] <= t2),
            "stop",
        ] = t1

        # truncate start if stop is greater than t2 but start is within t1,t2
        epochs_df.loc[
            (epochs_df["start"] > t1)
            & (epochs_df["start"] <= t2)
            & (epochs_df["stop"] > t2),
            "start",
        ] = t2

        # if epoch starts before and ends after range,
        flank_start = epochs_df[
            (epochs_df["start"] < t1) & (epochs_df["stop"] > t2)
        ].copy()
        flank_start["stop"] = t1
        flank_stop = epochs_df[
            (epochs_df["start"] < t1) & (epochs_df["stop"] > t2)
        ].copy()
        flank_stop["start"] = t2
        epochs_df = epochs_df[~((epochs_df["start"] < t1) & (epochs_df["stop"] > t2))]
        epochs_df = epochs_df.append(flank_start)
        epochs_df = epochs_df.append(flank_stop)
        epochs_df = epochs_df.reset_index(drop=True)

        return Epoch(epochs_df)

    def get_proportion_by_label(self, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.starts[0]
        if t_stop is None:
            t_stop = self.stops[-1]

        duration = t_stop - t_start

        ep = self._df.copy()
        ep = ep[(ep.stop > t_start) & (ep.start < t_stop)].reset_index(drop=True)

        if ep["start"].iloc[0] < t_start:
            ep.at[0, "start"] = t_start

        if ep["stop"].iloc[-1] > t_stop:
            ep.at[ep.index[-1], "stop"] = t_stop

        ep["duration"] = ep.stop - ep.start

        ep_group = ep.groupby("label").sum().duration / duration

        label_proportion = {}
        for label in self.get_unique_labels():
            label_proportion[label] = 0.0

        for state in ep_group.index.values:
            label_proportion[state] = ep_group[state]

        return label_proportion

    def count(self, t_start=None, t_stop=None, binsize=300):
        if t_start is None:
            t_start = 0

        if t_stop is None:
            t_stop = np.max(self.stops)

        mid_times = self.starts + self.durations / 2
        bins = np.arange(t_start, t_stop + binsize, binsize)
        return np.histogram(mid_times, bins=bins)[0]

    def to_neuroscope(self, ext="PHO", override_filepath=None):
        """ exports to a Neuroscope compatable .evt file. """
        if not isinstance(ext, str):
            ext = '.'.join(ext) # assume it is an list, tuple or something. Join its elements by periods

        if override_filepath is not None:
            out_filepath = override_filepath.resolve()
            out_filepath = out_filepath.with_suffix(f".{ext}.evt")
        else:
            assert self.filename is not None
            out_filepath = self.filename.with_suffix(f".{ext}.evt")

        with out_filepath.open("w") as a:
            for event in self._df.itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} end\n")
        return out_filepath

    @classmethod
    def from_neuroscope(cls, in_filepath, metadata=None):
        """ imports from a Neuroscope compatible .evt file.
        Usage:
            from neuropy.core.epoch import Epoch

            evt_filepath = Path('/Users/pho/Downloads/2006-6-07_16-40-19.bst.evt').resolve()
            # evt_filepath = Path('/Users/pho/Downloads/2006-6-08_14-26-15.bst.evt').resolve()
            evt_epochs: Epoch = Epoch.from_neuroscope(in_filepath=evt_filepath)
            evt_epochs

        """
        if isinstance(in_filepath, str):
            in_filepath = Path(in_filepath).resolve()
        assert in_filepath.exists()

        # Read the .evt file and reconstruct the data
        data = []
        with in_filepath.open("r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamp: float = float(parts[0]) / 1000.0
                    event_type = parts[1]
                    if event_type in ['start', 'st']:
                        start = timestamp
                    elif event_type in ['end', 'e']:
                        end = timestamp
                        data.append({'start': start, 'stop': end})

        # Convert the reconstructed data into a DataFrame
        df = pd.DataFrame(data)
        df['label'] = df.index.astype('str', copy=True)
        _obj = cls.from_dataframe(df=df)
        _obj.filename = in_filepath.stem
        _obj.metadata = metadata
        return _obj
        
    def as_array(self):
        return self.to_dataframe()[["start", "stop"]].to_numpy()

    # Requires Optional `portion` library
    @classmethod
    def from_PortionInterval(cls, portion_interval, metadata=None):
        return Epoch(epochs=EpochsAccessor.from_PortionInterval(portion_interval), metadata=metadata) 

    def to_PortionInterval(self):
        return self._df.epochs.to_PortionInterval()

    def get_non_overlapping(self, debug_print=False):
        """ Returns a copy with overlapping epochs removed. """
        return Epoch(epochs=self._df.epochs.get_non_overlapping_df(debug_print=debug_print), metadata=self.metadata)
    
    def get_in_between(self, copy_metadata:bool=False) -> "Epoch":
        """ gets the epochs that are in-between (non-overlapping) the current epochs."""
        return Epoch(epochs=self._df.epochs.get_in_between(copy_metadata=copy_metadata), metadata=self.metadata)

    def modify_each_epoch_by(self, additive_factor: float = 0.0, multiplicative_factor: float=1.0, final_output_minimum_epoch_duration:float=0.0, copy_metadata:bool=False) -> "Epoch":
        """ gets a copy of the epochs where each epoch is modified by the provided scale factors.
        if safe_contract_epochs is not 0.0, a value of 0.0 is used to safely
        """
        return Epoch(epochs=self._df.epochs.modify_each_epoch_by(additive_factor=additive_factor, multiplicative_factor=multiplicative_factor, final_output_minimum_epoch_duration=final_output_minimum_epoch_duration, copy_metadata=copy_metadata), metadata=self.metadata)
        
    def subtracting(self, other_epochs_df: pd.DataFrame, skip_get_non_overlapping:bool=False) -> "Epoch":
        """ gets a copy of the epochs after subtracting the epochs provided in `other_epochs_df`.            
        """
        return Epoch(epochs=self._df.epochs.subtracting(other_epochs_df=other_epochs_df, skip_get_overlapping=skip_get_non_overlapping), metadata=self.metadata)
        

    def split_into_training_and_test(self, training_data_portion: float=5.0/6.0, group_column_name: str='label', additional_epoch_identity_column_names:List[str]=['label'], skip_get_non_overlapping:bool=False, debug_print: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Splits laps into separate training and test sections
        
        Usage:

            ### Get the laps to train on
            training_data_portion: float = 5.0/6.0
            test_data_portion: float = 1.0 - training_data_portion # test data portion is 1/6 of the total duration

            print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')

            a_laps_df: pd.DataFrame = ensure_dataframe(deepcopy(a_config['pf_params'].computation_epochs))
            a_laps_training_df, a_laps_test_df = a_laps_df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name ='lap_id', additional_epoch_identity_column_names=['label', 'lap_id', 'lap_dir'], skip_get_non_overlapping=False, debug_print=False) # a_laps_training_df, a_laps_test_df both comeback good here.

            laps_df
            laps_training_df
            laps_test_df 
            
        Usage 2:
            training_data_portion: float = 5.0/6.0
            a_new_training_df, a_new_test_df = global_epoch_only_non_PBE_epoch_df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name ='label', additional_epoch_identity_column_names=['label'], skip_get_non_overlapping=False, debug_print=False) # a_laps_training_df, a_laps_test_df both comeback good here.

            a_new_training_df
            a_new_test_df

        """
        return Epoch(epochs=self._df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name=group_column_name, additional_epoch_identity_column_names=additional_epoch_identity_column_names, skip_get_overlapping=skip_get_non_overlapping, debug_print=debug_print), metadata=self.metadata)


    # HDF5 Serialization _________________________________________________________________________________________________ #
    # HDFMixin Conformances ______________________________________________________________________________________________ #

    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pos_obj: Position = long_one_step_decoder_1D.pf.position
            _pos_obj.to_hdf(hdf5_output_path, key='pos')
        """
        _df = self.to_dataframe()
        _df.to_hdf(path_or_buf=file_path, key=key, format=kwargs.pop('format', 'table'), data_columns=kwargs.pop('data_columns',True), **kwargs)
        return
    
        # # create_group
        # a_key = Path(key)
        # with tb.open_file(file_path, mode='r+') as f:
        #     # group = f.create_group(str(a_key.parent), a_key.name, title='epochs.', createparents=True)
        #     group = f.get_node(str(a_key.parent))
        #     # group = f[key]
        #     table = f.create_table(group, a_key.name, EpochTable, "Epochs")
        #     # Serialization
        #     for i, t_start, t_stop, a_label in zip(np.arange(self.n_epochs), self.starts, self.stops, self.labels):
        #         row = table.row
        #         row['t_start'] = t_start
        #         row['t_end'] = t_stop  # Provide an appropriate session identifier here
        #         row['label'] = str(a_label)
        #         row.append()
                
        #     table.flush()
        #     # Metadata:
        #     group.attrs['t_start'] = self.t_start
        #     group.attrs['t_stop'] = self.t_stop
        #     group.attrs['n_epochs'] = self.n_epochs

    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "Epoch":
        """  Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = Epoch.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        _df = pd.read_hdf(file_path, key=key, **kwargs)
        return cls(_df, metadata=None) # TODO: recover metadata


    # DataFrameInitializable Conformances ________________________________________________________________________________ #
    
    def to_dataframe(self) -> pd.DataFrame:
        df = self._df.copy()
        return df


    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, metadata=None) -> "Epoch":
        return cls(df, metadata=metadata)


    @classmethod
    def from_starts_stops_arrays(cls, starts: NDArray, stops: NDArray, labels: Optional[NDArray]=None, metadata=None) -> "Epoch":
        """ initalizes from equal length starts/stops arrays 
        
        Usage:
            
            from neuropy.core.epoch import Epoch

            starts = deepcopy(results1D.continuous_results['global'].time_bin_containers[0].left_edges)
            stops = deepcopy(results1D.continuous_results['global'].time_bin_containers[0].right_edges)
            an_epoch: Epoch = Epoch.from_starts_stops_arrays(starts=starts, stops=stops)
            an_epoch
        """
        assert len(starts) == len(stops)
        if labels is not None:
            assert len(labels) == len(starts)
        else:
            labels = np.arange(len(starts)) ## ascending indexes
        df = pd.DataFrame(dict(start=deepcopy(starts), stop=deepcopy(stops), label=labels))
        df[['start', 'stop']] = df[['start', 'stop']].astype(float)
        return cls.from_dataframe(df=df, metadata=metadata)


    # ==================================================================================================================== #
    # `Epoch` object / pd.DataFrame exchangeability                                                                         #
    # ==================================================================================================================== #
    # NOTE: `def to_dataframe(self) -> pd.DataFrame` is defined above

    def to_Epoch(self) -> "Epoch":
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        return Epoch(epochs=self._df.copy(), metadata=self.metadata)


def ensure_dataframe(epochs: Union[Epoch, pd.DataFrame]) -> pd.DataFrame:
    """ Ensures that the epochs are returned as an Pandas DataFrame, does nothing if they already are a DataFrame.
    
    Reciprocal to `ensure_Epoch(...)`

    Usage:

        from neuropy.core.epoch import ensure_dataframe

    """
    if isinstance(epochs, pd.DataFrame):
        return epochs
    else:
        return epochs.to_dataframe()


def ensure_Epoch(epochs: Union[Epoch, pd.DataFrame], metadata=None) -> Epoch:
    """ Ensures that the epochs are returned as an Epoch object, does nothing if they already are an Epoch object.

        Reciprocal to `ensure_dataframe(...)`

        Usage:

        from neuropy.core.epoch import ensure_Epoch

    """
    if isinstance(epochs, pd.DataFrame):
        ## convert to Epoch
        return Epoch.from_dataframe(epochs, metadata=metadata)
    else:
        ## assume already an Epoch
        if metadata is not None:
            if epochs.metadata is None:
                epochs.metadata = metadata
            else:
                epochs.metadata = epochs.metadata | metadata # merge metadata
        
        return epochs
    
