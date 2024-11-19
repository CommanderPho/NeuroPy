"""
This type stub file was generated by pyright.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, TypeVar, Union
from nptyping import NDArray
from contextlib import ContextDecorator

T = TypeVar('T')
def collapse_if_identical(iterable: Iterable[T], return_original_on_failure: bool = ...) -> Optional[Iterable[T]]:
    """
    Collapse an iterable to its first item if all items in the iterable are identical.
    If not all items are identical and 'return_original_on_failure' is True, the original
    iterable is returned as much collapsed as possible; otherwise, None is returned.

    Parameters
    ----------
    iterable : Iterable[T]
        An iterable containing items of any type (denoted by T).
    return_original_on_failure : bool, default=False
        If True, return the original iterable when it's not collapsible. If False, return None.

    Returns
    -------
    Optional[Iterable[T]]
        The first item of the iterable if all items are identical, or if 'return_original_on_failure'
        is set to True, the original iterable is returned when items are not identical. Otherwise, None
        is returned.

    Raises
    ------
    StopIteration
        If the provided iterable is empty, a StopIteration exception is raised internally
        and caught by the function to return None or the original iterable.
        
    Examples
    --------
    from neuropy.utils.indexing_helpers import collapse_if_identical
    
    >>> identical_items = ["a", "a", "a"]
    >>> collapse_if_identical(identical_items)
    "a"
    
    >>> non_identical_items = ["a", "b", "a"]
    >>> collapse_if_identical(non_identical_items)
    None
    
    >>> collapse_if_identical(non_identical_items, return_original_on_failure=True)
    ["a", "b", "a"]
    
    >>> empty_iterable = []
    >>> collapse_if_identical(empty_iterable)
    None
    """
    ...

def flatten(A): # -> list[Any]:
    """ safely flattens lists of lists without flattening top-level strings also. 
    https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python

    Usage:

        from neuropy.utils.indexing_helpers import flatten

    Example:
        list(flatten(['format_name', ('animal','exper_name', 'session_name')] ))
        >>> ['format_name', 'animal', 'exper_name', 'session_name']

    """
    ...

def unwrap_single_item(lst): # -> None:
    """ if the item contains at least one item, return it, otherwise return None.

    from neuropy.utils.indexing_helpers import unwrap_single_item


    """
    ...

def find_desired_sort_indicies(extant_arr, desired_sort_arr): # -> tuple[NDArray[Any], NDArray[Any] | Any]:
    """ Finds the set of sort indicies that can be applied to extant_arr s.t.
        (extant_arr[out_sort_idxs] == desired_sort_arr)
    
    INEFFICIENT: O^n^2
    
    Usage:
        from neuropy.utils.indexing_helpers import find_desired_sort_indicies
        new_all_aclus_sort_indicies, desired_sort_arr = find_desired_sort_indicies(active_2d_plot.neuron_ids, all_sorted_aclus)
        assert len(new_all_aclus_sort_indicies) == len(active_2d_plot.neuron_ids), f"need to have one new_all_aclus_sort_indicies value for each neuron_id"
        assert np.all(active_2d_plot.neuron_ids[new_all_aclus_sort_indicies] == all_sorted_aclus), f"must sort "
        new_all_aclus_sort_indicies
    """
    ...

def union_of_arrays(*arrays) -> np.array:
    """ 
    from neuropy.utils.indexing_helpers import union_of_arrays
    
    """
    ...

def intersection_of_arrays(*arrays) -> np.array:
    """ 
    from neuropy.utils.indexing_helpers import union_of_arrays
    
    """
    ...

class NumpyHelpers:
    """ various extensions and generalizations for numpy arrays 
    
    from neuropy.utils.indexing_helpers import NumpyHelpers


    """
    @classmethod
    def all_array_generic(cls, pairwise_numpy_fn, list_of_arrays: List[NDArray], **kwargs) -> bool:
        """ A n-element generalization of a specified pairwise numpy function such as `np.array_equiv`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_generic(list_of_arrays=list_of_arrays)

        """
        ...
    
    @classmethod
    def assert_all_array_generic(cls, pairwise_numpy_assert_fn, list_of_arrays: List[NDArray], **kwargs): # -> None:
        """ A n-element generalization of a specified pairwise np.testing.assert* function such as `np.testing.assert_array_equal` or `np.testing.assert_allclose`

        TODO: doesn't really work yet

        msg: a use-provided assert message
        

        
        Usage:

            list_of_arrays = list(xbins.values())
            NumpyHelpers.assert_all_array_generic(np.testing.assert_array_equal, list_of_arrays=list_of_arrays, msg=f'test message')
            NumpyHelpers.assert_all_array_generic(np.testing.assert_array_equal, list_of_arrays=list(neuron_ids.values()), msg=f'test message')

        """
        ...
    
    @classmethod
    def all_array_equal(cls, list_of_arrays: List[NDArray], equal_nan=...) -> bool:
        """ A n-element generalization of `np.array_equal`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_equal(list_of_arrays=list_of_arrays)

        """
        ...
    
    @classmethod
    def all_array_equiv(cls, list_of_arrays: List[NDArray]) -> bool:
        """ A n-element generalization of `np.array_equiv`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_equiv(list_of_arrays=list_of_arrays)

        """
        ...
    
    @classmethod
    def all_allclose(cls, list_of_arrays: List[NDArray], rtol: float = ..., atol: float = ..., equal_nan: bool = ...) -> bool:
        """ A n-element generalization of `np.allclose`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_allclose(list_of_arrays=list_of_arrays)

        """
        ...
    
    @classmethod
    def safe_concat(cls, np_concat_list: Union[List[NDArray], Dict[Any, NDArray]], **np_concat_kwargs) -> NDArray:
        """ returns an empty dataframe if the key isn't found in the group.
        Usage:
            from neuropy.utils.indexing_helpers import NumpyHelpers

            NumpyHelpers.safe_concat
            
        """
        ...
    
    @classmethod
    def logical_generic(cls, pairwise_numpy_logical_fn, *list_of_arrays, **kwargs) -> NDArray:
        """ generalizes the logical_and function to multiple arrays 

                from neuropy.utils.indexing_helpers import NumpyHelpers
        df1_matches_all_conditions = NumpyHelpers.logical_generic(np.logical_and, *[df1[k] == v for k, v in a_values_dict.items()])
        
        """
        ...
    


def paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists): # -> list[Any]:
    """ builds up a list of `sorted_lists` 

    Given a list of neuron_IDs (usually aclus) and equal-sized lists containing values which to sort the lists of neuron_IDs, returns the list of incrementally sorted neuron_IDs. e.g.:

    Inputs: neuron_IDs_lists, sortable_values_lists

    Usage:
        from neuropy.utils.indexing_helpers import paired_incremental_sorting

        neuron_IDs_lists = [a_decoder.neuron_IDs for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
        sortable_values_lists = [np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1) for a_decoder in decoders_dict.values()]
        sorted_neuron_IDs_lists = paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists)
        sort_helper_original_neuron_id_to_IDX_dicts = [dict(zip(neuron_ids, np.arange(len(neuron_ids)))) for neuron_ids in neuron_IDs_lists] # just maps each neuron_id in the list to a fragile_linear_IDX 

        # `sort_helper_neuron_id_to_sort_IDX_dicts` dictionaries in the appropriate order (sorted order) with appropriate indexes. Its .values() can be used to index into things originally indexed with aclus.
        sort_helper_neuron_id_to_sort_IDX_dicts = [{aclu:a_sort_helper_neuron_id_to_IDX_map[aclu] for aclu in sorted_neuron_ids} for a_sort_helper_neuron_id_to_IDX_map, sorted_neuron_ids in zip(sort_helper_original_neuron_id_to_IDX_dicts, sorted_neuron_IDs_lists)]
        # sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[a_sort_list, :] for a_decoder, a_sort_list in zip(decoders_dict.values(), sorted_neuron_IDs_lists)]

        sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]


    """
    ...

def paired_individual_sorting(neuron_IDs_lists, sortable_values_lists): # -> list[Any]:
    """ nothing "paired" about it, just individually sorts the items in `neuron_IDs_lists` by `sortable_values_lists`

    Given a list of neuron_IDs (usually aclus) and equal-sized lists containing values which to sort the lists of neuron_IDs, returns the list of incrementally sorted neuron_IDs. e.g.:

    Inputs: neuron_IDs_lists, sortable_values_lists

    Usage:
        from neuropy.utils.indexing_helpers import paired_individual_sorting

        neuron_IDs_lists = [a_decoder.neuron_IDs for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
        sortable_values_lists = [np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1) for a_decoder in decoders_dict.values()]
        sorted_neuron_IDs_lists = paired_individual_sorting(neuron_IDs_lists, sortable_values_lists)
        sort_helper_original_neuron_id_to_IDX_dicts = [dict(zip(neuron_ids, np.arange(len(neuron_ids)))) for neuron_ids in neuron_IDs_lists] # just maps each neuron_id in the list to a fragile_linear_IDX 

        # `sort_helper_neuron_id_to_sort_IDX_dicts` dictionaries in the appropriate order (sorted order) with appropriate indexes. Its .values() can be used to index into things originally indexed with aclus.
        sort_helper_neuron_id_to_sort_IDX_dicts = [{aclu:a_sort_helper_neuron_id_to_IDX_map[aclu] for aclu in sorted_neuron_ids} for a_sort_helper_neuron_id_to_IDX_map, sorted_neuron_ids in zip(sort_helper_original_neuron_id_to_IDX_dicts, sorted_neuron_IDs_lists)]
        # sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[a_sort_list, :] for a_decoder, a_sort_list in zip(decoders_dict.values(), sorted_neuron_IDs_lists)]

        sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]


    """
    ...

def find_nearest_time(df: pd.DataFrame, target_time: float, time_column_name: str = ..., max_allowed_deviation: float = ..., debug_print=...): # -> tuple[DataFrame[Any], int | None, Any | None, Any]:
    """ finds the nearest time in the time_column_name matching the provided target_time
    

    max_allowed_deviation: if provided, requires the difference between the found time in the dataframe and the target_time to be less than or equal to max_allowed_deviation
    Usage:

    from neuropy.utils.indexing_helpers import find_nearest_time
    df = deepcopy(_out_ripple_rasters.active_epochs_df)
    df, closest_index, closest_time, matched_time_difference = find_nearest_time(df=df, target_time=193.65)
    df.iloc[closest_index]

    """
    ...

def find_nearest_times(df: pd.DataFrame, target_times: np.ndarray, time_column_name: str = ..., max_allowed_deviation: float = ..., debug_print=...): # -> tuple[Series[Any], Series[Any]]:
    """
    !! Untested !! a ChatGPT GPT4-turbo written find_nearest_times which extends find_nearest_time to multiple target times. 
    Find the nearest time indices for each target time within the specified max_allowed_deviation.

    Usage:

        from neuropy.utils.indexing_helpers import find_nearest_times

        closest_indices, matched_time_differences = find_nearest_times(df=a_df, target_times=arr, time_column_name='start')
    """
    ...

def convert_to_dictlike(other) -> Dict:
    """ try every known trick to get a plain `dict` out of the provided object. """
    ...

def get_nested_value(d: Dict, keys: List[Any]) -> Any:
    """  how can I index into nested dictionaries using a list of keys? """
    ...

def flatten_dict(d: Dict, parent_key=..., sep=...) -> Dict:
    """ flattens a dictionary to a single non-nested dict

    Example:

    Input:
        {'computation_params': {'merged_directional_placefields': {'laps_decoding_time_bin_size': 0.25, 'ripple_decoding_time_bin_size': 0.025, 'should_validate_lap_decoding_performance': False},
        'rank_order_shuffle_analysis': {'num_shuffles': 500, 'minimum_inclusion_fr_Hz': 5.0, 'included_qclu_values': [1, 2], 'skip_laps': False},
        'directional_decoders_decode_continuous': {'time_bin_size': None},
        'directional_decoders_evaluate_epochs': {'should_skip_radon_transform': False},
            ...
        }

    Output:

        {'merged_directional_placefields/laps_decoding_time_bin_size': 0.25,
            'merged_directional_placefields/ripple_decoding_time_bin_size': 0.025,
            'merged_directional_placefields/should_validate_lap_decoding_performance': False,
            'rank_order_shuffle_analysis/num_shuffles': 500,
            'rank_order_shuffle_analysis/minimum_inclusion_fr_Hz': 5.0,
            'rank_order_shuffle_analysis/included_qclu_values': [1, 2],
            'rank_order_shuffle_analysis/skip_laps': False,
            'directional_decoders_decode_continuous/time_bin_size': None,
            'directional_decoders_evaluate_epochs/should_skip_radon_transform': False,
            ...
            }

    Usage:
        from neuropy.utils.indexing_helpers import flatten_dict
    
    """
    ...

class PandasHelpers:
    """ various extensions and generalizations for numpy arrays 
    
    from neuropy.utils.indexing_helpers import PandasHelpers


    """
    @classmethod
    def require_columns(cls, dfs: Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]], required_columns: List[str], print_missing_columns: bool = ...) -> bool:
        """
        Check if all DataFrames in the given container have the required columns.
        
        Parameters:
            dfs: A container that may be a single DataFrame, a list/tuple of DataFrames, or a dictionary with DataFrames as values.
            required_columns: A list of column names that are required to be present in each DataFrame.
            print_changes: If True, prints the columns that are missing from each DataFrame.
        
        Returns:
            True if all DataFrames contain all the required columns, otherwise False.

        Usage:

            required_cols = ['missing_column', 'congruent_dir_bins_ratio', 'coverage', 'direction_change_bin_ratio', 'jump', 'laplacian_smoothness', 'longest_sequence_length', 'longest_sequence_length_ratio', 'monotonicity_score', 'sequential_correlation', 'total_congruent_direction_change', 'travel'] # Replace with actual column names you require
            has_required_columns = PandasHelpers.require_columns({a_name:a_result.filter_epochs for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}, required_cols, print_missing_columns=True)
            has_required_columns
            


        """
        ...
    
    @classmethod
    def reordering_columns(cls, df: pd.DataFrame, column_name_desired_index_dict: Dict[str, int]) -> pd.DataFrame:
        """Reorders specified columns in a DataFrame while preserving other columns.
        
        Pure: Does not modify the df

        Args:
            df (pd.DataFrame): The DataFrame to reorder.
            column_name_desired_index_dict (Dict[str, int]): A dictionary where keys are column names
                to reorder and values are their desired indices in the reordered DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with specified columns reordered while preserving remaining columns.

        Raises:
            ValueError: If any column in the dictionary is not present in the DataFrame.
            
            
        Usage:
        
            from neuropy.utils.indexing_helpers import PandasHelpers
            dict(zip(['Long_LR_evidence', 'Long_RL_evidence', 'Short_LR_evidence', 'Short_RL_evidence'], np.arange(4)+4))
            PandasHelpers.reorder_columns(merged_complete_epoch_stats_df, column_name_desired_index_dict=dict(zip(['Long_LR_evidence', 'Long_RL_evidence', 'Short_LR_evidence', 'Short_RL_evidence'], np.arange(4)+4)))
            
            ## Move the "height" columns to the end
            result_df = PandasHelpers.reorder_columns(result_df, column_name_desired_index_dict=dict(zip(list(filter(lambda column: column.endswith('_peak_heights'), result_df.columns)), np.arange(len(result_df.columns)-4, len(result_df.columns)))))
            result_df
                    
        """
        ...
    
    @classmethod
    def reordering_columns_relative(cls, df: pd.DataFrame, column_names: list[str], relative_mode=...) -> pd.DataFrame:
        """Reorders specified columns in a DataFrame while preserving other columns.
        
        Pure: Does not modify the df

        Args:
            df (pd.DataFrame): The DataFrame to reorder.
            column_name_desired_index_dict (Dict[str, int]): A dictionary where keys are column names
                to reorder and values are their desired indices in the reordered DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with specified columns reordered while preserving remaining columns.

        Raises:
            ValueError: If any column in the dictionary is not present in the DataFrame.
            
            
        Usage:
        
            ffrom neuropy.utils.indexing_helpers import PandasHelpers
            
            ## Move the "height" columns to the end
            result_df = PandasHelpers.reordering_columns_relative(result_df, column_names=list(filter(lambda column: column.endswith('_peak_heights'), existing_columns)), relative_mode='end')
            result_df
                    
        """
        ...
    
    @classmethod
    def safe_pandas_get_group(cls, dataframe_group, key):
        """ returns an empty dataframe if the key isn't found in the group."""
        ...
    
    @classmethod
    def safe_concat(cls, df_concat_list: Union[List[pd.DataFrame], Dict[Any, pd.DataFrame]], **pd_concat_kwargs) -> Optional[pd.DataFrame]:
        """ returns an empty dataframe if the list of dataframes is empty.
        
        NOTE: does not perform intellegent merging, just handles empty lists
            
            
        Usage:
            from neuropy.utils.indexing_helpers import PandasHelpers

            PandasHelpers.safe_concat
            
        """
        ...
    
    @classmethod
    def convert_dataframe_columns_to_datatype_if_possible(cls, df: pd.DataFrame, datatype_str_column_names_list_dict, debug_print=...): # -> None:
        """ If the columns specified in datatype_str_column_names_list_dict exist in the dataframe df, their type is changed to the key of the dict. See usage example below:
        
        Inputs:
            df: Pandas.DataFrame 
            datatype_str_column_names_list_dict: {'int':['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']}

        Usage:
            from neuropy.utils.indexing_helpers import PandasHelpers
            PandasHelpers.convert_dataframe_columns_to_datatype_if_possible(curr_active_pipeline.sess.spikes_df, {'int':['shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap']})
        """
        ...
    
    @classmethod
    def add_explicit_dataframe_columns_from_lookup_df(cls, df: pd.DataFrame, lookup_properties_map_df: pd.DataFrame, join_column_name=...) -> pd.DataFrame:
        """ Uses a value (specified by `join_column_name`) in each row of `df` to lookup the appropriate values in `lookup_properties_map_df` to be explicitly added as columns to `df`
        df: a dataframe. Each row has a join_column_name value (e.g. 'aclu')
        
        lookup_properties_map_df: a dataframe with one row for each `join_column_name` value (e.g. one row for each 'aclu', describing various properties of that neuron)
        
        
        By default lookup_properties_map_df can be obtained from curr_active_pipeline.sess.neurons._extended_neuron_properties_df and has the columns:
            ['aclu', 'qclu', 'neuron_type', 'shank', 'cluster']
        Which will be added to the spikes_df
        
        WARNING: the df will be unsorted after this operation, and you'll need to sort it again if you want it sorted
        
        
        Usage:
            from neuropy.utils.indexing_helpers import PandasHelpers
            curr_active_pipeline.sess.flattened_spiketrains._spikes_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(curr_active_pipeline.sess.spikes_df, curr_active_pipeline.sess.neurons._extended_neuron_properties_df)
            curr_active_pipeline.sess.spikes_df.sort_values(by=['t_seconds'], inplace=True) # Need to re-sort by timestamps once done
            curr_active_pipeline.sess.spikes_df

        """
        ...
    
    @classmethod
    def adding_additional_df_columns(cls, original_df: pd.DataFrame, additional_cols_df: pd.DataFrame) -> pd.DataFrame:
        """ Adds the columns in `additional_cols_df` to `original_df`, horizontally concatenating them without considering either index.

        Usage:
            
            from neuropy.utils.indexing_helpers import PandasHelpers

            a_result.filter_epochs = PandasHelpers.adding_additional_df_columns(original_df=a_result.filter_epochs, additional_cols_df=_out_new_scores[a_name]) # update the filter_epochs with the new columns

        """
        ...
    


class ColumnTracker(ContextDecorator):
    """A context manager to track changes in the columns of DataFrames.

    The ColumnTracker can handle a single DataFrame, a list or tuple of DataFrames,
    or a dictionary with DataFrames as values. It prints the new columns added to the
    DataFrames during the block where the context manager is active.

    Attributes:
        result_cont: Container (single DataFrame, list/tuple of DataFrames, dict of DataFrames) being monitored.
        pre_cols: Set or structure containing sets of column names before the block execution.
        post_cols: Set or structure containing sets of column names after the block execution.
        added_cols: Set or structure containing sets of added column names after the block execution.
        all_added_columns: List of all unique added column names across all DataFrames after the block execution.


    Usage:

        from neuropy.utils.indexing_helpers import ColumnTracker

        dfs_dict = {a_name:a_result.filter_epochs for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}
        with ColumnTracker(dfs_dict) as tracker:
            # Here you perform the operations that might modify the DataFrame columns
            filtered_decoder_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict)

        # >>> ['longest_sequence_length', 'travel', 'sequential_correlation', 'monotonicity_score', 'jump', 'congruent_dir_bins_ratio', 'total_congruent_direction_change', 'direction_change_bin_ratio', 'longest_sequence_length_ratio', 'laplacian_smoothness', 'coverage']

    """
    def __init__(self, result_cont: Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]]) -> None:
        ...
    
    def __enter__(self) -> ColumnTracker:
        """Enter the runtime context related to this object.

        The with statement will bind this method's return value to the target(s)
        specified in the as clause of the statement, if any.
        """
        ...
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context and perform any finalization actions."""
        ...
    


