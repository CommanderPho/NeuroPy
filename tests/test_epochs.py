import unittest
from neuropy.utils.misc import split_array
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from nptyping import NDArray
# import the package
import sys, os
from pathlib import Path
from copy import deepcopy

from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter

# Add Neuropy to the path as needed
tests_folder = Path(os.path.dirname(__file__))

try:
    import neuropy
except ModuleNotFoundError as e:    
    root_project_folder = tests_folder.parent
    print('root_project_folder: {}'.format(root_project_folder))
    neuropy_folder = root_project_folder.joinpath('neuropy')
    print('neuropy_folder: {}'.format(neuropy_folder))
    sys.path.insert(0, str(root_project_folder))
finally:
    from neuropy.utils.misc import is_iterable
    from neuropy.core.epoch import find_data_indicies_from_epoch_times, Epoch
    from neuropy.utils.indexing_helpers import find_nearest_time



def _test_find_data_index_methods(test_epochs_data_df: pd.DataFrame, selection_start_stop_times: NDArray):
    print(f'np.shape(selection_start_stop_times): {np.shape(selection_start_stop_times)}')

    print(f'np.shape(test_epochs_data_df): {np.shape(test_epochs_data_df)}')

    # 2D_search (for both start, end times):
    found_data_indicies_2D_search = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=selection_start_stop_times)
    print(f'np.shape(found_data_indicies): {np.shape(found_data_indicies_2D_search)}')

    # 1D_search (only for start times):
    found_data_indicies_1D_search = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=np.squeeze(selection_start_stop_times[:, 0]))
    print(f'np.shape(found_data_indicies_1D_search): {np.shape(found_data_indicies_1D_search)}')
    # found_data_indicies_1D_search
    
    return found_data_indicies_2D_search, found_data_indicies_1D_search

def _perform_test_find_nearest_time(df: pd.DataFrame, max_allowed_deviation:float = 0.001):
    """ 
    df = deepcopy(original_epochs_df)
    found_df = _perform_test_find_nearest_time(df=df, max_allowed_deviation = 0.001)
    found_df

    """
    ## Get the rounded versions of each epoch's start/stop times, like would be done for user annotations:
    _input_df_shape = np.shape(df)
    print(f'_input_df_shape: {_input_df_shape}')
    # [f"{start:.3f}, {stop:.3f}" for start, stop in original_epochs_df[['start', 'stop']].itertuples(index=False)]
    _full_precision_start_stop_times_arr = deepcopy(df[['start', 'stop']].to_numpy())
    original_epochs_df_derived_start_stop_times_strs: List[Tuple[str, str]] = [(f"{start:.3f}", f"{stop:.3f}") for start, stop in df[['start', 'stop']].itertuples(index=False)] # throw aways the precision like is done when saving the annotations
    ## Convert them back:
    original_epochs_df_derived_start_stop_times: List[Tuple[float, float]] = [(float(start_str), float(stop_str)) for (start_str, stop_str) in original_epochs_df_derived_start_stop_times_strs]
    original_epochs_df_derived_start_stop_times_arr = np.array(original_epochs_df_derived_start_stop_times)
    original_epochs_df_derived_start_stop_times_arr.shape # (718, 2)
    original_epochs_df_derived_start_stop_times_arr

    theoretical_deviations_diff = (_full_precision_start_stop_times_arr - original_epochs_df_derived_start_stop_times_arr)
    max_deviation: float = np.nanmax(np.abs(theoretical_deviations_diff)) # 0.0004999862649128772
    print(f'max_deviation: {max_deviation}')

    [(start, stop) for start, stop in original_epochs_df_derived_start_stop_times_arr]
    # original_epochs_df_derived_start_stop_times_arr

    # df, closest_index, closest_time, matched_time_difference = find_nearest_time(df=df, target_time=193.65)

    # [find_nearest_time(df=df, target_time=start) for start, stop in original_epochs_df_derived_start_stop_times_arr]
    # max_allowed_deviation:float = 0.0001
    # max_allowed_deviation:float = 0.001
    closest_epoch_info_tuples = [find_nearest_time(df=df, target_time=start, max_allowed_deviation=max_allowed_deviation)[1:] for start, stop in original_epochs_df_derived_start_stop_times_arr]
    closest_epoch_indicies = [closest_index for closest_index, closest_time, matched_time_difference in closest_epoch_info_tuples]
    closest_epoch_indicies
    # closest_index, closest_time, matched_time_difference

    closest_epoch_indicies_arr = np.array(closest_epoch_indicies)
    closest_epoch_indicies_arr

    # [df.loc[found_idx] for found_idx in closest_epoch_indicies]

    found_df = deepcopy(df.loc[closest_epoch_indicies_arr])
    found_df

    _out_df_shape = np.shape(found_df)
    print(f'_out_df_shape: {_out_df_shape}')
    assert _input_df_shape[0] == _out_df_shape[0], f"_input_df_shape[0]: {_input_df_shape[0]} != _out_df_shape[0]: {_out_df_shape[0]}"
    return found_df


# find_data_indicies_from_epoch_times

class TestEpochMethods(unittest.TestCase):
    def setUp(self):
        """ Corresponding load for Neuropy Testing file 'NeuroPy/tests/neuropy_pf_testing.h5': 
            ## Save for NeuroPy testing:
            finalized_testing_file='../NeuroPy/tests/neuropy_pf_testing.h5'
            sess_identifier_key='sess'
            spikes_df.to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
            active_pos.to_dataframe().to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df', format='table')
        """
        ## Load testing variables from file 'NeuroPy/tests/neuropy_pf_testing.h5'
        """ Corresponding load for Neuropy Testing file 'NeuroPy/tests/neuropy_epochs_testing.h5': 
            ## Save for NeuroPy testing:
            finalized_output_cache_file='../NeuroPy/tests/neuropy_epochs_testing.h5'
            sess_identifier_key='sess'
            active_epochs_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/selected_epochs_df', format='table')
            test_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/test_df', format='table')
        """
        self.finalized_output_cache_file='../NeuroPy/tests/neuropy_epochs_testing.h5'
        self.sess_identifier_key='sess'
        # Load the saved .h5 spikes_df and active_pos dataframes for testing:
        # self.epochs_df = pd.read_hdf(self.finalized_output_cache_file, key=f'{sess_identifier_key}/epochs_df')
        # self.selected_epochs_df = pd.read_hdf(self.finalized_output_cache_file, key=f'{sess_identifier_key}/selected_epochs_df')
        # self.test_df = pd.read_hdf(self.finalized_output_cache_file, key=f'{sess_identifier_key}/test_df')
        self.original_epochs_df = pd.read_hdf(self.finalized_output_cache_file, key=f'{self.sess_identifier_key}/original_epochs_df')
        self.filtered_epochs_df = pd.read_hdf(self.finalized_output_cache_file, key=f'{self.sess_identifier_key}/filtered_epochs_df')
        any_good_selected_epoch_times_df = pd.read_hdf(self.finalized_output_cache_file, key=f'{self.sess_identifier_key}/any_good_selected_epoch_times_df')
        self.any_good_selected_epoch_times = any_good_selected_epoch_times_df[['start', 'stop']].to_numpy()

        self.enable_debug_plotting = False
        self.enable_debug_printing = True


    def tearDown(self):
        # Clean up the test file
        # if os.path.exists(self.hdf_tests_file):
        #     os.remove(self.hdf_tests_file)


        pass


    def test_find_nearest_time_after_string_truncation(self):
        from neuropy.utils.indexing_helpers import find_nearest_time
        print(f'self.original_epochs_df: {self.original_epochs_df.shape}')        
        ## Get the rounded versions of each epoch's start/stop times, like would be done for user annotations:
        # with self.assertRaises(AssertionError):
        df = deepcopy(self.original_epochs_df)
        found_df = _perform_test_find_nearest_time(df=df, max_allowed_deviation = 0.001)
        found_df
            # self.assertFalse
        # self.raise


    def test_find_nearest_time_after_string_truncation_filtered_epochs(self):
        from neuropy.utils.indexing_helpers import find_nearest_time
        print(f'self.filtered_epochs_df: {self.filtered_epochs_df.shape}')        
        ## Get the rounded versions of each epoch's start/stop times, like would be done for user annotations:
        ## Test on filtered epochs
        df = deepcopy(self.filtered_epochs_df)
        found_df = _perform_test_find_nearest_time(df=df, max_allowed_deviation = 0.001)
        found_df





    def test_real_world_use(self):

        print(f'self.original_epochs_df: {self.original_epochs_df.shape}')
        print(f'self.filtered_epochs_df: {self.filtered_epochs_df.shape}')
        print(f'self.any_good_selected_epoch_times: {np.shape(self.any_good_selected_epoch_times)}')

        # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import filter_and_update_epochs_and_spikes

        # 2024-03-04 - Filter out the epochs based on the criteria:
        # filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, active_min_num_unique_aclu_inclusions_requirement, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
        # print(f'')


        # _test_find_data_index_methods(filtered_epochs_df)

        ## filter the epochs by something and only show those:
        # INPUTS: filtered_epochs_df
        # filtered_ripple_simple_pf_pearson_merged_df = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(active_epochs_df[['start', 'stop']].to_numpy())
        ## Update the `decoder_ripple_filter_epochs_decoder_result_dict` with the included epochs:
        # filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(filtered_epochs_df[['start', 'stop']].to_numpy()) for a_name, a_result in decoder_ripple_filter_epochs_decoder_result_dict.items()}
        # filtered_decoder_filter_epochs_decoder_result_dict



    # def test_isiterable(self):
    #     self.assertFalse(is_iterable(0), "integer should return false for is_iterable")
    #     self.assertFalse(is_iterable(0.011), "float should return false for is_iterable")
    #     self.assertFalse(is_iterable('string_value'), "a string literal should return false for is_iterable")
        
    #     self.assertTrue(is_iterable(['string_value', 'string_value_2']), "a list of string literals should return true for is_iterable")
    #     self.assertTrue(is_iterable((0, 1, 2)), "tuple should return true for is_iterable")
    #     self.assertTrue(is_iterable([0, 1, 2]), "list should return true for is_iterable")
    #     self.assertTrue(is_iterable(set()), "set should return true for is_iterable")
    #     self.assertTrue(is_iterable(dict()), "dict should return true for is_iterable")
        
    #     self.assertTrue(is_iterable(np.array([0, 1, 2])), "ndArray should return true for is_iterable")
    #     self.assertTrue(is_iterable(pd.DataFrame({'col':[0, 1, 2]})), "Pandas DataFrame should return true for is_iterable")
        


    # def test_string_literal_comparable_enum(self):
    #     test1 = UnitColoringMode.PRESERVE_FRAGILE_LINEAR_NEURON_IDXS
    #     test2 = UnitColoringMode.COLOR_BY_INDEX_ORDER

    #     self.assertIsInstance(test1.value, str, "Value of enum must be string")
    #     self.assertEqual(test1.value, "preserve_fragile_linear_neuron_IDXs")
    #     self.assertNotEqual(test1, test2)
    #     self.assertEqual(test1, "preserve_fragile_linear_neuron_IDXs")
    #     self.assertEqual(test1, "preserve_fragile_linear_NEURON_IDXs")
    #     self.assertNotEqual(test1, "color_by_index_order") # compare to wrong value
    #     self.assertNotEqual(test1, "a_fake_value") # compare to fake value



    # def test_progress_message_printer(self):
    #     # TODO: not sure how to test this
    #     mat_import_file = 'data/RoyMaze1/positionAnalysis.mat'
    #     with ProgressMessagePrinter(mat_import_file, 'Loading', 'matlab import file', returns_string=False):
    #         print('\t inside the with statement', end=' ', file=sys.stdout)

    #     # TODO: check the output buffer for something like this:
    #     # 'Loading matlab import file results to data/RoyMaze1/positionAnalysis.mat... 	 inside the with statement done.'
    #     # assert is_iterable([0, 1, 2]), "list should return true for is_iterable"
    #     # assert is_iterable(np.array([0, 1, 2])), "ndArray should return true for is_iterable"
    #     # assert is_iterable(pd.DataFrame({'col', [0, 1, 2]})), "Pandas DataFrame should return true for is_iterable"

    # def test_split_array(self):
    #     arr = np.array([1, 2, 3, 4, 5, 6])
    #     sub_element_lengths = np.array([2, 1, 3])
    #     # expected_output = [[1, 2], [3], [4, 5, 6]]
    #     expected_output = [np.array([1, 2]), np.array([3]), np.array([4, 5, 6])]
    #     output = split_array(arr, sub_element_lengths)
    #     print(f'expected_output: {expected_output}\n output: {output}')
    #     # self.assertEqual([a == b] in zip(output, expected_output)])
    #     # self.assertListEqual(output, expected_output)
    #     # self.assertTrue(np.allclose(output, expected_output))
        
    # def test_sum_of_sub_element_lengths_not_equal_to_N(self):
    #     arr = np.array([1, 2, 3, 4, 5, 6])
    #     sub_element_lengths = np.array([2, 1, 2])
    #     with self.assertRaises(ValueError):
    #         split_array(arr, sub_element_lengths)
            
    # def test_return_type(self):
    #     arr = np.array([1, 2, 3, 4, 5, 6])
    #     sub_element_lengths = np.array([2, 1, 3])
    #     self.assertIsInstance(split_array(arr, sub_element_lengths), list)
        
    # def test_empty_array(self):
    #     arr = np.array([])
    #     sub_element_lengths = np.array([])
    #     expected_output = []
    #     self.assertListEqual(split_array(arr, sub_element_lengths), expected_output).all()
        
    # def test_sub_element_lengths_containing_zeros(self):
    #     arr = np.array([1, 2, 3, 4, 5, 6])
    #     sub_element_lengths = np.array([2, 0, 3, 1])
    #     with self.assertRaises(ValueError):
    #         split_array(arr, sub_element_lengths)


    def test_get_non_overlapping_df_empty(self):
        """Test with an empty dataframe."""
        empty_df = pd.DataFrame(columns=['start', 'stop', 'label', 'duration'])
        result = empty_df.epochs.get_non_overlapping_df()
        self.assertEqual(len(result), 0, "Empty dataframe should return empty result")

    def test_get_non_overlapping_df_single_epoch(self):
        """Test with a single epoch."""
        # Extract the first epoch from the existing test data
        if len(self.original_epochs_df) > 0:
            single_epoch_df = self.original_epochs_df.iloc[[0]].reset_index(drop=True)
            result = single_epoch_df.epochs.get_non_overlapping_df()
            self.assertEqual(len(result), 1, "Single epoch should be preserved")
            self.assertEqual(result.iloc[0]['start'], single_epoch_df.iloc[0]['start'])
            self.assertEqual(result.iloc[0]['stop'], single_epoch_df.iloc[0]['stop'])

    def test_get_non_overlapping_df_non_overlapping(self):
        """Test with non-overlapping epochs."""
        # Create non-overlapping epochs from existing test data by ensuring gaps
        if len(self.original_epochs_df) >= 3:
            indices = [0, 10, 20]  # Assuming these indices exist and are non-overlapping
            non_overlapping_df = self.original_epochs_df.iloc[indices].copy().reset_index(drop=True)
            
            # Ensure they don't overlap
            for i in range(len(non_overlapping_df)-1):
                if non_overlapping_df.iloc[i]['stop'] >= non_overlapping_df.iloc[i+1]['start']:
                    non_overlapping_df.at[i, 'stop'] = non_overlapping_df.iloc[i+1]['start'] - 0.1
                    non_overlapping_df.at[i, 'duration'] = non_overlapping_df.iloc[i]['stop'] - non_overlapping_df.iloc[i]['start']
            
            result = non_overlapping_df.epochs.get_non_overlapping_df()
            self.assertEqual(len(result), len(non_overlapping_df), "Non-overlapping epochs should all be preserved")
            self.assertTrue(np.array_equal(result['start'].values, non_overlapping_df['start'].values))
            self.assertTrue(np.array_equal(result['stop'].values, non_overlapping_df['stop'].values))

    def test_get_non_overlapping_df_gapless(self):
        """Test with gapless epochs (stop[i] == start[i+1])."""
        # Create gapless epochs
        gapless_df = pd.DataFrame({
            'start': [0.0, 1.0, 2.0],
            'stop': [1.0, 2.0, 3.0],
            'label': ['0', '1', '2'],
            'duration': [1.0, 1.0, 1.0]
        })
        
        result = gapless_df.epochs.get_non_overlapping_df()
        
        # Should apply epsilon to make them non-overlapping
        self.assertEqual(len(result), 3, "All gapless epochs should be preserved")
        
        # Check that stop times have been modified slightly
        # The first two stops should be slightly less than the original values
        self.assertLess(result.iloc[0]['stop'], 1.0)
        self.assertLess(result.iloc[1]['stop'], 2.0)
        
        # But they should be very close to the original values
        self.assertTrue(np.isclose(result.iloc[0]['stop'], 1.0, rtol=1e-8))
        self.assertTrue(np.isclose(result.iloc[1]['stop'], 2.0, rtol=1e-8))
        
        # The last stop should be unchanged
        self.assertEqual(result.iloc[2]['stop'], 3.0)

    def test_get_non_overlapping_df_overlapping(self):
        """Test with overlapping epochs."""
        # Create overlapping epochs
        overlapping_df = pd.DataFrame({
            'start': [0.0, 1.0, 2.0],
            'stop': [1.5, 2.5, 3.0],
            'label': ['0', '1', '2'],
            'duration': [1.5, 1.5, 1.0]
        })
        
        result = overlapping_df.epochs.get_non_overlapping_df()
        
        # PortionInterval should handle the overlaps
        self.assertTrue(len(result) > 0, "Should have some epochs")
        
        # Check that result does not have overlaps
        for i in range(len(result) - 1):
            self.assertLessEqual(result.iloc[i]['stop'], result.iloc[i+1]['start'], 
                                "Result should not have overlapping epochs")

    def test_get_non_overlapping_df_mixed(self):
        """Test with a mix of gapless and overlapping epochs."""
        mixed_df = pd.DataFrame({
            'start': [0.0, 1.0, 1.8, 3.0],
            'stop':  [1.0, 2.0, 3.0, 4.0],
            'label': ['0', '1', '2', '3'],
            'duration': [1.0, 1.0, 1.2, 1.0]
        })
        
        result = mixed_df.epochs.get_non_overlapping_df()
        
        # The function should handle both gapless and overlapping cases
        self.assertTrue(len(result) > 0, "Should have some epochs")
        
        # Check that result does not have overlaps
        for i in range(len(result) - 1):
            self.assertLessEqual(result.iloc[i]['stop'], result.iloc[i+1]['start'], 
                                "Result should not have overlapping epochs")

    def test_get_non_overlapping_df_metadata_preservation(self):
        """Test that metadata is preserved."""
        # Clone existing dataframe and add metadata
        df_with_metadata = self.original_epochs_df.iloc[:2].copy().reset_index(drop=True)
        test_metadata = {'test_key': 'test_value'}
        df_with_metadata.attrs = test_metadata
        
        result = df_with_metadata.epochs.get_non_overlapping_df()
        self.assertTrue(hasattr(result, 'attrs'), "Result should have attrs attribute")
        self.assertEqual(result.attrs, test_metadata, "Metadata should be preserved")

    def test_get_non_overlapping_df_extra_columns(self):
        """Test that extra columns are preserved when possible."""
        # Clone existing dataframe and add extra columns
        df_with_extra = self.original_epochs_df.iloc[:3].copy().reset_index(drop=True)
        
        # Ensure non-overlapping for simple test
        for i in range(len(df_with_extra)-1):
            if df_with_extra.iloc[i]['stop'] >= df_with_extra.iloc[i+1]['start']:
                df_with_extra.at[i, 'stop'] = df_with_extra.iloc[i+1]['start'] - 0.1
                df_with_extra.at[i, 'duration'] = df_with_extra.iloc[i]['stop'] - df_with_extra.iloc[i]['start']
        
        # Add extra columns
        df_with_extra['test_column'] = range(len(df_with_extra))
        df_with_extra['another_column'] = [f"value_{i}" for i in range(len(df_with_extra))]
        
        result = df_with_extra.epochs.get_non_overlapping_df()
        
        # Should preserve all columns in the simple non-overlapping case
        self.assertIn('test_column', result.columns, "Column 'test_column' should be preserved")
        self.assertIn('another_column', result.columns, "Column 'another_column' should be preserved")
        if len(result) == len(df_with_extra):
            self.assertTrue(np.array_equal(result['test_column'].values, df_with_extra['test_column'].values),
                        "Values in column 'test_column' should be preserved")

    def test_get_non_overlapping_df_with_debug_print(self):
        """Test the debug_print parameter."""
        # Use existing test data with debug_print=True
        if len(self.filtered_epochs_df) > 0:
            test_df = self.filtered_epochs_df.iloc[:2].copy().reset_index(drop=True)
            result = test_df.epochs.get_non_overlapping_df(debug_print=True)
            self.assertEqual(len(result), len(test_df), "Should preserve epochs with debug_print=True")

    def test_get_non_overlapping_df_with_epoch_object(self):
        """Test integration with the Epoch class."""
        if len(self.original_epochs_df) > 0:
            # Create an Epoch object from existing test data
            from neuropy.core.epoch import Epoch
            epoch_df = self.original_epochs_df.iloc[:3].copy().reset_index(drop=True)
            
            # Create some overlap to test the functionality
            if len(epoch_df) >= 2:
                epoch_df.at[0, 'stop'] = epoch_df.iloc[1]['start'] + 0.1
                epoch_df.at[0, 'duration'] = epoch_df.iloc[0]['stop'] - epoch_df.iloc[0]['start']
            
            epoch_obj = Epoch(epoch_df)
            
            # Get non-overlapping epochs
            result_epoch = epoch_obj.get_non_overlapping(debug_print=self.enable_debug_printing)
            
            # Check the result
            self.assertIsInstance(result_epoch, Epoch, "Result should be an Epoch object")
            self.assertTrue(len(result_epoch) > 0, "Result should have epochs")
            
            # Verify no overlaps in result
            result_df = result_epoch.to_dataframe()
            for i in range(len(result_df) - 1):
                self.assertLessEqual(result_df.iloc[i]['stop'], result_df.iloc[i+1]['start'], 
                                    "Result should not have overlapping epochs")

    def test_get_non_overlapping_df_with_real_data(self):
        """Test with real data from the test fixtures."""
        # Use the self.original_epochs_df directly
        if self.enable_debug_printing:
            print(f"Testing with original epochs data: {len(self.original_epochs_df)} epochs")
        
        result = self.original_epochs_df.epochs.get_non_overlapping_df(debug_print=self.enable_debug_printing)
        
        # Check that result has no overlaps
        for i in range(len(result) - 1):
            self.assertLessEqual(result.iloc[i]['stop'], result.iloc[i+1]['start'], 
                            "Result should not have overlapping epochs")
        
        if self.enable_debug_printing:
            print(f"Result has {len(result)} non-overlapping epochs")



if __name__ == '__main__':
    unittest.main()
