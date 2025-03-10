import unittest
from neuropy.utils.mixins.binning_helpers import BinningContainer
import numpy as np
import pandas as pd
# import the package
import sys, os
from pathlib import Path
from copy import deepcopy

from neuropy.core.epoch import Epoch

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
    from neuropy.core import Position, Neurons
    from neuropy.analyses.placefields import PlacefieldComputationParameters
    from neuropy.analyses.placefields import PfND
    from neuropy.core.neuron_identities import NeuronType
    from neuropy.core.flattened_spiketrains import SpikesAccessor, FlattenedSpiketrains
    from neuropy.utils.debug_helpers import debug_print_placefield, debug_print_subsession_neuron_differences
    from neuropy.utils.debug_helpers import debug_print_ratemap, debug_print_spike_counts, debug_plot_2d_binning, compare_placefields_info
    from neuropy.utils.debug_helpers import parameter_sweeps, _plot_parameter_sweep
    from neuropy.utils.debug_helpers import print_aligned_columns
    from neuropy.analyses.decoders import epochs_spkcount
    


class TestEpochsSpkcountMethods(unittest.TestCase):

    def setUp(self):
        """ Corresponding load for Neuropy Testing file 'NeuroPy/tests/neuropy_pf_testing.h5': 
            ## Save for NeuroPy testing:
            finalized_testing_file='../NeuroPy/tests/neuropy_pf_testing.h5'
            sess_identifier_key='sess'
            spikes_df.to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
            active_pos.to_dataframe().to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df', format='table')
        """
        self.enable_debug_plotting = False
        self.enable_debug_printing = True

        finalized_testing_file = tests_folder.joinpath('neuropy_pf_testing.h5')
        sess_identifier_key='sess'
        # Load the saved .h5 spikes_df and active_pos dataframes for testing:
        self.spikes_df = pd.read_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
        active_pos_df = pd.read_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df')
        self.active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object
        
        self.test_epochs_df = pd.DataFrame({'start': [0.0, 1029.316608761903], 'stop': [1029.316608761903, 1737.1968310000375], 'label': ['maze1', 'maze2'], 'duration': [1029.316608761903, 707.8802222381346]})
        self.epochs = Epoch(self.test_epochs_df) # Epoch(...) # Create an Epoch object as needed
        
        # Create a PfND object
        self.config = PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(2, 2), grid_bin_bounds=((29.16, 261.7), (130.23, 150.99)), smooth=(2.0, 2.0), frate_thresh=1.0)
        self.pfnd = PfND(self.spikes_df, self.active_pos, self.epochs, config=self.config, position_srate=self.active_pos.sampling_rate)
        self.hdf_tests_file = 'test_pfnd.h5'
        
        # self.spikes_df = pd.DataFrame({
        #     'neuron_id': [1, 1, 2, 2, 3],
        #     'time': [0.1, 0.2, 0.15, 0.25, 0.3]
        # })


    def test_single_time_bin_per_epoch(self):
        spkcount, nbins, time_bins = epochs_spkcount(
            self.spikes_df, 
            self.test_epochs_df,
            bin_size=0.1,
            export_time_bins=True,
            use_single_time_bin_per_epoch=True
        )
        self.assertEqual(len(nbins), 2)
        self.assertTrue(all(nbins == 1))
        self.assertEqual(len(time_bins), 2)

    def test_short_epoch_handling(self):
        short_epochs_df = pd.DataFrame({
            'start': [0.0, 0.5],
            'stop': [0.005, 0.51]
        })
        spkcount, nbins, time_bins = epochs_spkcount(
            self.spikes_df,
            short_epochs_df,
            bin_size=0.01,
            export_time_bins=True
        )
        self.assertEqual(len(nbins), 2)
        self.assertTrue(all(nbins == 1))

    def test_included_neuron_ids(self):
        specific_neuron_ids = [1, 2, 3, 4]
        spkcount, nbins, _ = epochs_spkcount(
            self.spikes_df,
            self.test_epochs_df,
            included_neuron_ids=specific_neuron_ids,
            bin_size=0.1
        )
        self.assertEqual(len(spkcount[0]), len(specific_neuron_ids))

    def test_variable_slideby(self):
        spkcount, nbins, _ = epochs_spkcount(
            self.spikes_df,
            self.test_epochs_df,
            bin_size=0.1,
            slideby=0.05
        )
        self.assertTrue(all(n > 0 for n in nbins))

    def test_empty_epochs(self):
        empty_epochs_df = pd.DataFrame({
            'start': [],
            'stop': []
        })
        spkcount, nbins, time_bins = epochs_spkcount(
            self.spikes_df,
            empty_epochs_df,
            bin_size=0.1,
            export_time_bins=True
        )
        self.assertEqual(len(spkcount), 0)
        self.assertEqual(len(nbins), 0)
        self.assertEqual(len(time_bins), 0)
        

# class TestEpochsSpkcountAdvanced(unittest.TestCase):

#     def setUp(self):
#         self.spikes_df = pd.DataFrame({
#             'neuron_id': [1, 1, 2, 2, 3, 3, 4],
#             'time': [0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4]
#         })
#         self.epochs_df = pd.DataFrame({
#             'start': [0.0, 0.5, 1.0],
#             'stop': [0.4, 0.8, 1.2]
#         })

#     def test_overlapping_epochs(self):
#         overlapping_epochs = pd.DataFrame({
#             'start': [0.0, 0.2, 0.4],
#             'stop': [0.3, 0.5, 0.6]
#         })
#         spkcount, nbins, _ = epochs_spkcount(
#             self.spikes_df,
#             overlapping_epochs,
#             bin_size=0.1
#         )
#         self.assertEqual(len(spkcount), 3)
#         self.assertEqual(len(nbins), 3)

#     def test_empty_spikes(self):
#         empty_spikes_df = pd.DataFrame({
#             'neuron_id': [],
#             'time': []
#         })
#         spkcount, nbins, _ = epochs_spkcount(
#             empty_spikes_df,
#             self.epochs_df,
#             bin_size=0.1
#         )
#         self.assertEqual(len(spkcount), 3)
#         for count in spkcount:
#             self.assertEqual(np.sum(count), 0)

#     def test_large_bin_size(self):
#         spkcount, nbins, time_bins = epochs_spkcount(
#             self.spikes_df,
#             self.epochs_df,
#             bin_size=1.0,
#             export_time_bins=True
#         )
#         self.assertTrue(all(n == 1 for n in nbins))
#         self.assertEqual(len(time_bins), 3)

#     def test_microsecond_bin_size(self):
#         spkcount, nbins, _ = epochs_spkcount(
#             self.spikes_df,
#             self.epochs_df,
#             bin_size=0.001
#         )
#         self.assertTrue(all(n > 0 for n in nbins))

#     def test_non_chronological_epochs(self):
#         reversed_epochs = pd.DataFrame({
#             'start': [1.0, 0.5, 0.0],
#             'stop': [1.2, 0.8, 0.4]
#         })
#         spkcount, nbins, _ = epochs_spkcount(
#             self.spikes_df,
#             reversed_epochs,
#             bin_size=0.1
#         )
#         self.assertEqual(len(spkcount), 3)
#         self.assertEqual(len(nbins), 3)

#     def test_zero_duration_epochs(self):
#         zero_duration_epochs = pd.DataFrame({
#             'start': [0.1, 0.2, 0.3],
#             'stop': [0.1, 0.2, 0.3]
#         })
#         spkcount, nbins, time_bins = epochs_spkcount(
#             self.spikes_df,
#             zero_duration_epochs,
#             bin_size=0.1,
#             export_time_bins=True
#         )
#         self.assertEqual(len(spkcount), 3)
#         self.assertTrue(all(n == 1 for n in nbins))

#     def test_variable_slideby_with_export(self):
#         spkcount, nbins, time_bins = epochs_spkcount(
#             self.spikes_df,
#             self.epochs_df,
#             bin_size=0.2,
#             slideby=0.1,
#             export_time_bins=True
#         )
#         self.assertEqual(len(time_bins), 3)
#         self.assertTrue(all(isinstance(container, BinningContainer) for container in time_bins))

#     def test_negative_time_values(self):
#         negative_spikes_df = pd.DataFrame({
#             'neuron_id': [1, 2],
#             'time': [-0.1, -0.2]
#         })
#         negative_epochs = pd.DataFrame({
#             'start': [-0.5, -0.3],
#             'stop': [-0.3, -0.1]
#         })
#         spkcount, nbins, _ = epochs_spkcount(
#             negative_spikes_df,
#             negative_epochs,
#             bin_size=0.1
#         )
#         self.assertEqual(len(spkcount), 2)
#         self.assertEqual(len(nbins), 2)
    

class TestDecodersMethods(unittest.TestCase):

    def setUp(self):
        """ Corresponding load for Neuropy Testing file 'NeuroPy/tests/neuropy_pf_testing.h5': 
            ## Save for NeuroPy testing:
            finalized_testing_file='../NeuroPy/tests/neuropy_pf_testing.h5'
            sess_identifier_key='sess'
            spikes_df.to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
            active_pos.to_dataframe().to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df', format='table')
        """
        self.enable_debug_plotting = False
        self.enable_debug_printing = True

        finalized_testing_file = tests_folder.joinpath('neuropy_pf_testing.h5')
        sess_identifier_key='sess'
        # Load the saved .h5 spikes_df and active_pos dataframes for testing:
        self.spikes_df = pd.read_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
        active_pos_df = pd.read_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df')
        self.active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object
        self.epochs = None # Epoch(...) # Create an Epoch object as needed
        
        # Create a PfND object
        self.config = PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(2, 2), grid_bin_bounds=((29.16, 261.7), (130.23, 150.99)), smooth=(2.0, 2.0), frate_thresh=1.0)
        self.pfnd = PfND(self.spikes_df, self.active_pos, self.epochs, config=self.config, position_srate=self.active_pos.sampling_rate)
        self.hdf_tests_file = 'test_pfnd.h5'



    def tearDown(self):
        # Clean up the test file
        if os.path.exists(self.hdf_tests_file):
            os.remove(self.hdf_tests_file)




if __name__ == '__main__':
    unittest.main()
