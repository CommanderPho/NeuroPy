"""
This type stub file was generated by pyright.
"""

import numpy as np
from nptyping import NDArray
from neuropy.core.neuron_identities import NeuronIdentitiesDisplayerMixin
from neuropy.utils.mixins.binning_helpers import BinnedPositionsMixin
from neuropy.plotting.mixins.ratemap_mixins import RatemapPlottingMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
from neuropy.utils.mixins.HDF5_representable import HDFMixin
from neuropy.utils.mixins.peak_location_representing import ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin
from . import DataWriter

"""
This type stub file was generated by pyright.
"""
class Ratemap(HDFMixin, NeuronIdentitiesDisplayerMixin, RatemapPlottingMixin, ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin, NeuronUnitSlicableObjectProtocol, BinnedPositionsMixin, DataWriter):
    """A Ratemap holds information about each unit's firing rate across binned positions. 
        In addition, it also holds (tuning curves).
        
        
    Internal:
        # Map Properties:
        self.occupancy 
        self.spikes_maps
        self.tuning_curves
        self.unsmoothed_tuning_maps

        # Neuron Identity:
        self._neuron_ids
        self._neuron_extended_ids

        # Position Identity:
        self.xbin
        self.ybin
        
        # Other:
        self.metadata
        
    Args:
        NeuronIdentitiesDisplayerMixin (_type_): _description_
        RatemapPlottingMixin (_type_): _description_
        DataWriter (_type_): _description_
    """
    def __init__(self, tuning_curves, unsmoothed_tuning_maps=..., spikes_maps=..., xbin=..., ybin=..., occupancy=..., neuron_ids=..., neuron_extended_ids=..., metadata=...) -> None:
        ...
    
    @property
    def neuron_ids(self):
        """The neuron_ids property."""
        ...
    
    @neuron_ids.setter
    def neuron_ids(self, value):
        ...
    
    @property
    def neuron_extended_ids(self):
        """The neuron_extended_ids property."""
        ...
    
    @neuron_extended_ids.setter
    def neuron_extended_ids(self, value):
        ...
    
    @property
    def n_neurons(self) -> int:
        ...
    
    @property
    def ndim(self) -> int:
        ...
    
    @property
    def normalized_tuning_curves(self) -> NDArray:
        ...
    
    @property
    def never_visited_occupancy_mask(self) -> NDArray:
        """ a boolean mask that's True everyhwere the animal has never visited according to self.occupancy, and False everyhwere else. """
        ...
    
    @property
    def nan_never_visited_occupancy(self) -> NDArray:
        """ returns the self.occupancy after replacing all never visited locations, indicated by a zero occupancy, by NaNs for the purpose of building visualizations. """
        ...
    
    @property
    def probability_normalized_occupancy(self) -> NDArray:
        """ returns the self.occupancy after converting it to a probability (with each entry between [0.0, 1.0]) by dividing by the sum. """
        ...
    
    @property
    def pdf_normalized_tuning_curves(self):
        """ AOC (area-under-curve) normalization for tuning curves. """
        ...
    
    @property
    def tuning_curve_peak_firing_rates(self):
        """ the non-normalized peak location of each tuning curve. Represents the peak firing rate of that curve. """
        ...
    
    @property
    def tuning_curve_unsmoothed_peak_firing_rates(self):
        """ the non-normalized and unsmoothed value of the maximum firing rate at the peak of each tuning curve in NumSpikes/Second. Represents the peak firing rate of that curve. """
        ...
    
    @property
    def unit_max_tuning_curves(self):
        """ tuning curves normalized by scaling their max value down to 1.0.
            The peak of each placefield will have height 1.0.
        """
        ...
    
    @property
    def minmax_normalized_tuning_curves(self):
        """ tuning curves normalized by scaling their min/max values down to the range (0, 1).
            The peak of each placefield will have height 1.0.
        """
        ...
    
    @property
    def spatial_sparcity(self) -> np.ndarray:
        """ computes the sparcity as a measure of spatial selectivity as in Silvia et al. 2015
        
        Sparcity = \frac{ <f>^2 }{ <f^2> }
        
        """
        ...
    
    def compute_tuning_curve_modes(self):
        """ 2023-12-19 - Uses `scipy.signal.find_peaks to find the number of peaks or ("modes") for each of the cells in the ratemap. 
        Can detect bimodal (or multi-modal) placefields.
        
        Depends on:
            self.tuning_curves
        
        Returns:
            aclu_n_peaks_dict: Dict[int, int] - A mapping between aclu:n_tuning_curve_modes
        Usage:    
            active_ratemap = deepcopy(long_LR_pf1D.ratemap)
            peaks_dict, aclu_n_peaks_dict, unimodal_peaks_dict = compute_tuning_curve_modes(active_ratemap)
            aclu_n_peaks_dict # {2: 4, 5: 4, 7: 2, 8: 2, 9: 2, 10: 5, 17: 2, 24: 2, 25: 3, 26: 1, 31: 3, 32: 5, 34: 2, 35: 1, 36: 2, 37: 2, 41: 4, 45: 3, 48: 4, 49: 4, 50: 4, 51: 3, 53: 5, 54: 3, 55: 5, 56: 4, 57: 4, 58: 5, 59: 3, 61: 4, 62: 3, 63: 4, 64: 4, 66: 3, 67: 4, 68: 2, 69: 2, 71: 3, 73: 3, 74: 3, 75: 5, 76: 5, 78: 3, 81: 3, 82: 1, 83: 4, 84: 4, 86: 3, 87: 3, 88: 4, 89: 3, 90: 3, 92: 4, 93: 4, 96: 2, 97: 4, 98: 5, 100: 4, 102: 7, 107: 1, 108: 5, 109: 2}

        """
        ...
    
    def __getitem__(self, i) -> Ratemap:
        """ Allows accessing via indexing brackets: e.g. `a_ratemap[i]`. Returns a copy of self at the certain indicies """
        ...
    
    def get_by_id(self, ids):
        """Returns self with neuron_ids equal to ids"""
        ...
    
    def get_sort_indicies(self, sortby=...):
        ...
    
    def to_1D_maximum_projection(self):
        ...
    
    @staticmethod
    def nan_ptp(a, **kwargs):
        ...
    
    @staticmethod
    def nanmin_nanmax_scaler(x, axis=..., **kwargs):
        """Scales the values x to lie between 0 and 1 along the specfied axis, ignoring NaNs!
        Parameters
        ----------
        x : np.array
            numpy ndarray
        Returns
        -------
        np.array
            scaled array
        """
        ...
    
    @staticmethod
    def NormalizeData(data):
        """ Simple alternative to the mathutil.min_max_scalar that doesn't produce so man NaN values. """
        ...
    
    @classmethod
    def perform_AOC_normalization(cls, active_tuning_curves, debug_print=...):
        """ Normalizes each cell's tuning map in ratemap by dividing by each cell's area under the curve (AOC). The resultant tuning maps are therefore converted into valid PDFs 
        
        Inputs:
            active_tuning_curves: nd.array
        """
        ...
    
    @staticmethod
    def build_never_visited_mask(occupancy):
        """ returns a mask of never visited locations for the provided occupancy """
        ...
    
    @staticmethod
    def nan_never_visited_locations(occupancy):
        """ replaces all never visited locations, indicated by a zero occupancy, by NaNs for the purpose of building visualizations. """
        ...
    
    @classmethod
    def build_1D_maximum_projection(cls, ratemap_2D: Ratemap) -> Ratemap:
        """ builds a 1D ratemap from a 2D ratemap
        creation_date='2023-04-05 14:02'

        Usage:
            ratemap_1D = build_1D_maximum_projection(ratemap_2D)
        """
        ...
    
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
        """
        ...
    
    @classmethod
    def build_merged_ratemap(cls, lhs: Ratemap, rhs: Ratemap, debug_print=...) -> Ratemap:
        """ Combine the non-directional PDFs and renormalize to get the directional PDF 
        
        Usage:
        
            # Inputs: long_LR_pf1D, long_RL_pf1D
            lhs: Ratemap = deepcopy(long_RL_pf1D.ratemap)
            rhs: Ratemap = deepcopy(long_RL_pf1D.ratemap)
            combined_directional_ratemap = Ratemap.build_merged_ratemap(lhs, rhs)
            combined_directional_ratemap
        
        """
        ...
    


