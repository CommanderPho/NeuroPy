"""
This type stub file was generated by pyright.
"""

from typing import Optional
from nptyping import NDArray

def compute_placefield_center_of_mass_coord_indicies(tuning_curves: NDArray) -> NDArray:
    """ returns the coordinates (index-space, not track-position space) of the center of mass for each of the tuning_curves. """
    ...

def compute_placefield_center_of_mass_positions(tuning_curves: NDArray, xbin: NDArray, ybin: Optional[NDArray] = ...) -> NDArray:
    """ returns the locations of the center of mass for each of the tuning_curves. """
    ...

class ContinuousPeakLocationRepresentingMixin:
    """ Implementors provides peaks in position-space (e.g. a location on the maze) which are computed from a `ContinuousPeakLocationRepresentingMixin_peak_curves_variable` it provides, such as the turning curves.
        
    from neuropy.utils.mixins.peak_location_representing import ContinuousPeakLocationRepresentingMixin
    
    Provides:
        peak_tuning_curve_center_of_mass_bin_coordinates
        peak_tuning_curve_center_of_masses
        
    """
    @property
    def ContinuousPeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
        """ the variable that the peaks are calculated and returned for """
        ...
    
    @property
    def peak_tuning_curve_center_of_mass_bin_coordinates(self) -> NDArray:
        """ returns the coordinates (in bin-index space) of the center of mass of each of the tuning curves."""
        ...
    
    @property
    def peak_tuning_curve_center_of_masses(self) -> NDArray:
        """ returns the locations of the center of mass of each of the tuning curves."""
        ...
    


class PeakLocationRepresentingMixin:
    """ Implementor provides peaks.
    requires: .xbin_centers, .ybin_centers
    requires .tuning_curves
    
	Example:

		from neuropy.utils.mixins.peak_location_representing import PeakLocationRepresentingMixin

		class AClass:		
			...
			# PeakLocationRepresentingMixin conformances:
			@property
			def PeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
				return self.ratemap.PeakLocationRepresentingMixin_peak_curves_variable
			

    """
    @property
    def PeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
        """ the variable that the peaks are calculated and returned for """
        ...
    
    @property
    def peak_indicies(self) -> NDArray:
        ...
    
    @property
    def peak_locations(self) -> NDArray:
        """ returns the peak locations using self.xbin_centers and self.peak_indicies """
        ...
    


