"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from neuropy.analyses.placefields import PlacefieldComputationParameters

if TYPE_CHECKING:
    ...
def unwrap_placefield_computation_parameters(computation_config) -> PlacefieldComputationParameters:
    """ Extract the older PlacefieldComputationParameters from the newer computation_config (which is a dynamic_parameters object), which should have a field .pf_params
    If the computation_config is passed in the old-style (already a PlacefieldComputationParameters) it's returned unchanged.

    Usage:
        active_pf_computation_params = unwrap_placefield_computation_parameters(active_config.computation_config)
    """
    ...

