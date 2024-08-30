"""
This type stub file was generated by pyright.
"""

import pandas as pd
from .. import core
from enum import Enum

class RegularizationApproach(Enum):
    """Docstring for RegularizationApproach."""
    RAW_VALUES = ...
    SUBTRACT_MIN = ...
    RESTORE_X_RANGE = ...


def linearize_position_df(pos_df: pd.DataFrame, sample_sec=..., method=..., sigma=..., override_position_sampling_rate_Hz=..., regularization_approach: RegularizationApproach = ...): # -> DataFrame[Any]:
    """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.
    ISOMAP is more versatile but also more computationally expensive.

    Parameters
    ----------
    track_names: list of track names, each must match an epoch in epochs class.
    sample_sec : int, optional
        sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results
    method : str, optional
        by default 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020) or
        'PCA' (for straight tracks)
    override_position_sampling_rate_Hz: float, optional - ignored except for method="isomap". If provided, used for downsampling dataframe prior to computation. Otherwise sampling rate is approximated from pos_df['t'] column.
    
    Modifies:
        Adds the 'lin_pos' column to the provided position dataframe.
    """
    ...

def linearize_position(position: core.Position, sample_sec=..., method=..., sigma=..., **kwargs) -> core.Position:
    """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.
    ISOMAP is more versatile but also more computationally expensive.

    Parameters
    ----------
    track_names: list of track names, each must match an epoch in epochs class.
    sample_sec : int, optional
        sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results
    method : str, optional
        by default 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020) or
        'PCA' (for straight tracks)

    """
    ...

def calculate_run_direction(position: core.Position, speedthresh=..., merge_dur=..., min_dur=..., smooth_speed=..., min_dist=...):
    """Divide running epochs into forward and backward.
    Currently only works for one dimensional position data

    Parameters
    ----------
    speedthresh : tuple, optional
        low and high speed threshold for speed, by default (10, 20)
    merge_dur : int, optional
        two epochs if less than merge_dur (seconds) apart they will be merged , by default 2 seconds
    min_dur : int, optional
        minimum duration of a run epoch, by default 2 seconds
    smooth_speed : int, optional
        speed is smoothed, increase if epochs are fragmented, by default 50
    min_dist : int, optional
        the animal should cover this much distance in one direction within the lap to be included, by default 50
    plot : bool, optional
        plots the epochs with position and speed data, by default True
    """
    ...

def calculate_run_epochs(position: core.Position, speedthresh=..., merge_dur=..., min_dur=..., smooth_speed=...):
    """Divide running epochs into forward and backward.
    Currently only works for one dimensional position data

    Parameters
    ----------
    speedthresh : tuple, optional
        low and high speed threshold for speed, by default (10, 20)
    merge_dur : int, optional
        two epochs if less than merge_dur (seconds) apart they will be merged , by default 2 seconds
    min_dur : int, optional
        minimum duration of a run epoch, by default 2 seconds
    smooth_speed : int, optional
        speed is smoothed, increase if epochs are fragmented, by default 50
    min_dist : int, optional
        the animal should cover this much distance in one direction within the lap to be included, by default 50
    plot : bool, optional
        plots the epochs with position and speed data, by default True
    """
    ...

def compute_position_grid_size(*any_1d_series, num_bins: tuple): # -> tuple[NDArray[float64], list[Any], list[Any]]:
    """  Computes the required bin_sizes from the required num_bins (for each dimension independently)
    Usage:
    out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64))
    active_grid_bin = tuple(out_grid_bin_size)
    print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
    
    History:
        Extracted from pyphocorehelpers.indexing_helpers import compute_position_grid_size for use in BaseDataSessionFormats
    
    """
    ...

