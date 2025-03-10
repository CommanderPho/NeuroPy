import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import nptyping as ND

import numpy as np
import pandas as pd

from pathlib import Path
from copy import deepcopy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.special import factorial

from neuropy.analyses.placefields import PfND


from neuropy import core
# from .. import core

from neuropy.utils.mixins.binning_helpers import BinningContainer, BinningInfo # for epochs_spkcount getting the correct time bins
from neuropy.utils.mixins.binning_helpers import build_spanning_grid_matrix # for Decode2d reverse transformations from flat points

from attrs import define, field, Factory

class RadonTransformComputation:
    """
    Helpers for radon transform:

    from neuropy.analyses.decoders import RadonTransformComputation

    """
    @classmethod
    def phi(cls, velocity, time_bin_size, pos_bin_size):
        return np.arctan(velocity * time_bin_size / pos_bin_size)
    
    @classmethod
    def rho(cls, icpt, t_mid, x_mid, velocity, time_bin_size, pos_bin_size):
        phi = cls.phi(velocity=velocity, time_bin_size=time_bin_size, pos_bin_size=pos_bin_size)
        return ((icpt + (velocity * t_mid) - x_mid)/pos_bin_size) * np.sin(phi)
    
    # Conversion functions: ______________________________________________________________________________________________ #
    @classmethod
    def convert_real_space_x_to_index_space_ri(cls, ri_mid, x_mid, pos_bin_size):
        convert_real_space_x_to_index_space_ri = lambda x: (((x - x_mid)/pos_bin_size) + ri_mid)
        return convert_real_space_x_to_index_space_ri

    @classmethod
    def convert_real_time_t_to_index_time_ci(cls, ci_mid, t_mid, time_bin_size):
        convert_real_time_t_to_index_time_ci = lambda t: (((t - t_mid)/time_bin_size) + ci_mid)
        return convert_real_time_t_to_index_time_ci

    # Used in `radon_transform` __________________________________________________________________________________________ #
    @classmethod
    def velocity(cls, phi, time_bin_size, pos_bin_size):
        """ Not working.
        
        """
        return pos_bin_size / (time_bin_size * np.tan(phi)) # 1/np.tan(x) == cot(x)
    
    @classmethod
    def intercept(cls, phi, rho, t_mid, x_mid, time_bin_size, pos_bin_size):
        """ Not working.
            t_mid, x_mid: the continuous-time versions
        """
        return (
            (pos_bin_size * t_mid) / (time_bin_size * np.tan(phi))
            + (rho / np.sin(phi)) * pos_bin_size
            + x_mid
        )

    @classmethod
    def y_line_idxs(cls, phi, rho, ci_mid, ri_mid, ci_mat=None):
        """ 
        returns a lambda function that takes `ci_mat`
        y_line_idxs_fn = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, ci_mid=ci_mid, ri_mid=ri_mid)
        y_line_idxs = y_line_idxs_fn(ci_mat=ci_mat)

        ---

        y_line_idxs = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, ci_mid=ci_mid, ri_mid=ri_mid, ci_mat=ci_mat)

        """
        if ci_mat is None:
            # return a lambda function:
            return lambda ci_mat: np.rint(((rho - (ci_mat - ci_mid) * np.cos(phi)) / np.sin(phi)) + ri_mid).astype("int")
        else:
            # return the literal
            return np.rint(((rho - (ci_mat - ci_mid) * np.cos(phi)) / np.sin(phi)) + ri_mid).astype("int") 
    
    @classmethod
    def y_line(cls, phi, rho, t_mid, x_mid, t_mat=None):
        """
        
        y_line = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, t_mid=t_mid, x_mid=x_mid, t_mat=t_mat)
        """
        if t_mat is None:
            # return a lambda function:
            return lambda t: np.rint(((rho - (t - t_mid) * np.cos(phi)) / np.sin(phi)) + x_mid).astype("int")
        else:
            # return the literal
            return np.rint(((rho - (t_mat - t_mid) * np.cos(phi)) / np.sin(phi)) + x_mid).astype("int") 


        # y_line = ((rho_mat - (ci_mat - ci_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + ri_mid # (t_mat - ci_mid): makes it not matter whether absolute time bins or time bin indicies were used here:
        y_line = ((rho - (t_mat - t_mid) * np.cos(phi)) / np.sin(phi)) + x_mid
        return np.rint(y_line).astype("int") # (5000, 6) - (nlines, n_t)


    @classmethod
    def compute_score(cls, arr: NDArray, y_line_idxs: NDArray, nlines: int, n_neighbours:int):
        assert np.ndim(arr) >= 2
        n_pos, n_t = np.shape(arr)

        # using convolution to sum neighbours
        arr = np.apply_along_axis(
            np.convolve, axis=0, arr=arr, v=np.ones(2 * n_neighbours + 1), mode="same"
        )

        posterior = np.zeros((nlines, n_t)) # allocate output posterior

        # n_pos = np.shape(arr)[0]
        y_line_idxs = np.rint(y_line_idxs).astype("int")
        # if line falls outside of array in a given bin, replace that with median posterior value of that bin across all positions
        t_out = np.where((y_line_idxs < 0) | (y_line_idxs > (n_pos - 1)))
        t_in = np.where((y_line_idxs >= 0) & (y_line_idxs <= (n_pos - 1)))
        posterior[t_out] = np.median(arr[:, t_out[1]], axis=0)
        posterior[t_in] = arr[y_line_idxs[t_in], t_in[1]]

        # old_settings = np.seterr(all="ignore")
        posterior_mean = np.nanmean(posterior, axis=1)

        best_line_idx: int = np.argmax(posterior_mean)
        score = posterior_mean[best_line_idx]

        # np.seterr(**old_settings)
        return score, best_line_idx, (posterior, posterior_mean, y_line_idxs, (t_in, t_out))



@define(slots=False)
class RadonTransformDebugValue:
    t: NDArray = field()
    n_t: int = field()
    ci_mid: float = field()

    pos: NDArray = field()
    n_pos: int = field()
    ri_mid: float = field()

    diag_len: float = field()

    y_line_idxs: NDArray = field()
    y_line: NDArray = field() # these come back with all elements the same for a given line index? like [73, 73, 73, 73, 73, 73]
    t_out: NDArray = field()
    t_in: NDArray = field()

    posterior: NDArray = field()
    posterior_mean: NDArray = field()
    best_line_idx: int = field()
    best_phi: float = field()
    best_rho: float = field()
    
    ## real world
    time_mid: float = field()
    pos_mid: float = field()

    # @property
    # def n_t(self) -> int:
    #     return len(self.t)
    @property
    def ci(self) -> NDArray:
        """ ci: time indicies """
        return np.arange(self.n_t)

    # @property
    # def n_pos(self) -> int:
    #     return len(self.pos)
    @property
    def ri(self) -> NDArray:
        return np.arange(self.n_pos) # pos: position bin indicies

    @property
    def best_y_line(self) -> NDArray:
        """The best_y_line property."""
        return np.squeeze(self.y_line[self.best_line_idx, :])

    @property
    def best_y_line_idxs(self) -> NDArray:
        """The best_y_line property."""
        return np.squeeze(self.y_line_idxs[self.best_line_idx, :])


def radon_transform(arr: NDArray, nlines:int=10000, dt:float=1, dx:float=1, n_neighbours:int=1, enable_return_neighbors_arr=False, t0: Optional[float]=None, x0: Optional[float]=None):
    """Line fitting algorithm primarily used in decoding algorithm, a variant of radon transform, algorithm based on Kloosterman et al. 2012

    from neuropy.analyses.decoders import radon_transform
    
    Parameters
    ----------
    arr : 2d array
        time axis is represented by columns, position axis is represented by rows
    dt : float
        time binsize in seconds, only used for velocity/intercept calculation
    dx : float
        position binsize in cm, only used for velocity/intercept calculation
    n_neighbours : int,
        probability in each bin is replaced by sum of itself and these many 'neighbours' column wise, default 1 neighbour

    NOTE: when returning velocity the sign is flipped to match with position going from bottom to up

    Returns
    -------
    score:
        sum of values (posterior) under the best fit line
    velocity:
        speed of replay in cm/s
    intercept:
        intercept of best fit line

    References
    ----------
    1) Kloosterman et al. 2012
    """
    if t0 is None:
        t0 = 0.0

    if x0 is None:
        x0 = 0.0
        
    # if time_bin_centers is None:
    #     time_bin_centers = np.arange(arr.shape[1]) # index from [0, ... (NT-1)]
    # else:
    #     assert len(time_bin_centers) == np.shape(arr)[1]
    
    ci: NDArray = np.arange(arr.shape[1]) # ci: time indicies
    t: NDArray = (ci*float(dt)) + t0 # t: time bins in real seconds. When t0 is provided, these appear to be good.
    n_t: int = len(t)
    # ci_mid = (n_t + 1) / 2 - 1 # index space
    ci_mid: float = (float(n_t) / 2.0) # index space
    # time_mid = ((float(n_t) * dt) / 2.0) # real space
    time_mid: float = (t[-1] + t[0]) / 2.0 # real space

    # pos = np.arange(arr.shape[0]) # pos: position bin indicies
    ri: NDArray = np.arange(arr.shape[0]) # pos: position bin indicies
    pos: NDArray = (ri*float(dx)) + x0 # pos: position bin centers. When x0 is provided these perfeclty match `xbin_centers`
    n_pos: int = len(pos)
    # ri_mid = (n_pos + 1) / 2 - 1 # index space
    ri_mid: float = (float(n_pos) / 2.0) # index space
    # pos_mid = ((float(n_pos) * dx) / 2.0) # real space
    pos_mid: float = ((float(pos[-1]) + float(pos[0])) / 2.0) # real space

    diag_len: float = np.sqrt((n_t - 1) ** 2 + (n_pos - 1) ** 2)


    # exclude stationary events by choosing phi little below 90 degree
    # NOTE: angle of line is given by (90-phi), refer Kloosterman 2012
    phi = np.random.uniform(low=(-np.pi / 2), high=(np.pi / 2), size=nlines) # (nlines, )
    rho = np.random.uniform(low=-diag_len / 2, high=diag_len / 2, size=nlines) # (nlines, )

    rho_mat = np.tile(rho, (n_t, 1)).T
    phi_mat = np.tile(phi, (n_t, 1)).T
    
    ci_mat = np.tile(ci, (nlines, 1))
    t_mat = np.tile(t, (nlines, 1))

    # y_line = ((rho_mat - (ci_mat - ci_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + ri_mid # (t_mat - ci_mid): makes it not matter whether absolute time bins or time bin indicies were used here:
    # y_line_idxs = ((rho_mat - (ci_mat - ci_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + ri_mid
    # y_line_idxs = np.rint(y_line_idxs).astype("int")

    y_line_idxs = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, ci_mid=ci_mid, ri_mid=ri_mid, ci_mat=ci_mat) # (5000, 6) - (nlines, n_t) - note that the indicies returned can be actually outside the matrix bounds - e.g. negative (not python-wrapping index negative) or larger than the number of position bins
    y_line = RadonTransformComputation.y_line(phi=phi_mat, rho=rho_mat, t_mid=time_mid, x_mid=pos_mid, t_mat=t_mat) # seemingly incorrect

    # y_line = ((rho_mat - (t_mat - time_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + ri_mid ## 2024-05-07 - This seemed to be working, but it shouldn't be.

    # y_line = ((rho_mat - (t_mat - time_mid) * np.cos(phi_mat)) / np.sin(phi_mat)) + pos_mid
    # y_line = np.rint(y_line).astype("int") # (5000, 6) - (nlines, n_t)

    # old_settings = np.seterr(all="ignore")
    with np.errstate(all="ignore"):
        # posterior_mean = np.nanmean(posterior, axis=1)

        # best_line_idx: int = np.argmax(posterior_mean)
        # score = posterior_mean[best_line_idx]

        # score, best_line_idx, (posterior, posterior_mean, y_line, (t_in, t_out)) = RadonTransformComputation.compute_score(arr=arr, y_line=y_line, nlines=nlines, n_neighbours=n_neighbours)

        score, best_line_idx, (posterior, posterior_mean, y_line_idxs, (t_in, t_out)) = RadonTransformComputation.compute_score(arr=arr, y_line_idxs=y_line_idxs, nlines=nlines, n_neighbours=n_neighbours)
        best_phi = phi[best_line_idx]
        best_rho = rho[best_line_idx]
        best_y_line_idxs = np.squeeze(y_line_idxs[best_line_idx, :]) # (n_t, ) - confirmed to be correct
        # best_y_line = np.squeeze(y_line[best_line_idx, :]) # (n_t, ) # incorrect
        # converts to real world values

        ## Pho 2024-02-15 - Validated that below matches the original manuscript
        # velocity = RadonTransformComputation.velocity(phi=best_phi, time_bin_size=dt, pos_bin_size=dx)
        # intercept = RadonTransformComputation.intercept(phi=best_phi, rho=best_rho, t_mid=time_mid, x_mid=pos_mid, time_bin_size=dt, pos_bin_size=dx)

        ## Compute the correct intercept and velocity/slope from the debug line which seems to be correct:
        is_inside_matrix = np.logical_and((best_y_line_idxs >= 0), (best_y_line_idxs < n_pos))
        inside_matrix_only_best_y_line_idxs = best_y_line_idxs[is_inside_matrix]
        inside_matrix_only_t = t[is_inside_matrix]
        best_inside_y_line = np.array([pos[an_idx] for an_idx in inside_matrix_only_best_y_line_idxs])    
        velocity = (best_inside_y_line[-1]-best_inside_y_line[0])/(inside_matrix_only_t[-1]-inside_matrix_only_t[0]) # IndexError: index -1 is out of bounds for axis 0 with size 0 -- occuring with a (58, 1) array of all NaNs
        intercept = best_inside_y_line[0]-(velocity * inside_matrix_only_t[0])
        
        # best_y_line = np.array([pos[an_idx] for an_idx in best_y_line_idxs]) # (n_t, )

        # y_line = np.interp(t, xp=np.squeeze(inside_matrix_only_t), fp=np.squeeze(inside_matrix_only_best_y_line_idxs))


        # inside_only: (-48.92679149792471, 4450.167735614992)
        # best_y_line_segment = np.array([float(x0), (float(x0) + float(dx))])
        # t_segment = np.array([float(t0), float(t0)+float(dt)])
        # velocity = (best_y_line_segment[-1]-best_y_line_segment[0])/(t_segment[-1]-t_segment[0])
        # intercept = best_y_line_segment[0]-(velocity * t_segment[0]) # (19.027085582525963, -1566.9703125223657)

    # np.seterr(**old_settings)

    if enable_return_neighbors_arr:
        ## compute the real y_line for the debug value:
        y_line = (velocity * t) + intercept

        debug_info = RadonTransformDebugValue(t=t, n_t=n_t, ci_mid=ci_mid, time_mid=time_mid, 
            pos=pos, n_pos=n_pos, ri_mid=ri_mid, pos_mid=pos_mid,
            diag_len=diag_len, y_line_idxs=y_line_idxs, y_line=y_line, t_out=t_out, t_in=t_in, posterior=posterior, posterior_mean=posterior_mean,
            best_line_idx=best_line_idx, best_phi=best_phi, best_rho=best_rho,
         )
        return score, -velocity, intercept, (n_neighbours, arr.copy(), debug_info)
    else:
        return score, -velocity, intercept


def old_radon_transform(arr, nlines=5000):
    """Older version of the radon_transform that only returns the score and the slope (no intercept). Line fitting algorithm primarily used in decoding algorithm, a variant of radon transform, algorithm based on Kloosterman et al. 2012

    Parameters
    ----------
    arr : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    References
    ----------
    1) Kloosterman et al. 2012
    
    Usage:
        from neuropy.analyses.decoders import old_radon_transform

    """
    t = np.arange(arr.shape[1])
    nt = len(t)
    tmid = (nt + 1) / 2
    pos = np.arange(arr.shape[0])
    npos = len(pos)
    pmid = (npos + 1) / 2
    arr = np.apply_along_axis(np.convolve, axis=0, arr=arr, v=np.ones(3))

    theta = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=nlines)
    diag_len = np.sqrt((nt - 1) ** 2 + (npos - 1) ** 2)
    intercept = np.random.uniform(low=-diag_len / 2, high=diag_len / 2, size=nlines)

    cmat = np.tile(intercept, (nt, 1)).T
    mmat = np.tile(theta, (nt, 1)).T
    tmat = np.tile(t, (nlines, 1))
    posterior = np.zeros((nlines, nt))

    y_line = (((cmat - (tmat - tmid) * np.cos(mmat)) / np.sin(mmat)) + pmid).astype(int)
    t_out = np.where((y_line < 0) | (y_line > npos - 1))
    t_in = np.where((y_line >= 0) & (y_line <= npos - 1))
    posterior[t_out] = np.median(arr[:, t_out[1]], axis=0)
    posterior[t_in] = arr[y_line[t_in], t_in[1]]

    posterior_sum = np.nanmean(posterior, axis=1)
    max_line = np.argmax(posterior_sum)
    slope = -(1 / np.tan(theta[max_line]))

    return posterior_sum[max_line], slope


def wcorr(arr: NDArray) -> float:
    """weighted correlation
    Encountering issue when nx == 1, as in there is only one time bin, in which the wcorr doesn't make any sense.
    """
    # with warnings.catch_warnings():
    #     warnings.simplefilter("error", RuntimeWarning)

    nx, ny = arr.shape[1], arr.shape[0]
    y_mat: NDArray = np.tile(np.arange(ny)[:, np.newaxis], (1, nx))
    x_mat: NDArray = np.tile(np.arange(nx), (ny, 1))
    arr_sum: float = np.nansum(arr)
    ey: float = np.nansum(arr * y_mat) / arr_sum  # RuntimeWarning: invalid value encountered in double_scalars
    ex: float = np.nansum(arr * x_mat) / arr_sum  # RuntimeWarning: invalid value encountered in double_scalars
    cov_xy: float = np.nansum(arr * (y_mat - ey) * (x_mat - ex)) / arr_sum # RuntimeWarning: invalid value encountered in double_scalars
    cov_yy: float = np.nansum(arr * (y_mat - ey) ** 2) / arr_sum # RuntimeWarning: invalid value encountered in double_scalars
    cov_xx: float = np.nansum(arr * (x_mat - ex) ** 2) / arr_sum # RuntimeWarning: invalid value encountered in double_scalars

    return cov_xy / np.sqrt(cov_xx * cov_yy)


def jump_distance(posteriors, jump_stat="mean", norm=True):
    """Calculate jump distance for posterior matrices"""

    if jump_stat == "mean":
        f = np.mean
    elif jump_stat == "median":
        f = np.median
    elif jump_stat == "max":
        f = np.max
    else:
        raise ValueError("Invalid jump_stat. Valid values: mean, median, max")

    dx = 1 / posteriors[0].shape[0] if norm else 1
    jd = np.array([f(np.abs(np.diff(np.argmax(_, axis=0)))) for _ in posteriors])

    return jd * dx


def column_shift(arr, shifts=None):
    """Circular shift columns independently by a given amount"""

    assert arr.ndim == 2, "only 2d arrays accepted"

    if shifts is None:
        rng = np.random.default_rng()
        shifts = rng.integers(-arr.shape[0], arr.shape[0], arr.shape[1])

    assert arr.shape[1] == len(shifts)

    shifts = shifts % arr.shape[0]
    rows_indx, columns_indx = np.ogrid[: arr.shape[0], : arr.shape[1]]

    rows_indx = rows_indx - shifts[np.newaxis, :]

    return arr[rows_indx, columns_indx]


# @function_attributes(short_name=None, tags=['IMPROVED', 'FIXED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-10 15:03', related_items=[])
def epochs_spkcount(spikes: Union[pd.DataFrame, core.Neurons], epochs: Union[core.Epoch, pd.DataFrame], bin_size=0.01, export_time_bins:bool=False, included_neuron_ids=None, debug_print:bool=False, use_single_time_bin_per_epoch: bool=False) -> Tuple[List[NDArray[ND.Shape["N_ACLUS, N_TIME_BINS"], ND.Int]], NDArray[ND.Shape["N_ACLUS"], ND.Int], List[NDArray[ND.Shape['N_EPOCHS'], Any]], List[BinningContainer]]:
    """Binning events and calculating spike counts

    Args:
        spikes (Union[pd.DataFrame, core.Neurons]): _description_
        epochs (Union[core.Epoch, pd.DataFrame]): _description_
        bin_size (float, optional): _description_. Defaults to 0.01.
        export_time_bins (bool, optional): If True returns a list of the actual time bin centers for each epoch in time_bins. Defaults to False.
        included_neuron_ids (bool, optional): Only relevent if using a spikes_df for the neurons input. Ensures there is one spiketrain built for each neuron in included_neuron_ids, even if there are no spikes.
        debug_print (bool, optional): _description_. Defaults to False.
        use_single_time_bin_per_epoch (bool, optional): If True, a single time bin is used per epoch instead of using the provided `bin_size`. This means that each epoch will have exactly one bin, but it will be variablely-sized depending on the epoch's duration. Defaults to false.
        
    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        list: spkcount - one for each epoch in filter_epochs
        list: nbins - A count of the number of time bins that compose each decoding epoch e.g. nbins: [7 2 7 1 5 2 7 6 8 5 8 4 1 3 5 6 6 6 3 3 4 3 6 7 2 6 4 1 7 7 5 6 4 8 8 5 2 5 5 8]
        list: time_bin_containers_list - None unless export_time_bins is True. 
        
    Usage:
    
        spkcount, nbins, time_bin_containers_list = 
        
    
        
    Extra:
    
        If the epoch is shorter than the bin_size the time_bins returned should be the edges of the epoch
        
    2025-02-20 12:06 Gemni 2.0 Suggestion for fix: "The most robust solution is to always use the single-bin-per-epoch approach when the epoch is shorter than the bin size, even when use_single_time_bin_per_epoch is false. This ensures consistency and avoids the sliding_window_view issues.:"
    
    2025-03-10 Replacing the old epochs_spkcount (backed-up to `_OLD_epochs_spkcount`) with new, much simpler version
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
        from neuropy.analyses.decoders import epochs_spkcount
        
        spikes_df = get_proper_global_spikes_df(curr_active_pipeline)
        spkcount, included_neuron_ids, nbins, time_bin_containers_list = epochs_spkcount(spikes_df, epochs=filter_epochs, bin_size=decoding_time_bin_size, slideby=decoding_time_bin_size, export_time_bins=True, included_neuron_ids=neuron_IDs, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
    
        
    """
    from neuropy.core.epoch import ensure_dataframe
    from neuropy.utils.mixins.binning_helpers import compute_spanning_bins, BinningContainer, BinningInfo
    
    if use_single_time_bin_per_epoch:
        assert bin_size is None, f"use_single_time_bin_per_epoch is True but bin_size = {bin_size} has been provided. This bin_size will not be used as each epoch will be treated as a single time bin (meaning different epochs will have different length time bins). Set to None to continue 2025-03-10 15:01."

    if isinstance(spikes, core.Neurons):
        # spiketrains: NDArray = neurons.spiketrains
        spikes_df: pd.DataFrame = spikes.to_dataframe()
    elif isinstance(spikes, pd.DataFrame):
        # a spikes_df is passed in, build the spiketrains
        spikes_df: pd.DataFrame = spikes
    else:
        raise NotImplementedError

    if included_neuron_ids is None:
        unique_units: NDArray[ND.Shape["N_ACLUS"], ND.Int] = np.unique(spikes_df['aclu']) # sorted
        included_neuron_ids = unique_units

    spikes_df = spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids)

    # Handle either core.Epoch or pd.DataFrame objects:
    epoch_df: pd.DataFrame = ensure_dataframe(epochs)
    n_epochs: int = np.shape(epoch_df)[0] # there is one row per epoch

    spkcount: List[NDArray[ND.Shape["N_ACLUS, N_TIME_BINS"], ND.Int]] = []
    if export_time_bins:
        time_bin_containers_list = []
    else:
        time_bin_containers_list = None

    nbins: NDArray[ND.Shape['N_EPOCHS'], Any] = np.zeros(n_epochs, dtype="int")

    for i, epoch in enumerate(epoch_df.itertuples()):
        
        if use_single_time_bin_per_epoch:
            time_bin_edges = np.array([epoch.start, epoch.stop]) # two edges for the epoch
        else:
            ## Binning with Fixed Bin Sizes: fixed time-bin duration -> variable num time bins per epoch depending on epoch length
            time_bin_edges, time_bin_edges_binning_info = compute_spanning_bins(variable_values=None, bin_size=bin_size, variable_start_value=epoch.start, variable_end_value=epoch.stop) # fixed_step mode
        nbins[i] = (len(time_bin_edges) -1 ) ## #TODO 2025-03-10 14:57: - [ ] MAJOR: !!! is this supposed to be centers, or edges?!?
 
        unit_specific_time_binned_spike_counts, _included_neuron_ids = spikes_df.spikes.compute_unit_time_binned_spike_counts(time_bin_edges=time_bin_edges, included_neuron_ids=included_neuron_ids)
        
        spkcount.append(unit_specific_time_binned_spike_counts)
        
        if debug_print:
            print(f'i: {i}, epoch: [{epoch.start}, {epoch.stop}], bins: {np.shape(time_bin_edges)}, np.shape(unit_specific_time_binned_spike_counts): {np.shape(unit_specific_time_binned_spike_counts)}')

        if export_time_bins:
            if debug_print:
                print(f'nbins[i]: {nbins[i]}') # nbins: 20716

            bin_container = BinningContainer.init_from_edges(edges=time_bin_edges, edge_info=time_bin_edges_binning_info)
            assert len(bin_container.centers) == nbins[i], f"The length of the produced bin_container.centers and the nbins[i] should be the same, but len(bin_container.centers): {len(bin_container.centers)} and nbins[i]: {nbins[i]}!"
            time_bin_containers_list.append(bin_container)

    # END for i, epoch in enumerate(epoch_df.itertuples())
    return spkcount, included_neuron_ids, nbins, time_bin_containers_list # Tuple[List[NDArray[ND.Shape["N_ACLUS, N_TIME_BINS"], ND.Int]], NDArray[ND.Shape["N_ACLUS"], ND.Int], List[NDArray[ND.Shape['N_EPOCHS'], Any]], List[BinningContainer]]

def _OLD_epochs_spkcount(neurons: Union[core.Neurons, pd.DataFrame], epochs: Union[core.Epoch, pd.DataFrame], bin_size=0.01, slideby=None, export_time_bins:bool=False, included_neuron_ids=None, debug_print:bool=False, use_single_time_bin_per_epoch: bool=False):
    """Binning events and calculating spike counts

    Args:
        neurons (Union[core.Neurons, pd.DataFrame]): _description_
        epochs (Union[core.Epoch, pd.DataFrame]): _description_
        bin_size (float, optional): _description_. Defaults to 0.01.
        slideby (_type_, optional): _description_. Defaults to None.
        export_time_bins (bool, optional): If True returns a list of the actual time bin centers for each epoch in time_bins. Defaults to False.
        included_neuron_ids (bool, optional): Only relevent if using a spikes_df for the neurons input. Ensures there is one spiketrain built for each neuron in included_neuron_ids, even if there are no spikes.
        debug_print (bool, optional): _description_. Defaults to False.
        use_single_time_bin_per_epoch (bool, optional): If True, a single time bin is used per epoch instead of using the provided `bin_size`. This means that each epoch will have exactly one bin, but it will be variablely-sized depending on the epoch's duration. Defaults to false.
        
    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        list: spkcount - one for each epoch in filter_epochs
        list: nbins - A count of the number of time bins that compose each decoding epoch e.g. nbins: [7 2 7 1 5 2 7 6 8 5 8 4 1 3 5 6 6 6 3 3 4 3 6 7 2 6 4 1 7 7 5 6 4 8 8 5 2 5 5 8]
        list: time_bin_containers_list - None unless export_time_bins is True. 
        
    Usage:
    
        spkcount, nbins, time_bin_containers_list = 
        
    Extra:
    
        If the epoch is shorter than the bin_size the time_bins returned should be the edges of the epoch
        
    2025-02-20 12:06 Gemni 2.0 Suggestion for fix: "The most robust solution is to always use the single-bin-per-epoch approach when the epoch is shorter than the bin size, even when use_single_time_bin_per_epoch is false. This ensures consistency and avoids the sliding_window_view issues.:"
    
        
    """
    from neuropy.core.epoch import ensure_dataframe

    # Handle extracting the spiketrains, which are a list with one entry for each neuron and each list containing the timestamps of the spike event
    if isinstance(neurons, core.Neurons):
        if included_neuron_ids is None:
            included_neuron_ids = deepcopy(neurons.neuron_ids)

        spiketrains: NDArray = neurons.get_by_id(included_neuron_ids).spiketrains
    elif isinstance(neurons, pd.DataFrame):
        # a spikes_df is passed in, build the spiketrains
        spikes_df = neurons
        if included_neuron_ids is None:
            unique_units: NDArray[ND.Shape["N_ACLUS"], ND.Int] = np.unique(spikes_df['aclu']) # sorted
            included_neuron_ids = unique_units
        else:
            spikes_df = spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids)
        spiketrains: NDArray = spikes_df.spikes.get_unit_spiketrains(included_neuron_ids=included_neuron_ids)
    else:
        raise NotImplementedError

    assert included_neuron_ids is not None
    
    # Handle either core.Epoch or pd.DataFrame objects:
    epoch_df = ensure_dataframe(epochs)
    n_epochs = np.shape(epoch_df)[0] # there is one row per epoch

    spkcount = []
    if export_time_bins:
        time_bin_containers_list = []
    else:
        time_bin_containers_list = None

    nbins = np.zeros(n_epochs, dtype="int")

    if slideby is None:
        slideby = bin_size
    if debug_print:
        print(f'slideby: {slideby}')

    if not use_single_time_bin_per_epoch:
        window_shape  = int(bin_size * 1000) # Ah, forces integer binsizes!
        if debug_print:
            print(f'window_shape: {window_shape}')

    # ----- little faster but requires epochs to be non-overlapping ------
    # bins_epochs = []
    # for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
    #     bins = np.arange(epoch.start, epoch.stop, bin_size)
    #     nbins[i] = len(bins) - 1
    #     bins_epochs.extend(bins)
    # spkcount = np.asarray(
    #     [np.histogram(_, bins=bins_epochs)[0] for _ in spiketrains]
    # )

    # deleting unwanted columns that represent time between events
    # cumsum_nbins = np.cumsum(nbins)
    # del_columns = cumsum_nbins[:-1] + np.arange(len(cumsum_nbins) - 1)
    # spkcount = np.delete(spkcount, del_columns.astype(int), axis=1)

    for i, epoch in enumerate(epoch_df.itertuples()):
        #TODO 2024-01-25 16:52: - [ ] It seems that when the epoch duration is shorter than the bin size we should impose the same bins as the single-time-bin-per-epoch case, but idk what to do with the slideby.
        # Something like: if (use_single_time_bin_per_epoch or (window_shape > spkcount_.shape[1])): 
        if use_single_time_bin_per_epoch:
            bins = np.array([epoch.start, epoch.stop]) # NOTE: not subdivided
        else:
            # fixed time-bin duration -> variable num time bins per epoch depending on epoch length    
            # first dividing in 1ms
            bins = np.arange(epoch.start, epoch.stop, 0.001) # subdivided by 1ms, so way more bins here than we'd expect for either bin_centers or bin_edges
            
        spkcount_ = np.asarray(
            [np.histogram(_, bins=bins)[0] for _ in spiketrains]
        )
        if debug_print:
            print(f'i: {i}, epoch: [{epoch.start}, {epoch.stop}], bins: {np.shape(bins)}, np.shape(spkcount_): {np.shape(spkcount_)}')
        
        # the 2nd condition ((window_shape > spkcount_.shape[1])) prevents ValueError: window shape cannot be larger than input array shape spkcount_.shape: (80,60), window_shape: 75
        if (use_single_time_bin_per_epoch or (window_shape > spkcount_.shape[1])): 
            slide_view = spkcount_  # In this case, your spike count stays as it is
            nbins[i] = 1 # always 1 bin. #TODO 2024-01-19 04:45: - [ ] What is slide_view and do I need it?
        else:        
            slide_view = np.lib.stride_tricks.sliding_window_view(spkcount_, window_shape, axis=1)[:, :: int(slideby * 1000), :].sum(axis=2) # ValueError: window shape cannot be larger than input array shape spkcount_.shape: (80,60), window_shape: 75
            nbins[i] = slide_view.shape[1]
        
        if export_time_bins:
            if debug_print:
                print(f'nbins[i]: {nbins[i]}') # nbins: 20716
            
            # reduced_time_bins: only the FULL number of bin *edges*
            # reduced_time_bins # array([22.26, 22.36, 22.46, ..., 2093.66, 2093.76, 2093.86])
            if use_single_time_bin_per_epoch:
                # For single bin case, the bin edges are just the epoch start and stop times (which are importantly smaller than the time_bin_size)
                reduced_time_bin_edges = bins
                # And the bin center is just the middle of the epoch
                reduced_time_bin_centers = np.asarray([(epoch.start + epoch.stop) / 2])
                actual_window_size = float(epoch.stop - epoch.start) # the actual (variable) bin size
                assert len(reduced_time_bin_edges) >= 2, f"epochs_spkcount(...): epoch[{i}], nbins[{i}]: cannot build extents because len(reduced_time_bin_edges) < 2: reduced_time_bin_edges: {reduced_time_bin_edges}"
                manual_center_info = BinningInfo(variable_extents=(reduced_time_bin_edges[0], reduced_time_bin_edges[-1]), step=actual_window_size, num_bins=len(reduced_time_bin_centers)) # BinningInfo(variable_extents: tuple, step: float, num_bins: int)
                # center_info = BinningContainer.build_center_binning_info(reduced_time_bin_centers, reduced_time_bin_edges) # the second argument (edge_extents) is just the edges
                bin_container = BinningContainer(edges=reduced_time_bin_edges, centers=reduced_time_bin_centers, center_info=manual_center_info) # have to manually provide center_info because it doesn't work with two or less entries.
                
            else:
                reduced_slide_by_amount = int(slideby * 1000)
                reduced_time_bin_edges = bins[:: reduced_slide_by_amount] # WTH does this notation mean?
                
                # assert len(reduced_time_bin_edges) >= 2, f"epochs_spkcount(...): epoch[{i}], nbins[{i}]: cannot build extents because len(reduced_time_bin_edges) < 2: reduced_time_bin_edges: {reduced_time_bin_edges}"
                assert len(reduced_time_bin_edges) > 0, f"epochs_spkcount(...): epoch[{i}], nbins[{i}]: cannot build extents because reduced_time_bin_edges is empty (len(reduced_time_bin_edges) == 0): reduced_time_bin_edges: {reduced_time_bin_edges}"
                
                try:
                    n_reduced_edges: int = len(reduced_time_bin_edges)
                    if n_reduced_edges == 1:
                        # Built using `epoch` - have to manually build center_info from subsampled `bins` because it doesn't work with two or less entries.
                        print(f'ERROR: epochs_spkcount(...): {i}/{n_epochs}, epoch[{i}], nbins[{i}]: {nbins[i]} - TODO 2024-08-07 19:11: Building BinningContainer for epoch with fewer than 2 edges (occurs when epoch duration is shorter than the bin size). Using the epoch.start, epoch.stop as the two edges (giving a single bin) but this might be off and cause problems, as they are the edges of the epoch but maybe not "real" edges?')
                        reduced_time_bin_edges = np.array([epoch.start, epoch.stop])
                        # reduced_time_bin_edges = deepcopy(bins) #TODO 2024-08-07 19:11: - [ ] This might be off, as they are the edges of the epoch but maybe not "real" edges?
                        reduced_time_bin_centers = np.asarray([(epoch.start + epoch.stop) / 2]) # And the bin center is just the middle of the epoch
                        actual_window_size = float(epoch.stop - epoch.start) # the actual (variable) bin size
                        variable_extents = (epoch.start, epoch.stop)
                        manual_center_info = BinningInfo(variable_extents=variable_extents, step=actual_window_size, num_bins=1) # num_bins == 1, just like when (len(reduced_time_bin_edges) == 2)
                        bin_container = BinningContainer(edges=reduced_time_bin_edges, centers=reduced_time_bin_centers, center_info=manual_center_info) # have to manually provide center_info because it doesn't work with two or less entries.   
                        print(f'\t ERROR (cont.): even after this hack `slide_view` is not updated, so the returned spkcount is not valid and has the old (wrong, way too many) number of bins. This results in decoded posteriors/postitions/etc with way too many bins downstream. see `SOLUTION 2024-08-07 20:08: - [ ] Recompute the Invalid Quantities with the known correct number of time bins` for info.')                     
                    elif n_reduced_edges == 2:
                        # have to manually build center_info from subsampled `bins` because it doesn't work with two or less entries.
                        reduced_time_bin_edges = deepcopy(bins)
                        # And the bin center is just the middle of the epoch
                        reduced_time_bin_centers = np.asarray([(reduced_time_bin_edges[0] + reduced_time_bin_edges[1]) / 2]) # just a single element?
                        actual_window_size = float(reduced_time_bin_edges[1] - reduced_time_bin_edges[0]) # the actual (variable) bin size... #TODO 2024-08-07 18:50: - [ ] this might be the subsampled bin size
                        manual_center_info = BinningInfo(variable_extents=(reduced_time_bin_edges[0], reduced_time_bin_edges[-1]), step=actual_window_size, num_bins=len(reduced_time_bin_centers))
                        bin_container = BinningContainer(edges=reduced_time_bin_edges, centers=reduced_time_bin_centers, center_info=manual_center_info) # have to manually provide center_info because it doesn't work with two or less entries.
                    else:
                        # can do it like normal:
                        ## automatically computes reduced_time_bin_centers and both infos:
                        bin_container = BinningContainer(edges=reduced_time_bin_edges)
                        reduced_time_bin_centers = deepcopy(bin_container.centers)                 

                except Exception as err:
                    print(f'ERROR: epochs_spkcount(...): {i}/{n_epochs}, epoch[{i}], nbins[{i}]: while building time bins, encountered exception err: {err}.')
                    raise err                
            
            if debug_print:
                num_bad_time_bins = len(bins)
                print(f'num_bad_time_bins: {num_bad_time_bins}')
                if not use_single_time_bin_per_epoch:
                    print(f'reduced_slide_by_amount: {reduced_slide_by_amount}')
                print(f'reduced_time_bin_edges.shape: {reduced_time_bin_edges.shape}') # reduced_time_bin_edges.shape: (20717,)
                print(f'reduced_time_bin_centers.shape: {reduced_time_bin_centers.shape}') # reduced_time_bin_centers.shape: (20716,)

            assert len(reduced_time_bin_centers) == nbins[i], f"The length of the produced reduced_time_bin_centers and the nbins[i] should be the same, but len(reduced_time_bin_centers): {len(reduced_time_bin_centers)} and nbins[i]: {nbins[i]}!"
            # time_bin_centers_list.append(reduced_time_bin_centers)
            time_bin_containers_list.append(bin_container)
            
        spkcount.append(slide_view)

    return spkcount, included_neuron_ids, nbins, time_bin_containers_list # Tuple[List[NDArray[ND.Shape["N_ACLUS, N_TIME_BINS"], ND.Int]], NDArray[ND.Shape["N_ACLUS"], ND.Int], List[NDArray[ND.Shape['N_EPOCHS'], Any]], List[BinningContainer]]



# ==================================================================================================================== #
# Evaluate the differences between `epochs_spkcount` and `_OLD_epochs_spkcount`                                        #
# ==================================================================================================================== #

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from neuropy.analyses.decoders import epochs_spkcount, _OLD_epochs_spkcount
from neuropy.utils.mixins.binning_helpers import BinningContainer

def compare_epochs_spkcount_implementations(spikes_df: pd.DataFrame, epochs: pd.DataFrame, bin_size: float = 0.01, export_time_bins: bool = True, included_neuron_ids = None, use_single_time_bin_per_epoch: bool = False, debug_print: bool = False) -> Dict[str, Any]:
    """
    Evaluates both the new and old epochs_spkcount implementations with identical parameters
    and compares their outputs to identify any differences.
    
    Parameters:
    -----------
    spikes_df : pd.DataFrame
        The dataframe containing spike data
    epochs : pd.DataFrame
        The dataframe containing epoch data
    bin_size : float, optional
        Size of time bins for analysis, defaults to 0.01
    export_time_bins : bool, optional
        Whether to export time bin information, defaults to True
    included_neuron_ids : array-like, optional
        Specific neuron IDs to include, defaults to None (all neurons)
    use_single_time_bin_per_epoch : bool, optional
        If True, uses a single time bin per epoch, defaults to False
    debug_print : bool, optional
        Whether to print debug information, defaults to False
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing comparison results
    """
    # Run new implementation
    print("Running new implementation...")
    new_spkcount, new_included_neuron_ids, new_nbins, new_time_bin_containers_list = epochs_spkcount(
        spikes_df, 
        epochs=epochs, 
        bin_size=bin_size if not use_single_time_bin_per_epoch else None, 
        export_time_bins=export_time_bins, 
        included_neuron_ids=included_neuron_ids,
        use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, 
        debug_print=debug_print
    )
    
    # Run old implementation
    print("Running old implementation...")
    old_spkcount, old_included_neuron_ids, old_nbins, old_time_bin_containers_list = _OLD_epochs_spkcount(
        spikes_df, 
        epochs=epochs, 
        bin_size=bin_size, 
        slideby=bin_size,  # Set slideby equal to bin_size as in the example
        export_time_bins=export_time_bins, 
        included_neuron_ids=included_neuron_ids,
        use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, 
        debug_print=debug_print
    )
    
    old_outputs = (old_spkcount, old_included_neuron_ids, old_nbins, old_time_bin_containers_list)
    new_outputs = (new_spkcount, new_included_neuron_ids, new_nbins, new_time_bin_containers_list)
    # raw_outputs_tuple = (new_outputs,
    #                old_outputs)
    
        # Prepare results dictionary
    results = {
        "same_neuron_ids": np.array_equal(new_included_neuron_ids, old_included_neuron_ids),
        "nbins_comparison": {
            "match": np.array_equal(new_nbins, old_nbins),
            "new_nbins": new_nbins,
            "old_nbins": old_nbins,
            "difference_count": np.sum(new_nbins != old_nbins),
            "mean_difference": np.mean(np.abs(new_nbins - old_nbins)) if len(new_nbins) == len(old_nbins) else None
        },
        "epoch_count": len(epochs),
        "epoch_differences": []
    }
    
    # Compare spkcount arrays (one per epoch)
    n_epochs = min(len(new_spkcount), len(old_spkcount))
    
    results["spkcount_arrays"] = {
        "length_match": len(new_spkcount) == len(old_spkcount),
        "new_length": len(new_spkcount),
        "old_length": len(old_spkcount)
    }
    
    # Compare each epoch's spike counts
    for i in range(n_epochs):
        new_shape = new_spkcount[i].shape
        old_shape = old_spkcount[i].shape
        
        epoch_diff = {
            "epoch_index": i,
            "shapes_match": new_shape == old_shape,
            "new_shape": new_shape,
            "old_shape": old_shape,
            "values_match": False  # Default to False, will set to True if applicable
        }
        
        if new_shape == old_shape:
            # Check if the actual values match
            is_equal = np.array_equal(new_spkcount[i], old_spkcount[i])
            epoch_diff["values_match"] = is_equal
            
            if not is_equal:
                # Calculate statistics of differences
                diff = new_spkcount[i] - old_spkcount[i]
                epoch_diff["max_diff"] = np.max(np.abs(diff))
                epoch_diff["mean_diff"] = np.mean(np.abs(diff))
                epoch_diff["nonzero_diff_count"] = np.count_nonzero(diff)
                epoch_diff["nonzero_diff_percentage"] = 100 * np.count_nonzero(diff) / diff.size
                epoch_diff["total_count_diff"] = np.sum(new_spkcount[i]) - np.sum(old_spkcount[i])
        else:
            # Different shapes means different binning approach or different count
            epoch_diff["new_total_count"] = np.sum(new_spkcount[i])
            epoch_diff["old_total_count"] = np.sum(old_spkcount[i])
            epoch_diff["total_count_diff"] = epoch_diff["new_total_count"] - epoch_diff["old_total_count"]
            
        results["epoch_differences"].append(epoch_diff)
    
    # Compare time bin containers if they exist
    if export_time_bins:
        bin_container_diffs = []
        
        n_containers = min(len(new_time_bin_containers_list), len(old_time_bin_containers_list))
        results["bin_containers"] = {
            "length_match": len(new_time_bin_containers_list) == len(old_time_bin_containers_list),
            "new_length": len(new_time_bin_containers_list),
            "old_length": len(old_time_bin_containers_list)
        }
        
        for i in range(n_containers):
            new_container = new_time_bin_containers_list[i]
            old_container = old_time_bin_containers_list[i]
            
            container_diff = {
                "epoch_index": i,
                "edges_match": False,
                "centers_match": False,
                "new_edges_len": len(new_container.edges),
                "old_edges_len": len(old_container.edges),
                "new_centers_len": len(new_container.centers),
                "old_centers_len": len(old_container.centers),
                "new_step": new_container.center_info.step,
                "old_step": old_container.center_info.step,
                "steps_match": np.isclose(new_container.center_info.step, old_container.center_info.step)
            }
            
            # Check if edges match
            if len(new_container.edges) == len(old_container.edges):
                container_diff["edges_match"] = np.allclose(new_container.edges, old_container.edges)
                if not container_diff["edges_match"]:
                    container_diff["max_edge_diff"] = np.max(np.abs(new_container.edges - old_container.edges))
            
            # Check if centers match
            if len(new_container.centers) == len(old_container.centers):
                container_diff["centers_match"] = np.allclose(new_container.centers, old_container.centers)
                if not container_diff["centers_match"]:
                    container_diff["max_center_diff"] = np.max(np.abs(new_container.centers - old_container.centers))
            
            bin_container_diffs.append(container_diff)
                
        results["bin_container_differences"] = bin_container_diffs
    
    # Generate summary statistics
    matching_epoch_counts = sum(1 for d in results["epoch_differences"] if d["values_match"])
    results["summary_stats"] = {
        "matching_epoch_counts": matching_epoch_counts,
        "matching_epoch_percentage": 100 * matching_epoch_counts / len(results["epoch_differences"]) if results["epoch_differences"] else 0
    }
    
    if export_time_bins:
        matching_containers = sum(1 for d in results["bin_container_differences"] if d["edges_match"] and d["centers_match"])
        results["summary_stats"]["matching_containers"] = matching_containers
        results["summary_stats"]["matching_containers_percentage"] = 100 * matching_containers / len(results["bin_container_differences"]) if results["bin_container_differences"] else 0
    
    # Print summary
    print("\nCOMPARISON SUMMARY:")
    print(f"  Neuron IDs match: {results['same_neuron_ids']}")
    
    if results["nbins_comparison"]["match"]:
        print(f"  Bin counts match for all {len(new_nbins)} epochs")
    else:
        print(f"  Bin counts differ in {results['nbins_comparison']['difference_count']} out of {len(new_nbins)} epochs")
        if results["nbins_comparison"]["mean_difference"] is not None:
            print(f"  Average bin count difference: {results['nbins_comparison']['mean_difference']:.2f}")
    
    print(f"  Epochs with identical spike counts: {matching_epoch_counts} out of {len(results['epoch_differences'])}")
    
    if export_time_bins:
        print(f"  Epochs with identical time bins: {matching_containers} out of {len(results['bin_container_differences'])}")
    
    # Call the detailed differences function
    print_detailed_epoch_differences(results)
        
    return results, new_outputs, old_outputs


def print_detailed_epoch_differences(results: Dict[str, Any], max_items: int = 5):
    """
    Prints detailed information about differences between epoch results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results dictionary from compare_epochs_spkcount_implementations
    max_items : int, optional
        Maximum number of items to show in each section, defaults to 5
    """
    print("\nDETAILED DIFFERENCES:")
    
    # Bin count differences
    if not results["nbins_comparison"]["match"]:
        print("\n  BIN COUNT DIFFERENCES:")
        new_nbins = results["nbins_comparison"]["new_nbins"]
        old_nbins = results["nbins_comparison"]["old_nbins"]
        
        diff_indices = np.where(new_nbins != old_nbins)[0]
        for i, idx in enumerate(diff_indices[:max_items]):
            print(f"    Epoch {idx}: New: {new_nbins[idx]}, Old: {old_nbins[idx]}, Diff: {new_nbins[idx] - old_nbins[idx]}")
        
        if len(diff_indices) > max_items:
            print(f"    ... and {len(diff_indices) - max_items} more differences")
    
    # Shape differences
    shape_mismatches = [d for d in results["epoch_differences"] if not d["shapes_match"]]
    if shape_mismatches:
        print("\n  SHAPE DIFFERENCES:")
        for i, diff in enumerate(shape_mismatches[:max_items]):
            print(f"    Epoch {diff['epoch_index']}: New shape: {diff['new_shape']}, Old shape: {diff['old_shape']}")
            if "new_total_count" in diff:
                print(f"      Total spike counts - New: {diff['new_total_count']}, Old: {diff['old_total_count']}, Diff: {diff['total_count_diff']}")
        
        if len(shape_mismatches) > max_items:
            print(f"    ... and {len(shape_mismatches) - max_items} more epochs with shape differences")
    
    # Value differences (same shape but different values)
    value_diffs = [d for d in results["epoch_differences"] if d["shapes_match"] and not d["values_match"]]
    if value_diffs:
        print("\n  VALUE DIFFERENCES (same shapes but different counts):")
        
        # Sort by maximum difference
        if value_diffs and "max_diff" in value_diffs[0]:
            value_diffs.sort(key=lambda x: x.get("max_diff", 0), reverse=True)
        
        for i, diff in enumerate(value_diffs[:max_items]):
            print(f"    Epoch {diff['epoch_index']} (shape {diff['new_shape']}):")
            print(f"      Max difference: {diff['max_diff']}")
            print(f"      Mean difference: {diff['mean_diff']:.4f}")
            print(f"      Cells with differences: {diff['nonzero_diff_count']} ({diff['nonzero_diff_percentage']:.2f}%)")
            print(f"      Total count difference: {diff['total_count_diff']}")
        
        if len(value_diffs) > max_items:
            print(f"    ... and {len(value_diffs) - max_items} more epochs with value differences")
    
    # Bin container differences
    if "bin_container_differences" in results:
        edge_diffs = [d for d in results["bin_container_differences"] if not d["edges_match"]]
        center_diffs = [d for d in results["bin_container_differences"] if not d["centers_match"]]
        
        if edge_diffs or center_diffs:
            print("\n  TIME BIN DIFFERENCES:")
            
            if edge_diffs:
                print("\n    EDGE DIFFERENCES:")