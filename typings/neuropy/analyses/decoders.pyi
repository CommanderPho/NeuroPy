"""
This type stub file was generated by pyright.
"""

import pandas as pd
from typing import Optional, Union
from nptyping import NDArray
from neuropy.analyses.placefields import PfND
from neuropy import core
from attrs import define

class RadonTransformComputation:
    """
    Helpers for radon transform:

    from neuropy.analyses.decoders import RadonTransformComputation

    """
    @classmethod
    def phi(cls, velocity, time_bin_size, pos_bin_size): # -> Any:
        ...
    
    @classmethod
    def rho(cls, icpt, t_mid, x_mid, velocity, time_bin_size, pos_bin_size):
        ...
    
    @classmethod
    def velocity(cls, phi, time_bin_size, pos_bin_size):
        ...
    
    @classmethod
    def intercept(cls, phi, rho, t_mid, x_mid, time_bin_size, pos_bin_size):
        """
            t_mid, x_mid: the continuous-time versions
        """
        ...
    
    @classmethod
    def y_line_idxs(cls, phi, rho, ci_mid, ri_mid, ci_mat=...): # -> Callable[..., Any] | Any:
        """ 
        returns a lambda function that takes `ci_mat`
        y_line_idxs_fn = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, ci_mid=ci_mid, ri_mid=ri_mid)
        y_line_idxs = y_line_idxs_fn(ci_mat=ci_mat)

        ---

        y_line_idxs = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, ci_mid=ci_mid, ri_mid=ri_mid, ci_mat=ci_mat)

        """
        ...
    
    @classmethod
    def y_line(cls, phi, rho, t_mid, x_mid, t_mat=...): # -> Callable[..., Any] | Any:
        """
        
        y_line = RadonTransformComputation.y_line_idxs(phi=phi_mat, rho=rho_mat, t_mid=t_mid, x_mid=x_mid, t_mat=t_mat)
        """
        ...
    
    @classmethod
    def compute_score(cls, arr: NDArray, y_line_idxs: NDArray, nlines: int, n_neighbours: int): # -> tuple[Any, int, tuple[NDArray[float64], Any, NDArray[Any], tuple[tuple[NDArray[intp], ...], tuple[NDArray[intp], ...]]]]:
        ...
    


@define(slots=False)
class RadonTransformDebugValue:
    t: NDArray = ...
    n_t: int = ...
    ci_mid: float = ...
    pos: NDArray = ...
    n_pos: int = ...
    ri_mid: float = ...
    y_line: NDArray = ...
    t_out: NDArray = ...
    t_in: NDArray = ...
    posterior_mean: NDArray = ...
    best_line_idx: int = ...
    best_phi: float = ...
    best_rho: float = ...
    time_mid: float = ...
    pos_mid: float = ...
    @property
    def best_y_line(self) -> NDArray:
        """The best_y_line property."""
        ...
    


def radon_transform(arr: NDArray, nlines: int = ..., dt: float = ..., dx: float = ..., n_neighbours: int = ..., enable_return_neighbors_arr=..., t0: Optional[float] = ..., x0: Optional[float] = ...): # -> tuple[Any, Any, Any, tuple[int, NDArray, RadonTransformDebugValue]] | tuple[Any, Any, Any]:
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
    ...

def old_radon_transform(arr, nlines=...): # -> tuple[Any, Any]:
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
    ...

def wcorr(arr):
    """weighted correlation
    Encountering issue when nx == 1, as in there is only one time bin, in which the wcorr doesn't make any sense.
    """
    ...

def jump_distance(posteriors, jump_stat=..., norm=...): # -> NDArray[floating[Any]]:
    """Calculate jump distance for posterior matrices"""
    ...

def column_shift(arr, shifts=...):
    """Circular shift columns independently by a given amount"""
    ...

def epochs_spkcount(neurons: Union[core.Neurons, pd.DataFrame], epochs: Union[core.Epoch, pd.DataFrame], bin_size=..., slideby=..., export_time_bins: bool = ..., included_neuron_ids=..., debug_print: bool = ..., use_single_time_bin_per_epoch: bool = ...): # -> tuple[list[Any], NDArray[Any], list[Any] | None]:
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
        
        
    """
    ...

class Decode1d:
    n_jobs = ...
    def __init__(self, neurons: core.Neurons, ratemap: core.Ratemap, epochs: core.Epoch = ..., time_bin_size=..., slideby=...) -> None:
        ...
    
    def calculate_shuffle_score(self, n_iter=..., method=...): # -> None:
        """Shuffling and decoding epochs"""
        ...
    
    def score_posterior(self, p): # -> tuple[NDArray[Any], NDArray[Any]]:
        """Scoring of epochs

        Returns
        -------
        [type]
            [description]

        References
        ----------
        1) Kloosterman et al. 2012
        """
        ...
    
    @property
    def p_value(self): # -> Any:
        ...
    
    def plot_in_bokeh(self): # -> None:
        ...
    
    def plot_replay_epochs(self, pval=..., speed_thresh=..., cmap=...): # -> None:
        ...
    


class Decode2d:
    """ 2D Decoder 
    
    """
    def __init__(self, pf2d_obj: PfND) -> None:
        ...
    
    def estimate_behavior(self, spikes_df, t_start_end, time_bin_size=..., smooth=..., plot=...): # -> None:
        """ 
        Updates:
            ._all_positions_matrix
            ._original_data_shape
            ._flat_all_positions_matrix
            .bin_size
            .decodingtime
            .time_bin_centers
            .actualbin
            .posterior
            .actualpos
            .decodedPos
        """
        ...
    
    def decode_events(self, binsize=..., slideby=...): # -> tuple[Any, Any]:
        """Decodes position within events which are set using self.events

        Parameters
        ----------
        binsize : float, seconds, optional
            size of binning withing each events, by default 0.02
        slideby : float, seconds optional
            sliding by this much, by default 0.005

        Returns
        -------
        [type]
            [description]
        """
        ...
    
    def plot(self): # -> None:
        ...
    


