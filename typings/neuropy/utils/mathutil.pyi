"""
This type stub file was generated by pyright.
"""

def min_max_scaler(x, axis=...):
    """Scales the values x to lie between 0 and 1 along the specfied axis

    Parameters
    ----------
    x : np.array
        numpy ndarray

    Returns
    -------
    np.array
        scaled array


    ERRORS:
        2023-03-02 - ValueError: zero-size array to reduction operation minimum which has no identity
            Occurs when np.shape(x) is (0,)


    """
    ...

def bounded(v: float, vmin: float = ..., vmax: float = ...) -> float:
    """Returns the value bounded between two optional lower and upper bounds.
    It clips to the bounds.

    Usage:
        from neuropy.utils.mathutil import bounded

        v_bounded = bounded(value, vmin=0.0, vmax=1.0)
    
    Usage 2:
        value = [-150, -65, -0.9, 0, 0.9, 65, 150]

        bounded(value, vmin=0.0, vmax=1.0) # array([0. , 0. , 0. , 0. , 0.9, 1. , 1. ])
        bounded(value, vmin=-1.0, vmax=1.0) # array([-1. , -1. , -0.9,  0. ,  0.9,  1. ,  1. ])

        
    """
    ...

def map_to_fixed_range(lin_pos, x_min: float = ..., x_max: float = ...):
    ...

def map_value(value, from_range: tuple[float, float], to_range: tuple[float, float]):
    """ Maps values from a range `from_low_high_tuple`: (a, b) to a new range `to_low_high_tuple`: (A, B). Similar to arduino's `map(value, fromLow, fromHigh, toLow, toHigh)` function
    
    Usage:
        from neuropy.utils.mathutil import map_value
        track_change_mapped_idx = map_value(track_change_time, (Flat_epoch_time_bins_mean[0], Flat_epoch_time_bins_mean[-1]), (0, (num_epochs-1)))
        track_change_mapped_idx
        
    Example 2: Defining Shortcut mapping
        map_value_time_to_epoch_idx_space = lambda v: map_value(v, (Flat_epoch_time_bins_mean[0], Flat_epoch_time_bins_mean[-1]), (0, (num_epochs-1))) # same map

    """
    ...

def compute_grid_bin_bounds(*args): # -> tuple[Any, ...]:
    """ computes the (min, max) bound for each passed array and returns a tuple of these (min, max) tuples. 
    from neuropy.utils.mathutil import compute_grid_bin_bounds
    
    grid_bin_bounds: `((x_min, x_max), (y_min, y_max), ...)
    
    """
    ...

def cdf(x, bins): # -> NDArray[Any]:
    """Returns cummulative distribution for x at bins"""
    ...

def partialcorr(x, y, z): # -> Any:
    """
    correlation between x and y , with controlling for z
    """
    ...

def parcorr_mult(x, y, z): # -> tuple[NDArray[float64], NDArray[float64]]:
    """
    correlation between multidimensional x and y , with controlling for multidimensional z

    """
    ...

def parcorr_muglt(x, y, z): # -> tuple[NDArray[float64], NDArray[float64]]:
    """
    correlation between multidimensional x and y , with controlling for multidimensional z

    """
    ...

def getICA_Assembly(x):
    """extracting statisticaly independent components from significant eigenvectors as detected using Marcenko-Pasteur distributionvinput = Matrix  (m x n) where 'm' are the number of cells and 'n' time bins ICA weights thus extracted have highiest weight positive (as done in Gido M. van de Ven et al. 2016) V = ICA weights for each neuron in the coactivation (weight having the highiest value is kept positive) M1 =  originally extracted neuron weights

    Arguments:
        x {[ndarray]} -- [an array of size n * m]

    Returns:
        [type] -- [Independent assemblies]
    """
    ...

def threshPeriods(sig, lowthresh=..., highthresh=..., minDistance=..., minDuration=...): # -> NDArray[Any]:
    ...

def contiguous_regions(condition): # -> Any | NDArray[signedinteger[_NBitIntP]]:
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index. Taken directly from stackoverflow:
    https://stackoverflow.com/questions/4494404/find-large-number-of-
    consecutive-values-fulfilling-condition-in-a-numpy-array
        
    Usage:
        from neuropy.utils.mathutil import contiguous_regions
        idnan = mathutil.contiguous_regions(np.isnan(x))  # identify missing data points

            for ids in idnan:
                missing_ids = range(ids[0], ids[-1])
                bracket_ids = ids + [-1, 0]
                xgood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], x[bracket_ids])
                ygood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], y[bracket_ids])
                zgood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], z[bracket_ids])

    """
    ...

def hmmfit1d(Data, n_comp=..., n_iter=...): # -> NDArray[Any] | None:
    ...

def eventpsth(ref, event, fs, quantparam=..., binsize=..., window=..., nQuantiles=...): # -> ndarray[Any, dtype[Any]] | ndarray[Any, dtype[signedinteger[_32Bit]]]:
    """psth of 'event' with respect to 'ref'

    Parameters
    ----------
    ref (array):
        1-D array of timings of reference event in seconds
    event (1D array):
        timings of events whose psth will be calculated
    fs:
        sampling rate
    quantparam (1D array):
        values used to divide 'ref' into quantiles
    binsize (float, optional):
        [description]. Defaults to 0.01.
    window (int, optional):
        [description]. Defaults to 1.
    nQuantiles (int, optional):
        [description]. Defaults to 10.

    Returns
    -------
        [type]: [description]
    """
    ...

