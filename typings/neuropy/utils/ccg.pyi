"""
This type stub file was generated by pyright.
"""

"""Cross-correlograms."""
_ACCEPTED_ARRAY_DTYPES = ...
def firing_rate(spike_clusters, cluster_ids=..., bin_size=..., duration=...):
    """Compute the average number of spikes per cluster per bin."""
    ...

def correlograms(spike_times, spike_clusters, cluster_ids=..., sample_rate=..., bin_size=..., window_size=..., symmetrize=...):
    """Compute all pairwise cross-correlograms among the clusters appearing
    in `spike_clusters`.
    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    spike_clusters : array-like
        Spike-cluster mapping.
    cluster_ids : array-like
        The list of *all* unique clusters, in any order. That order will be used
        in the output array.
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    sample_rate : float
        Sampling rate.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.
    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """
    ...

