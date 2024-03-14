"""
This type stub file was generated by pyright.
"""

from .. import core

"""
This type stub file was generated by pyright.
"""
def plot_raster(neurons: core.Neurons, ax=..., sort_by_frate=..., color=..., marker=..., markersize=..., add_vert_jitter=...):
    """creates raster plot using spiktrains in neurons

    Parameters
    ----------
    neurons : list, optional
        Each array within list represents spike times of that unit, by default None
    ax : obj, optional
        axis to plot onto, by default None
    sort_by_frate : bool, optional
        If true then sorts spikes by the number of spikes (frate), by default False
    color : [type], optional
        color for raster plots, by default None
    marker : str, optional
        marker style, by default "|"
    markersize : int, optional
        size of marker, by default 2
    add_vert_jitter: boolean, optional
        adds vertical jitter to help visualize super dense spiking, not standardly used for rasters...
    """
    ...

def plot_mua(mua: core.Mua, ax=..., **kwargs):
    ...

def plot_ccg(self, clus_use, type=..., bin_size=..., window_size=..., ax=...):
    """Plot CCG for clusters in clus_use (list, max length = 2). Supply only one cluster in clus_use for ACG only.
    type: 'all' or 'ccg_only'.
    ax (optional): if supplied len(ax) must be 1 for type='ccg_only' or nclus^2 for type 'all'"""
    ...

def plot_firing_rate(neurons: core.Neurons, bin_size=..., stacked=..., normalize=..., sortby=..., cmap=...):
    ...

def plot_waveforms(neurons: core.Neurons, sort_order=..., color=...):
    """Plot waveforms in the neurons object

    Parameters
    ----------
    neurons : core.Neurons
        [description]
    sort_order : array, optional
        sorting order for the neurons, by default None
    color : str, optional
        [description], by default "#afadac"

    Returns
    -------
    ax
    """
    ...

