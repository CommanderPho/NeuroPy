from ipywidgets import widgets
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .. import core
from neuropy.utils import mathutil
from .figure import Fig

## TODO: refactor plot_ratemap_1D and plot_ratemap_2D to a single flat function (if that's appropriate).
## TODO: refactor plot_ratemap_1D and plot_ratemap_2D to take a **kwargs and apply optional defaults (find previous code where I did that using the | and dict conversion. In my 3D code.

def plot_ratemap_1D(ratemap: core.Ratemap,
    normalize_xbin=False,
    ax=None,
    pad=2,
    normalize_tuning_curve=False,
    sortby=None,
    cmap="tab20b"):
    """Plot 1D place fields stacked

    Parameters
    ----------
    ax : [type], optional
        [description], by default None
    speed_thresh : bool, optional
        [description], by default False
    pad : int, optional
        [description], by default 2
    normalize : bool, optional
        [description], by default False
    sortby : bool, optional
        [description], by default True
    cmap : str, optional
        [description], by default "tab20b"

    Returns
    -------
    [type]
        [description]
    """
    cmap = mpl.cm.get_cmap(cmap)

    tuning_curves = ratemap.tuning_curves
    n_neurons = ratemap.n_neurons
    bin_cntr = ratemap.xbin_centers
    if normalize_xbin:
        bin_cntr = (bin_cntr - np.min(bin_cntr)) / np.ptp(bin_cntr)

    if ax is None:
        _, gs = Fig().draw(grid=(1, 1), size=(5.5, 11))
        ax = plt.subplot(gs[0])

    if normalize_tuning_curve:
        tuning_curves = mathutil.min_max_scaler(tuning_curves)
        pad = 1

    if sortby is None:
        # sort by the location of the placefield's maximum
        sort_ind = np.argsort(np.argmax(tuning_curves, axis=1))
    elif isinstance(sortby, (list, np.ndarray)):
        # use the provided sort indicies
        sort_ind = sortby
    else:
        sort_ind = np.arange(n_neurons)

    ## TODO: actually sort the ratemap object's neuron_ids and tuning_curves by the sort_ind
    # sorted_neuron_ids = ratemap.neuron_ids[sort_ind]
    
    sorted_neuron_ids = np.take_along_axis(np.array(ratemap.neuron_ids), sort_ind, axis=0)
    
    # sorted_alt_tuple_neuron_ids = np.take_along_axis(np.array(ratemap.metadata['tuple_neuron_ids']), sort_ind, axis=0)
    # sorted_alt_tuple_neuron_ids = np.take_along_axis(np.array(ratemap.tuple_neuron_ids), sort_ind, axis=0)

    sorted_alt_tuple_neuron_ids = ratemap.metadata['tuple_neuron_ids'].copy()
    sorted_alt_tuple_neuron_ids = [sorted_alt_tuple_neuron_ids[a_sort_idx] for a_sort_idx in sort_ind]
    
    
    # sorted_tuning_curves = tuning_curves[sorted_neuron_ids, :]
    # sorted_neuron_id_labels = [f'Cell[{a_neuron_id}]' for a_neuron_id in sorted_neuron_ids]
    sorted_neuron_id_labels = [f'C[{sorted_neuron_ids[i]}]({sorted_alt_tuple_neuron_ids[i][0]}|{sorted_alt_tuple_neuron_ids[i][1]})' for i in np.arange(len(sorted_neuron_ids))]
    
    colors_array = np.zeros((4, n_neurons))
    for i, neuron_ind in enumerate(sort_ind):
        color = cmap(i / len(sort_ind))
        colors_array[:, i] = color
        curr_neuron_id = sorted_neuron_ids[i]

        ax.fill_between(
            bin_cntr,
            i * pad,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            ec=None,
            alpha=0.5,
            zorder=i + 1,
        )
        ax.plot(
            bin_cntr,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            alpha=0.7,
        )
        ax.set_title('Cell[{}]'.format(curr_neuron_id)) # this doesn't appear to be visible, so what is it used for?

    # ax.set_yticks(list(range(len(sort_ind)) + 0.5))
    ax.set_yticks(list(np.arange(len(sort_ind)) + 0.5))
    # ax.set_yticklabels(list(sort_ind))
    ax.set_yticklabels(list(sorted_neuron_id_labels))
    
    ax.set_xlabel("Position")
    ax.spines["left"].set_visible(False)
    if normalize_xbin:
        ax.set_xlim([0, 1])
    ax.tick_params("y", length=0)
    ax.set_ylim([0, len(sort_ind)])
    # if self.run_dir is not None:
    #     ax.set_title(self.run_dir.capitalize() + " Runs only")

    return ax, sort_ind, colors_array

def plot_ratemap_2D(self, subplots=(10, 8), figsize=(6, 10), fignum=None, enable_spike_overlay=True):
    """Plots heatmaps of placefields with peak firing rate

    Parameters
    ----------
    speed_thresh : bool, optional
        [description], by default False
    subplots : tuple, optional
        number of cells within each figure window. If cells exceed the number of subplots, then cells are plotted in successive figure windows of same size, by default (10, 8)
    fignum : int, optional
        figure number to start from, by default None
    """
    map_use, thresh = self.ratemap.tuning_curves, self.speed_thresh

    nCells = len(map_use)
    nfigures = nCells // np.prod(subplots) + 1

    if fignum is None:
        if f := plt.get_fignums():
            fignum = f[-1] + 1
        else:
            fignum = 1

    figures, gs = [], []
    for fig_ind in range(nfigures):
        fig = plt.figure(fignum + fig_ind, figsize=figsize, clear=True)
        gs.append(gridspec(subplots[0], subplots[1], figure=fig))
        fig.subplots_adjust(hspace=0.2)
        
        title_string = f'2D Placemaps Placemaps ({len(self.ratemap.neuron_ids)} good cells)'
        if thresh is not None:
            title_string = f'{title_string} (speed_threshold = {str(thresh)})'
            
        fig.suptitle(title_string)
        figures.append(fig)

    mesh_X, mesh_Y = np.meshgrid(self.ratemap.xbin, self.ratemap.ybin)

    for cell, pfmap in enumerate(map_use):
        ind = cell // np.prod(subplots)
        subplot_ind = cell % np.prod(subplots)

        
        # Working:
        curr_pfmap = np.array(pfmap) / np.nanmax(pfmap)
        # curr_pfmap = np.rot90(np.fliplr(curr_pfmap)) ## Bug was introduced here! At least with pcolorfast, this order of operations is wrong!
        curr_pfmap = np.rot90(curr_pfmap)
        curr_pfmap = np.fliplr(curr_pfmap)
        # # curr_pfmap = curr_pfmap / np.nanmax(curr_pfmap) # for when the pfmap already had its transpose taken
        ax1 = figures[ind].add_subplot(gs[ind][subplot_ind])
        # ax1.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0, edgecolors='k', linewidths=0.1)
        # ax1.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0)
        
        im = ax1.pcolorfast(
            self.ratemap.xbin,
            self.ratemap.ybin,
            curr_pfmap,
            cmap="jet", vmin=0.0
        )
                
        
        # ax1.vlines(200, 'ymin'=0, 'ymax'=1, 'r')
        # ax1.set_xticks([25, 50])
        # ax1.vline(50, 'r')
        # ax1.vlines([50], 0, 1, transform=ax1.get_xaxis_transform(), colors='r')
        # ax1.vlines([50], 0, 1, colors='r')
            

        # im = ax1.pcolorfast(
        #     self.ratemap.xbin,
        #     self.ratemap.ybin,
        #     curr_pfmap,
        #     cmap="jet",
        #     vmin=0,
        # )
        # im = ax1.pcolorfast(
        #     self.ratemap.xbin,
        #     self.ratemap.ybin,
        #     np.rot90(np.fliplr(pfmap)) / np.nanmax(pfmap),
        #     cmap="jet",
        #     vmin=0,
        # )  # rot90(flipud... is necessary to match plotRaw configuration.
        # im = ax1.pcolor(
        #     self.ratemap.xbin,
        #     self.ratemap.ybin,
        #     np.rot90(np.fliplr(pfmap)) / np.nanmax(pfmap),
        #     cmap="jet",
        #     vmin=0,
        # )
        
        # ax1.scatter(self.spk_pos[ind]) # tODO: add spikes
        # max_frate =
        
        # if enable_spike_overlay:
        #     ax1.scatter(self.spk_pos[cell][0], self.spk_pos[cell][1], s=1, c='white', alpha=0.3, marker=',')
        #     # ax1.scatter(self.spk_pos[cell][1], self.spk_pos[cell][0], s=1, c='white', alpha=0.3, marker=',')
        
        curr_cell_alt_id = self.ratemap.tuple_neuron_ids[cell]
        curr_cell_shank = curr_cell_alt_id[0]
        curr_cell_cluster = curr_cell_alt_id[1]
        
        ax1.axis("off")
        ax1.set_title(
            f"Cell {self.ratemap.neuron_ids[cell]} - (shank {curr_cell_shank}, cluster {curr_cell_cluster}) \n{round(np.nanmax(pfmap),2)} Hz"
        )

        # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
        # cbar = fig.colorbar(im, cax=cbar_ax)
        # cbar.set_label("firing rate (Hz)")
        
    return figures, gs
    
    
    

def plot_raw(ratemap: core.Ratemap, t, x, run_dir, ax=None, subplots=(8, 9)):
    """Plot spike location on animal's path

    Parameters
    ----------
    speed_thresh : bool, optional
        [description], by default False
    ax : [type], optional
        [description], by default None
    subplots : tuple, optional
        [description], by default (8, 9)
    """

    # mapinfo = self.ratemaps
    mapinfo = ratemap
    nCells = len(mapinfo["pos"])

    def plot_(cell, ax):
        if subplots is None:
            ax.clear()
        ax.plot(x, t, color="gray", alpha=0.6)
        ax.plot(mapinfo["pos"][cell], mapinfo["spikes"][cell], ".", color="#ff5f5c")
        ax.set_title(
            " ".join(filter(None, ("Cell", str(cell), run_dir.capitalize())))
        )
        ax.invert_yaxis()
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Time (s)")

    if ax is None:

        if subplots is None:
            _, gs = Fig().draw(grid=(1, 1), size=(6, 8))
            ax = plt.subplot(gs[0])
            widgets.interact(
                plot_,
                cell=widgets.IntSlider(
                    min=0,
                    max=nCells - 1,
                    step=1,
                    description="Cell ID:",
                ),
                ax=widgets.fixed(ax),
            )
        else:
            _, gs = Fig().draw(grid=subplots, size=(10, 11))
            for cell in range(nCells):
                ax = plt.subplot(gs[cell])
                ax.set_yticks([])
                plot_(cell, ax)
