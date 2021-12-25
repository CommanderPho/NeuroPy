from enum import Enum, IntEnum, auto, unique
from collections import namedtuple
from ipywidgets import widgets
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple

from neuropy.utils.misc import AutoNameEnum, chunks
from .. import core
from neuropy.utils import mathutil
from .figure import Fig

## TODO: refactor plot_ratemap_1D and plot_ratemap_2D to a single flat function (if that's appropriate).
## TODO: refactor plot_ratemap_1D and plot_ratemap_2D to take a **kwargs and apply optional defaults (find previous code where I did that using the | and dict conversion. In my 3D code.


@unique
class enumTuningMap2DPlotMode(AutoNameEnum):
    PCOLORFAST = auto() # DEFAULT prior to 2021-12-24
    PCOLORMESH = auto() # UNTESTED
    PCOLOR = auto() # UNTESTED
    IMSHOW = auto() # New Default as of 2021-12-24

    
def plot_single_tuning_map_2D(xbin, ybin, pfmap, occupancy, neuron_extended_id: NeuronExtendedIdentityTuple=None, drop_below_threshold: float=0.0000001, plot_mode: enumTuningMap2DPlotMode=None, ax=None):
    """Plots a single tuning curve Heatmap

    Args:
        xbin ([type]): [description]
        ybin ([type]): [description]
        pfmap ([type]): [description]
        occupancy ([type]): [description]
        drop_below_threshold (float, optional): [description]. Defaults to 0.0000001.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if plot_mode is None:    
        # plot_mode = enumTuningMap2DPlotMode.PCOLORFAST
        plot_mode = enumTuningMap2DPlotMode.IMSHOW
    
    use_alpha_by_occupancy = False # Only supported in IMSHOW mode
    
    if ax is None:
        ax = plt.gca()
            
    curr_pfmap = np.array(pfmap.copy()) / np.nanmax(pfmap)
    if drop_below_threshold is not None:
        curr_pfmap[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy
    
    # curr_pfmap = np.rot90(np.fliplr(curr_pfmap)) ## Bug was introduced here! At least with pcolorfast, this order of operations is wrong!
    # curr_pfmap = np.rot90(curr_pfmap)
    # curr_pfmap = np.fliplr(curr_pfmap) # I thought stopping after this was sufficient, as the values lined up with the 1D placefields... but it seems to be flipped vertically now!
    
    ## Seems to work:
    curr_pfmap = np.rot90(curr_pfmap, k=-1)
    curr_pfmap = np.fliplr(curr_pfmap)
        
    # # curr_pfmap = curr_pfmap / np.nanmax(curr_pfmap) # for when the pfmap already had its transpose taken

    if plot_mode is enumTuningMap2DPlotMode.PCOLORFAST:
        im = ax.pcolorfast(
            xbin,
            ybin,
            curr_pfmap,
            cmap="jet", vmin=0.0
        )
        
    elif plot_mode is enumTuningMap2DPlotMode.PCOLORMESH:
        raise DeprecationWarning # 'Code not maintained, may be out of date'  
        mesh_X, mesh_Y = np.meshgrid(xbin, ybin)
        ax.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0, edgecolors='k', linewidths=0.1)
        # ax.pcolormesh(mesh_X, mesh_Y, curr_pfmap, cmap='jet', vmin=0)
        
    elif plot_mode is enumTuningMap2DPlotMode.PCOLOR: 
        raise DeprecationWarning # 'Code not maintained, may be out of date'
        im = ax.pcolor(
            xbin,
            ybin,
            np.rot90(np.fliplr(pfmap)) / np.nanmax(pfmap),
            cmap="jet",
            vmin=0,
        )    
    elif plot_mode is enumTuningMap2DPlotMode.IMSHOW:
        """ https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html """
        """ Use the brightness to reflect the confidence in the outcome. Could also use opacity. """
        # mesh_X, mesh_Y = np.meshgrid(xbin, ybin)
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1])
        # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
        extent = (xmin, xmax, ymin, ymax)
        # print(f'extent: {extent}')
        # extent = None
        # We'll also create a black background into which the pixels will fade
        background_black = np.full((*curr_pfmap.shape, 3), 0, dtype=np.uint8)
        
        vmax = np.abs(curr_pfmap).max()
                
        imshow_shared_kwargs = {
            'origin': 'lower',
            'extent': extent,
        }
        
        main_plot_kwargs = imshow_shared_kwargs | {
            # 'vmax': vmax,
            'vmin': 0,
            'cmap': 'jet',
        }
        
        if use_alpha_by_occupancy:
            # alphas = np.ones(curr_pfmap.shape)
            # alphas[:, :] = np.linspace(1, 0, curr_pfmap.shape[1]) # Test, blend transparency linearly
            # Normalize:
            # Create an alpha channel based on weight values
            # Any value whose absolute value is > .0001 will have zero transparency
            alphas = Normalize(clip=True)(np.abs(occupancy))
            # alphas = Normalize(0, .3, clip=True)(np.abs(occupancy))
            # alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4

            main_plot_kwargs['alpha'] = alphas
            pass
        else:
            main_plot_kwargs['alpha'] = None
        
        ax.imshow(background_black, **imshow_shared_kwargs)
        im = ax.imshow(curr_pfmap, **main_plot_kwargs)
        

    else:
        raise NotImplementedError   
    
    # ax.vlines(200, 'ymin'=0, 'ymax'=1, 'r')
    # ax.set_xticks([25, 50])
    # ax.vline(50, 'r')
    # ax.vlines([50], 0, 1, transform=ax.get_xaxis_transform(), colors='r')
    # ax.vlines([50], 0, 1, colors='r')


    
    # ax.axis("off")
    extended_id_string = f'(shank {neuron_extended_id.shank}, cluster {neuron_extended_id.cluster})'
    ax.set_title(
            f"Cell {neuron_extended_id.id} - {extended_id_string} \n{round(np.nanmax(pfmap),2)} Hz"
    ) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"
    
    return im
    

@unique
class enumTuningMap2DPlotVariables(AutoNameEnum):
    TUNING_MAPS = auto() # DEFAULT
    FIRING_MAPS = auto() 
    

RequiredSubplotsTuple = namedtuple('RequiredSubplotsTuple', 'num_required_subplots num_columns num_rows combined_indicies')
SubplotGridSizeTuple = namedtuple('SubplotGridSizeTuple', 'num_rows num_columns')
PaginatedGridIndexSpecifierTuple = namedtuple('PaginatedGridIndexSpecifierTuple', 'linear_idx row_idx col_idx data_idx')



def compute_paginated_grid_config(num_required_subplots, max_num_columns, max_subplots_per_page, data_indicies=None, last_figure_subplots_same_layout=True):
    """ Fills row-wise first 

    Args:
        num_required_subplots ([type]): [description]
        max_num_columns ([type]): [description]
        max_subplots_per_page ([type]): [description]
        data_indicies ([type], optional): your indicies into your original data that will also be accessible in the main loop. Defaults to None.
    """
    
    def _compute_subplots_grid_size(num_page_required_subplots, page_max_num_columns):
        """ For a single page """
        fixed_columns = min(page_max_num_columns, num_page_required_subplots) # if there aren't enough plots to even fill up a whole row, reduce the number of columns
        total_needed_rows = int(np.ceil(num_page_required_subplots / fixed_columns))
        return SubplotGridSizeTuple(total_needed_rows, fixed_columns)
    
    def _compute_num_subplots(num_required_subplots, max_num_columns, data_indicies=None):
        linear_indicies = np.arange(num_required_subplots)
        if data_indicies is None:
            data_indicies = np.arange(num_required_subplots) # the data_indicies are just the same as the lienar indicies unless otherwise specified
            
        # fixed_columns = min(max_num_columns, num_required_subplots) # if there aren't enough plots to even fill up a whole row, reduce the number of columns
        # total_needed_rows = int(np.ceil(num_required_subplots / fixed_columns))
        
        (total_needed_rows, fixed_columns) = _compute_subplots_grid_size(num_required_subplots, max_num_columns)
        
        all_row_column_indicies = np.unravel_index(linear_indicies, (total_needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
        all_combined_indicies = [PaginatedGridIndexSpecifierTuple(linear_indicies[i], all_row_column_indicies[0][i], all_row_column_indicies[1][i], data_indicies[i]) for i in np.arange(len(linear_indicies))]
         
        return RequiredSubplotsTuple(num_required_subplots, fixed_columns, total_needed_rows, all_combined_indicies)

    subplot_no_pagination_configuration = _compute_num_subplots(num_required_subplots, max_num_columns=max_num_columns, data_indicies=data_indicies)
    
    # flat_row_indices = np.arange(subplot_no_pagination_configuration.num_rows)
    
    included_combined_indicies_pages = [list(chunk) for chunk in chunks(subplot_no_pagination_configuration.combined_indicies, max_subplots_per_page)]
    
    if last_figure_subplots_same_layout:
        page_grid_sizes = [_compute_subplots_grid_size(subplot_no_pagination_configuration.num_rows, subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]
        
    else:
        page_grid_sizes = [_compute_subplots_grid_size(len(a_page), subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]

    print(f'page_grid_sizes: {page_grid_sizes}')
    return subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes
    

# all extracted from the 2D figures
def plot_ratemap_2D(ratemap: core.Ratemap, computation_config=None, included_unit_indicies=None, subplots:SubplotGridSizeTuple=(10, 8), figsize=(6, 10), fignum=None, enable_spike_overlay=False, spike_overlay_spikes=None, drop_below_threshold: float=0.0000001, plot_variable: enumTuningMap2DPlotVariables=enumTuningMap2DPlotVariables.TUNING_MAPS, plot_mode: enumTuningMap2DPlotMode=None):
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
    
    # last_figure_subplots_same_layout = False
    last_figure_subplots_same_layout = True
    
    if not isinstance(subplots, SubplotGridSizeTuple):
        subplots = SubplotGridSizeTuple(subplots[0], subplots[1])
    if included_unit_indicies is None:
        included_unit_indicies = np.arange(ratemap.n_neurons) # include all unless otherwise specified
    
    if plot_variable is enumTuningMap2DPlotVariables.TUNING_MAPS:
        active_maps = ratemap.tuning_curves[included_unit_indicies]
        title_substring = 'Placemaps'
    elif plot_variable == enumTuningMap2DPlotVariables.FIRING_MAPS:
        active_maps = ratemap.firing_maps[included_unit_indicies]
        title_substring = 'Firing Maps'
    else:
        raise ValueError

    nMapsToShow = len(active_maps)
    
    max_subplots_per_page = int(subplots.num_columns * subplots.num_rows)
    
    # TODO: constrain the subplots values to just those that you need
    # the subplot dimension with the fewest entries is the first candiate for removal
    
    #     if (subplots[0] < subplots[1]):
    #         first_removal_candiate_dim = 1
    #     else:
    #         first_removal_candiate_dim = 0
    
    subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=subplots.num_columns, max_subplots_per_page=max_subplots_per_page, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
    num_pages = len(included_combined_indicies_pages)

    nfigures = num_pages
    # nfigures = nMapsToShow // np.prod(subplots) + 1 # "//" is floor division (rounding result down to nearest whole number)

    if fignum is None:
        if f := plt.get_fignums():
            fignum = f[-1] + 1
        else:
            fignum = 1

    figures, page_gs, page_axes = [], []
    for fig_ind in range(nfigures):
        
        
        # fig, axs = plt.subplots(ncols=page_grid_sizes[fig_ind].num_columns, nrows=page_grid_sizes[fig_ind].num_rows, figsize=figsize, clear=True, constrained_layout=True)
        # page_axes.append(axs)

        fig = plt.figure(fignum + fig_ind, figsize=figsize, clear=True)        
        # gs.append(GridSpec(subplots[0], subplots[1], figure=fig))        
        if last_figure_subplots_same_layout:
            page_gs.append(GridSpec(subplot_no_pagination_configuration.num_rows, subplot_no_pagination_configuration.num_columns, figure=fig))
        else:
            # print(f'fig_ind {fig_ind}: page_grid_sizes[fig_ind]: {page_grid_sizes[fig_ind]}')
            page_gs.append(GridSpec(page_grid_sizes[fig_ind].num_rows, page_grid_sizes[fig_ind].num_columns, figure=fig))
            
        
        fig.subplots_adjust(hspace=0.2)
        
        title_string = f'2D Placemaps {title_substring} ({len(ratemap.neuron_ids)} good cells)'
        
        if computation_config is not None:
            if computation_config.speed_thresh is not None:
                title_string = f'{title_string} (speed_threshold = {str(computation_config.speed_thresh)})'
            
        fig.suptitle(title_string)
        figures.append(fig)

    # New page-based version:
    for page_idx in np.arange(num_pages):
        print(f'page_idx: {page_idx}')
        for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
            # curr_included_unit_index = included_unit_indicies[a_linear_index]
            # curr_included_unit_index = curr_data_idx[a_linear_index]
            
            
            # Need to convert to page specific:
            curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
            curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
            curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
            
            # print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')
            
            cell_idx = curr_included_unit_index
            pfmap = active_maps[a_linear_index]
            # Get the axis to plot on:
            # ax1 = figures[page_idx].add_subplot(gs[page_idx][a_linear_index])
            
            # ax1 = page_axes[page_idx][curr_page_relative_row, curr_page_relative_col]
            ax1 = figures[page_idx].add_subplot(page_gs[page_idx][curr_page_relative_linear_index])
                        
            # Plot the main heatmap for this pfmap:
            im = plot_single_tuning_map_2D(ratemap.xbin, ratemap.ybin, pfmap, ratemap.occupancy, neuron_extended_id=ratemap.neuron_extended_ids[cell_idx], drop_below_threshold=drop_below_threshold, plot_mode=plot_mode, ax=ax1)
            
            if enable_spike_overlay:
                ax1.scatter(spike_overlay_spikes[cell_idx][0], spike_overlay_spikes[cell_idx][1], s=2, c='white', alpha=0.10, marker=',')
            
            # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
            # cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar.set_label("firing rate (Hz)")
        

    # # Original version:
    # for active_map_idx, pfmap in enumerate(active_maps):
    #     figure_idx = active_map_idx // np.prod(subplots)
    #     subplot_ind = active_map_idx % np.prod(subplots)
    #     ax1 = figures[figure_idx].add_subplot(gs[figure_idx][subplot_ind])
        
    #     cell_idx = included_unit_indicies[active_map_idx]
        
    #     # Plot the main heatmap for this pfmap:
    #     im = plot_single_tuning_map_2D(ratemap.xbin, ratemap.ybin, pfmap, ratemap.occupancy, neuron_extended_id=ratemap.neuron_extended_ids[cell_idx], drop_below_threshold=drop_below_threshold, plot_mode=plot_mode, ax=ax1)
        
    #     if enable_spike_overlay:
    #         ax1.scatter(spike_overlay_spikes[cell_idx][0], spike_overlay_spikes[cell_idx][1], s=2, c='white', alpha=0.10, marker=',')
           
    #     # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
    #     # cbar = fig.colorbar(im, cax=cbar_ax)
    #     # cbar.set_label("firing rate (Hz)")
        
    return figures, page_gs
    

def plot_ratemap_1D(ratemap: core.Ratemap, normalize_xbin=False, ax=None, pad=2, normalize_tuning_curve=False, sortby=None, cmap="tab20b"):
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
    
    sorted_alt_tuple_neuron_ids = ratemap.neuron_extended_ids.copy()
    # sorted_alt_tuple_neuron_ids = ratemap.metadata['tuple_neuron_ids'].copy()
    sorted_alt_tuple_neuron_ids = [sorted_alt_tuple_neuron_ids[a_sort_idx] for a_sort_idx in sort_ind]
    
    # sorted_tuning_curves = tuning_curves[sorted_neuron_ids, :]
    # sorted_neuron_id_labels = [f'Cell[{a_neuron_id}]' for a_neuron_id in sorted_neuron_ids]
    sorted_neuron_id_labels = [f'C[{sorted_neuron_ids[i]}]({sorted_alt_tuple_neuron_ids[i].shank}|{sorted_alt_tuple_neuron_ids[i].cluster})' for i in np.arange(len(sorted_neuron_ids))]
    
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
