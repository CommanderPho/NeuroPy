from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, interpolation
from neuropy.core.epoch import Epoch
from neuropy.core.neurons import Neurons
from neuropy.core.position import Position
from neuropy.core.ratemap import Ratemap

from neuropy.plotting.figure import pretty_plot
from neuropy.plotting.mixins.placemap_mixins import PfnDPlottingMixin
from neuropy.utils.misc import is_iterable

# from pyphoplacecellanalysis.General.Configs.DynamicConfigs import PlottingConfig, InteractivePlaceCellConfig
from neuropy.utils.mixins.diffable import DiffableObject # for compute_placefields_as_needed type-hinting

# from .. import core
# import neuropy.core as core
from .. import plotting
from neuropy.utils.mixins.print_helpers import SimplePrintable, OrderedMeta


class PlacefieldComputationParameters(SimplePrintable, DiffableObject, metaclass=OrderedMeta):
    """A simple wrapper object for parameters used in placefield calcuations"""
    decimal_point_character=","
    param_sep_char='-'
    variable_names=['speed_thresh', 'grid_bin', 'smooth', 'frate_thresh']
    variable_inline_names=['speedThresh', 'gridBin', 'smooth', 'frateThresh']
    variable_inline_names=['speedThresh', 'gridBin', 'smooth', 'frateThresh']
    
    def __init__(self, speed_thresh=3, grid_bin=2, smooth=2, frate_thresh=1, **kwargs):
        self.speed_thresh = speed_thresh
        if not isinstance(grid_bin, (tuple, list)):
            grid_bin = (grid_bin, grid_bin) # make it into a 2 element tuple
        self.grid_bin = grid_bin
        if not isinstance(smooth, (tuple, list)):
            smooth = (smooth, smooth) # make it into a 2 element tuple
        self.smooth = smooth
        self.frate_thresh = frate_thresh
        
        # Dump all arguments into parameters.
        for key, value in kwargs.items():
            setattr(self, key, value)
                

    @property
    def grid_bin_1D(self):
        """The grid_bin_1D property."""
        if np.isscalar(self.grid_bin):
            return self.grid_bin
        else:
            return self.grid_bin[0]

    @property
    def smooth_1D(self):
        """The smooth_1D property."""
        if np.isscalar(self.smooth):
            return self.smooth
        else:
            return self.smooth[0]

    def _unlisted_parameter_strings(self):
        """ returns the string representations of all key/value pairs that aren't normally defined. """
        # Dump all arguments into parameters.
        out_list = []
        for key, value in self.__dict__.items():
            if (key is not None) and (key not in PlacefieldComputationParameters.variable_names):
                if value is None:
                    out_list.append(f"{key}_None")
                elif isinstance(value, float):
                    out_list.append(f"{key}_{value:.2f}")
                else:
                    try:
                        out_list.append(f"{key}_{value}")
                    # except TypeError as e:
                    #     print(f'TypeError: {e}. type(value): {type(value)}')
                    #     print(f'self.__dict__: {self.__dict__}')
                    #     # print(f"{key}_{value}")
                    #     # raise e
                    #     out_list.append(f"{key}_{type(value)}")
                    except Exception as e:
                        print(f'UNEXPECTED_EXCEPTION: {e}')
                        print(f'self.__dict__: {self.__dict__}')
                        raise e

        return out_list
        
    
    def str_for_filename(self, is_2D):
        extras_strings = self._unlisted_parameter_strings()
        if is_2D:
            return '-'.join([f"speedThresh_{self.speed_thresh:.2f}", f"gridBin_{self.grid_bin[0]:.2f}_{self.grid_bin[1]:.2f}", f"smooth_{self.smooth[0]:.2f}_{self.smooth[1]:.2f}", f"frateThresh_{self.frate_thresh:.2f}", *extras_strings])
            # return "speedThresh_{:.2f}-gridBin_{:.2f}_{:.2f}-smooth_{:.2f}_{:.2f}-frateThresh_{:.2f}".format(self.speed_thresh, self.grid_bin[0], self.grid_bin[1], self.smooth[0], self.smooth[1], self.frate_thresh)
            # return f"speedThresh_{self.speed_thresh:.2f}-gridBin_{self.grid_bin[0]:.2f}_{self.grid_bin[1]:.2f}-smooth_{self.smooth[0]:.2f}_{self.smooth[1]:.2f}-frateThresh_{self.frate_thresh:.2f}"
        else:
            return '-'.join([f"speedThresh_{self.speed_thresh:.2f}", f"gridBin_{self.grid_bin_1D:.2f}", f"smooth_{self.smooth_1D:.2f}", f"frateThresh_{self.frate_thresh:.2f}", *extras_strings])
            # return f"speedThresh_{self.speed_thresh:.2f}-gridBin_{self.grid_bin_1D:.2f}-smooth_{self.smooth_1D:.2f}-frateThresh_{self.frate_thresh:.2f}"
        
    def str_for_display(self, is_2D):
        """ For rendering in a title, etc """
        extras_string = ', '.join(self._unlisted_parameter_strings())
        if is_2D:
            return f"(speedThresh_{self.speed_thresh:.2f}, gridBin_{self.grid_bin[0]:.2f}_{self.grid_bin[1]:.2f}, smooth_{self.smooth[0]:.2f}_{self.smooth[1]:.2f}, frateThresh_{self.frate_thresh:.2f})" + extras_string
        else:
            return f"(speedThresh_{self.speed_thresh:.2f}, gridBin_{self.grid_bin_1D:.2f}, smooth_{self.smooth_1D:.2f}, frateThresh_{self.frate_thresh:.2f})" + extras_string

    @classmethod
    def _build_formatted_str_for_output(cls, dict_items, param_sep_char, key_val_sep_char) -> str:
        with np.printoptions(precision=3, suppress=True, threshold=5):
            properties_key_val_list = []
            for (name, val) in dict_items.items():
                try:
                    curr_string = f'{name}{key_val_sep_char}{np.array(val)}'
                except TypeError:
                    curr_string = f'{name}{key_val_sep_char}err'     
                properties_key_val_list.append(curr_string)
            # properties_key_val_list = [f'{name}{key_val_sep_char}{np.array(val)}' for (name, val) in dict_items.items()]
        
        return param_sep_char.join(properties_key_val_list)

    

    def str_for_attributes_list_display(self, param_sep_char='\n', key_val_sep_char='\t'):
        """ For rendering in attributes list like outputs 
        # Default for attributes lists outputs:
        Example Output:
            speed_thresh	2.0
            grid_bin	[3.777 1.043]
            smooth	[1.5 1.5]
            frate_thresh	0.1
            time_bin_size	0.5
        """
        # param_sep_char='\n'
        # key_val_sep_char='\t'
        return PlacefieldComputationParameters._build_formatted_str_for_output(self.__dict__, param_sep_char, key_val_sep_char)
        


    def __hash__(self):
        """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
        # return hash((self.age, self.name))
        member_names_tuple = list(self.__dict__.keys())
        values_tuple = list(self.__dict__.values())
        combined_tuple = tuple(member_names_tuple + values_tuple)
        return hash(combined_tuple)


def _normalized_occupancy(raw_occupancy, dt=None, position_srate=None):
    # raw occupancy is defined in terms of the number of samples that fall into each bin.
    # if position_srate is not None:
    #     dt = 1.0 / float(position_srate)
    #  seconds_occupancy is the number of seconds spent in each bin. This is computed by multiplying the raw occupancy (in # samples) by the duration of each sample.
    # seconds_occupancy = raw_occupancy * dt  # converting to seconds
    seconds_occupancy = raw_occupancy / (float(position_srate) + 1e-16) # converting to seconds
    # seconds_occupancy = occupancy / (position_srate + 1e-16)  # converting to seconds
    # normalized occupancy gives the ratio of samples that feel in each bin. ALL BINS ADD UP TO ONE.
    normalized_occupancy = raw_occupancy / np.nansum(raw_occupancy) # the normalized occupancy determines the relative number of samples spent in each bin

    return seconds_occupancy, normalized_occupancy



class PfnConfigMixin:
    def str_for_filename(self, is_2D=True):
        return self.config.str_for_filename(is_2D)

    
class PfnDMixin(SimplePrintable):
    
    should_smooth_speed = False
    should_smooth_firing_map = False
    should_smooth_spatial_occupancy_map = False
    should_smooth_final_tuning_map = True
    
    @property
    def spk_pos(self):
        return self.ratemap_spiketrains_pos
    
    @property
    def spk_t(self):
        return self.ratemap_spiketrains
    
    @property
    def cell_ids(self):
        return self.ratemap.neuron_ids

    def plot_raw(self, subplots=(10, 8), fignum=None, alpha=0.5, label_cells=False, ax=None, clus_use=None):
        """ Plots the Placefield raw spiking activity for all cells"""
        if self.ndim < 2:
            ## TODO: Pf1D Temporary Workaround:
            return plotting.plot_raw(self.ratemap, self.t, self.x, 'BOTH', ax=ax, subplots=subplots)
        else:        
            if ax is None:
                fig = plt.figure(fignum, figsize=(12, 20))
                gs = GridSpec(subplots[0], subplots[1], figure=fig)
                # fig.subplots_adjust(hspace=0.4)
            else:
                assert len(ax) == len(clus_use), "Number of axes must match number of clusters to plot"
                fig = ax[0].get_figure()

            # spk_pos_use = self.spk_pos
            spk_pos_use = self.ratemap_spiketrains_pos

            if clus_use is not None:
                spk_pos_tmp = spk_pos_use
                spk_pos_use = []
                [spk_pos_use.append(spk_pos_tmp[a]) for a in clus_use]

            for cell, (spk_x, spk_y) in enumerate(spk_pos_use):
                if ax is None:
                    ax1 = fig.add_subplot(gs[cell])
                else:
                    ax1 = ax[cell]
                ax1.plot(self.x, self.y, color="#d3c5c5") # Plot the animal's position. This will be the same for all cells
                ax1.plot(spk_x, spk_y, '.', markersize=0.8, color=[1, 0, 0, alpha]) # plot the cell-specific spike locations
                ax1.axis("off")
                if label_cells:
                    # Put cell info (id, etc) on title
                    info = self.cell_ids[cell]
                    ax1.set_title(f"Cell {info}")

            fig.suptitle(f"Place maps for cells with their peak firing rate (frate thresh={self.frate_thresh},speed_thresh={self.speed_thresh})")
            return fig
            
        
    def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
        """ Builds one subplot for each dimension of the position data
        Updated to work with both 1D and 2D Placefields
        """   
        if ax is None:
            fig, ax = plt.subplots(self.ndim, 1, sharex=True)
            fig.set_size_inches([23, 9.7])
        
        if not is_iterable(ax):
            ax = [ax]
            
        # plot trajectories
        if self.ndim < 2:
            variable_array = [self.x]
            label_array = ["X position (cm)"]
        else:
            variable_array = [self.x, self.y]
            label_array = ["X position (cm)", "Y position (cm)"]
            
        for a, pos, ylabel in zip(ax, variable_array, label_array):
            a.plot(self.t, pos)
            a.set_xlabel("Time (seconds)")
            a.set_ylabel(ylabel)
            pretty_plot(a)

        # Grab correct spike times/positions
        if speed_thresh:
            spk_pos_, spk_t_ = self.run_spk_pos, self.run_spk_t
        else:
            spk_pos_, spk_t_ = self.spk_pos, self.spk_t

        # plot spikes on trajectory
        for a, pos in zip(ax, spk_pos_[cellind]):
            a.plot(spk_t_[cellind], pos, ".", color=[0, 0, 0.8, alpha])

        # Put info on title
        ax[0].set_title(
            "Cell "
            + str(self.cell_ids[cellind])
            + ":, speed_thresh="
            + str(self.speed_thresh)
        )
        return ax

    def plot_all(self, cellind, speed_thresh=True, alpha=0.4, fig=None):
        if fig is None:
            fig_use = plt.figure(figsize=[28.25, 11.75])
        else:
            fig_use = fig
        gs = GridSpec(2, 4, figure=fig_use)
        ax2d = fig_use.add_subplot(gs[0, 0])
        axccg = np.asarray(fig_use.add_subplot(gs[1, 0]))
        axx = fig_use.add_subplot(gs[0, 1:])
        axy = fig_use.add_subplot(gs[1, 1:], sharex=axx)

        self.plot_raw(speed_thresh=speed_thresh, clus_use=[cellind], ax=[ax2d])
        self.plotRaw_v_time(cellind, speed_thresh=speed_thresh, ax=[axx, axy], alpha=alpha)
        self._obj.spikes.plot_ccg(clus_use=[cellind], type="acg", ax=axccg)

        return fig_use


class Pf1D(PfnConfigMixin, PfnDMixin):
    
    @staticmethod
    def _compute_occupancy(x, xbin, position_srate, smooth):
        # --- occupancy map calculation -----------
        # NRK todo: might need to normalize occupancy so sum adds up to 1
        raw_occupancy, xedges = np.histogram(x, bins=xbin)
        if ((smooth is not None) and (smooth > 0.0)):
            raw_occupancy = gaussian_filter1d(raw_occupancy, sigma=smooth)
        # # raw occupancy is defined in terms of the number of samples that fall into each bin.
        seconds_occupancy, normalized_occupancy = _normalized_occupancy(raw_occupancy, position_srate=position_srate)
        return seconds_occupancy, xedges
    
    @staticmethod   
    def _compute_firing_map(spk_x, xbin, smooth):
        firing_map = np.histogram(spk_x, bins=xbin)[0]
        if ((smooth is not None) and (smooth > 0.0)):
            firing_map = gaussian_filter1d(firing_map, sigma=smooth)
        return firing_map
    
    @staticmethod   
    def _compute_tuning_map(spk_x, xbin, occupancy, smooth, should_also_return_intermediate_firing_map=False):
        if not PfnDMixin.should_smooth_firing_map:
            smooth_firing_map = None
        else:
            smooth_firing_map = smooth
        firing_map = Pf1D._compute_firing_map(spk_x, xbin, smooth_firing_map)
        tuning_map = firing_map / occupancy
        if PfnDMixin.should_smooth_final_tuning_map and ((smooth is not None) and (smooth > 0.0)):
            tuning_map = gaussian_filter1d(tuning_map, sigma=smooth)
        if should_also_return_intermediate_firing_map:
            return tuning_map, firing_map
        else:
            return tuning_map
    
    def __init__(self, neurons: Neurons, position: Position, epochs: Epoch = None, frate_thresh=1, speed_thresh=5, grid_bin=1, smooth=1, ):
        raise DeprecationWarning



class Pf2D(PfnConfigMixin, PfnDMixin):

    @staticmethod
    def _compute_occupancy(x, y, xbin, ybin, position_srate, smooth, should_return_raw_occupancy=False):
        # --- occupancy map calculation -----------
        # NRK todo: might need to normalize occupancy so sum adds up to 1
        # Please note that the histogram does not follow the Cartesian convention where x values are on the abscissa and y values on the ordinate axis. Rather, x is histogrammed along the first dimension of the array (vertical), and y along the second dimension of the array (horizontal).
        raw_occupancy, xedges, yedges = np.histogram2d(x, y, bins=(xbin, ybin))
        # occupancy = occupancy.T # transpose the occupancy before applying other operations
        # raw_occupancy = raw_occupancy / position_srate + 10e-16  # converting to seconds
        if ((smooth is not None) and ((smooth[0] > 0.0) & (smooth[1] > 0.0))): 
            raw_occupancy = gaussian_filter(raw_occupancy, sigma=(smooth[1], smooth[0])) # 2d gaussian filter
        # Histogram does not follow Cartesian convention (see Notes),
        # therefore transpose occupancy for visualization purposes.
        # raw occupancy is defined in terms of the number of samples that fall into each bin.
        if should_return_raw_occupancy:
            return raw_occupancy, xedges, yedges
        else:   
            seconds_occupancy, normalized_occupancy = _normalized_occupancy(raw_occupancy, position_srate=position_srate)
            return seconds_occupancy, xedges, yedges


        # return seconds_occupancy, xedges, yedges
        
    @staticmethod   
    def _compute_firing_map(spk_x, spk_y, xbin, ybin, smooth):
        # firing_map: is the number of spike counts in each bin for this unit
        firing_map = np.histogram2d(spk_x, spk_y, bins=(xbin, ybin))[0]
        if ((smooth is not None) and ((smooth[0] > 0.0) & (smooth[1] > 0.0))):
            firing_map = gaussian_filter(firing_map, sigma=(smooth[1], smooth[0])) # need to flip smooth because the x and y are transposed
        return firing_map
    
    @staticmethod   
    def _compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, smooth, should_also_return_intermediate_firing_map=False):
        # raw_tuning_map: is the number of spike counts in each bin for this unit
        if not PfnDMixin.should_smooth_firing_map:
            smooth_firing_map = None
        else:
            smooth_firing_map = smooth
        firing_map = Pf2D._compute_firing_map(spk_x, spk_y, xbin, ybin, smooth_firing_map)
        occupancy[occupancy == 0.0] = np.nan # pre-set the zero occupancy locations to NaN to avoid a warning in the next step. They'll be replaced with zero afterwards anyway
        occupancy_weighted_tuning_map = firing_map / occupancy # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
        occupancy_weighted_tuning_map = np.nan_to_num(occupancy_weighted_tuning_map, copy=True, nan=0.0) # set any NaN values to 0.0, as this is the correct weighted occupancy
        occupancy[np.isnan(occupancy)] = 0.0 # restore these entries back to zero
        
        if PfnDMixin.should_smooth_final_tuning_map and ((smooth is not None) and ((smooth[0] > 0.0) & (smooth[1] > 0.0))):
            occupancy_weighted_tuning_map = gaussian_filter(occupancy_weighted_tuning_map, sigma=(smooth[1], smooth[0])) # need to flip smooth because the x and y are transposed
            
        if should_also_return_intermediate_firing_map:
            return occupancy_weighted_tuning_map, firing_map
        else:
            return occupancy_weighted_tuning_map

    def __init__(self, neurons: Neurons, position: Position, epochs: Epoch = None, frate_thresh=1, speed_thresh=5, grid_bin=(1,1), smooth=(1,1), ):
        raise DeprecationWarning

# First, interested in answering the question "where did the animal spend its time on the track" to assess the relative frequency of events that occur in a given region. If the animal spends a lot of time in a certain region,
# it's more likely that any cell, not just the ones that hold it as a valid place field, will fire there.
    # this can be done by either binning (lumping close position points together based on a standardized grid), neighborhooding, or continuous smearing. 

class PfND(PfnConfigMixin, PfnDMixin, PfnDPlottingMixin):
    """Represents a collection of placefields over binned,  N-dimensional space. """

    def __init__(self, spikes_df: pd.DataFrame, position: Position, epochs: Epoch = None, frate_thresh=1, speed_thresh=5, grid_bin=(1,1), smooth=(1,1)):
        """computes 2d place field using (x,y) coordinates. It always computes two place maps with and
        without speed thresholds.

        Parameters
        ----------
        spikes_df: pd.DataFrame
        position : core.Position
        epochs : core.Epoch
            specifies the list of epochs to include.
        grid_bin : int
            bin size of position bining, by default 5
        speed_thresh : int
            speed threshold for calculating place field
        """
        self._save_intermediate_firing_maps = True # False is not yet implemented
        # save the config that was used to perform the computations
        self.config = PlacefieldComputationParameters(speed_thresh=speed_thresh, grid_bin=grid_bin, smooth=smooth, frate_thresh=frate_thresh)
        self.position_srate = position.sampling_rate
        # Set the dimensionality of the PfND object from the position's dimensionality
        self.ndim = position.ndim
        
        self._included_thresh_neurons_indx = None
        self._peak_frate_filter_function = None        
        self.ratemap = None
        self.ratemap_spiketrains = None
        self.ratemap_spiketrains_pos = None
        # self.t = None
        # self.x = None
        # self.speed = None
        # self.y = None
        self._filtered_pos_df = None
        self._filtered_spikes_df = None
        self.xbin = None
        self.ybin = None 
        self.bin_info = None
        
        # Perform the primary setup to build the placefield
        self.setup(position, spikes_df, epochs)
        self.compute()
        
 
    
        # done!
    def setup(self, position: Position, spikes_df, epochs: Epoch, debug_print=False):
        """ do the preliminary setup required to build the placefields
        
        Adds columns to the spikes and positions dataframes, etc.
        """

        pos_df = position.to_dataframe()
        spk_df = spikes_df.copy()
        
        # filtering:
        if epochs is not None:
            # filter the spikes_df:
            self._filtered_spikes_df = spk_df.spikes.time_sliced(epochs.starts, epochs.stops)
            # filter the pos_df:
            self._filtered_pos_df = pos_df.position.time_sliced(epochs.starts, epochs.stops) # 5378 rows × 18 columns
        else:
            # if no epochs filtering, set the filtered objects to be sliced by the available range of the position data (given by position.t_start, position.t_stop)
            self._filtered_spikes_df = spk_df.spikes.time_sliced(position.t_start, position.t_stop)
            self._filtered_pos_df = pos_df.position.time_sliced(position.t_start, position.t_stop)
        
        # drop positions with either X or Y NA values:
        
        if (self.ndim > 1):
            pos_non_NA_column_labels = ['x','y']
        else:
            pos_non_NA_column_labels = ['x']

        self._filtered_pos_df.dropna(axis=0, how='any', subset=pos_non_NA_column_labels, inplace=True) # dropped NaN values
            
        # Set animal observed position member variables:
        # self.t = self.filtered_pos_df.t.to_numpy()
        # self.x = self.filtered_pos_df.x.to_numpy()
        # self.speed = self.filtered_pos_df.speed.to_numpy()
        if (self.should_smooth_speed and (self.config.smooth is not None) and (self.config.smooth[0] > 0.0)):
            # self.speed = gaussian_filter1d(self.speed, sigma=self.config.smooth[0])
            self._filtered_pos_df['speed_smooth'] = gaussian_filter1d(self._filtered_pos_df.speed.to_numpy(), sigma=self.config.smooth[0])
            
        # if (self.ndim > 1):
        #     self.y = self.filtered_pos_df.y.to_numpy()
        # else:
        #     self.y = None

        # Add interpolated velocity information to spikes dataframe:
        if 'speed' not in self._filtered_spikes_df.columns:
            self._filtered_spikes_df['speed'] = np.interp(self._filtered_spikes_df[spikes_df.spikes.time_variable_name].to_numpy(), self.filtered_pos_df.t.to_numpy(), self.speed)
    
        
        # Filter for speed:
        if debug_print:
            print(f'pre speed filtering: {np.shape(self._filtered_spikes_df)[0]} spikes.')
        self._filtered_spikes_df = self._filtered_spikes_df[self._filtered_spikes_df['speed'] > self.config.speed_thresh]
        if debug_print:
            print(f'post speed filtering: {np.shape(self._filtered_spikes_df)[0]} spikes.')
        
        ## Binning with Fixed Number of Bins:    
        # xbin, ybin, bin_info = PfND._bin_pos_nD(self.x, self.y, num_bins=grid_num_bins) # num_bins mode:
        # self.xbin, self.ybin, self.bin_info = PfND._bin_pos_nD(self.x, self.y, bin_size=self.config.grid_bin) # bin_size mode
        if (self.ndim > 1):
            self.xbin, self.ybin, self.bin_info = PfND._bin_pos_nD(self.filtered_pos_df.x.to_numpy(), self.filtered_pos_df.y.to_numpy(), bin_size=self.config.grid_bin) # bin_size mode            
        else:
            self.xbin, self.ybin, self.bin_info = PfND._bin_pos_nD(self.filtered_pos_df.x.to_numpy(), None, bin_size=self.config.grid_bin) # bin_size mode

        # Adds the 'binned_x' and 'binned_y' columns to the position dataframe:
        # self._filtered_pos_df.position.build_discretized_binned_positions(self.config, xbin_values=self.xbin, ybin_values=self.ybin, debug_print=False)
        
        if 'binned_x' not in self._filtered_pos_df.columns:
            self._filtered_pos_df, _, _, _ = PfND.build_position_df_discretized_binned_positions(self._filtered_pos_df, self.config, xbin_values=self.xbin, ybin_values=self.ybin, debug_print=False)
   
   
    def compute(self):
        """ actually compute the placefields after self.setup(...) is complete.
        
        Assigns:
        
            self.ratemap
            self.ratemap_spiketrains
            self.ratemap_spiketrains_pos
            
            self._included_thresh_neurons_indx
            self._peak_frate_filter_function
            
        """
        # --- occupancy map calculation -----------
        if not self.should_smooth_spatial_occupancy_map:
            smooth_occupancy_map = (0.0, 0.0)
        else:
            smooth_occupancy_map = self.config.smooth
        if (self.ndim > 1):
            occupancy, xedges, yedges = Pf2D._compute_occupancy(self.x, self.y, self.xbin, self.ybin, self.position_srate, smooth_occupancy_map)
        else:
            occupancy, xedges = Pf1D._compute_occupancy(self.x, self.xbin, self.position_srate, smooth_occupancy_map[0])
        
        # Output lists, for compatibility with Pf1D and Pf2D:
        spk_pos, spk_t, firing_maps, tuning_maps = [], [], [], []
        
        # Once filtering and binning is done, apply the grouping:
        # Group by the aclu (cluster indicator) column
        cell_grouped_spikes_df = self.filtered_spikes_df.groupby(['aclu'])
        cell_spikes_dfs = [cell_grouped_spikes_df.get_group(a_neuron_id) for a_neuron_id in self.filtered_spikes_df.spikes.neuron_ids] # a list of dataframes for each neuron_id

        # NOTE: regardless of whether should_smooth_final_tuning_map is true or not, we must pass in the actual smooth value to the _compute_tuning_map(...) function so it can choose to filter its firing map or not. Only if should_smooth_final_tuning_map is enabled will the final product be smoothed.
            
        # re-interpolate given the updated spks
        for cell_df in cell_spikes_dfs:
            # cell_spike_times = cell_df[spikes_df.spikes.time_variable_name].to_numpy()
            cell_spike_times = cell_df[self.filtered_spikes_df.spikes.time_variable_name].to_numpy()
            # spk_spd = np.interp(cell_spike_times, self.t, self.speed)
            spk_x = np.interp(cell_spike_times, self.t, self.x)
            
            # update the dataframe 'x','speed' and 'y' properties:
            # cell_df.loc[:, 'x'] = spk_x
            # cell_df.loc[:, 'speed'] = spk_spd
            if (self.ndim > 1):
                spk_y = np.interp(cell_spike_times, self.t, self.y)
                # cell_df.loc[:, 'y'] = spk_y
                spk_pos.append([spk_x, spk_y])
                # TODO: Make "firing maps" before "tuning maps"
                # raw_tuning_maps = np.asarray([Pf2D._compute_tuning_map(neuron_split_spike_dfs[i].x.to_numpy(), neuron_split_spike_dfs[i].y.to_numpy(), xbin, ybin, occupancy, None, should_return_raw_tuning_map=True) for i in np.arange(len(neuron_split_spike_dfs))]) # dataframes split for each ID:
                # tuning_maps = np.asarray([raw_tuning_maps[i] / occupancy for i in np.arange(len(raw_tuning_maps))])
                # ratemap = Ratemap(tuning_maps, xbin=xbin, ybin=ybin, neuron_ids=active_epoch_session.neuron_ids)
                curr_cell_tuning_map, curr_cell_firing_map = Pf2D._compute_tuning_map(spk_x, spk_y, self.xbin, self.ybin, occupancy, self.config.smooth, should_also_return_intermediate_firing_map=self._save_intermediate_firing_maps)
            else:
                # otherwise only 1D:
                spk_pos.append([spk_x])
                curr_cell_tuning_map, curr_cell_firing_map = Pf1D._compute_tuning_map(spk_x, self.xbin, occupancy, self.config.smooth[0], should_also_return_intermediate_firing_map=self._save_intermediate_firing_maps)
            
            spk_t.append(cell_spike_times)
            # tuning curve calculation:               
            tuning_maps.append(curr_cell_tuning_map)
            firing_maps.append(curr_cell_firing_map)
                
        # ---- cells with peak frate abouve thresh 
        self._included_thresh_neurons_indx, self._peak_frate_filter_function = PfND._build_peak_frate_filter(tuning_maps, self.config.frate_thresh)
        
        # there is only one tuning_map per neuron that means the thresh_neurons_indx:
        filtered_tuning_maps = np.asarray(self._peak_frate_filter_function(tuning_maps.copy()))
        filtered_firing_maps = self._peak_frate_filter_function(firing_maps.copy())
        filtered_neuron_ids = self._peak_frate_filter_function(self.filtered_spikes_df.spikes.neuron_ids)        
        filtered_tuple_neuron_ids = self._peak_frate_filter_function(self.filtered_spikes_df.spikes.neuron_probe_tuple_ids) # the (shank, probe) tuples corresponding to neuron_ids
        
        self.ratemap = Ratemap(filtered_tuning_maps, firing_maps=filtered_firing_maps, xbin=self.xbin, ybin=self.ybin, neuron_ids=filtered_neuron_ids, occupancy=occupancy, neuron_extended_ids=filtered_tuple_neuron_ids)
        self.ratemap_spiketrains = self._peak_frate_filter_function(spk_t)
        self.ratemap_spiketrains_pos = self._peak_frate_filter_function(spk_pos)
        
           
    @property
    def t(self):
        """The position timestamps property."""
        return self.filtered_pos_df.t.to_numpy()

    @property
    def x(self):
        """The position timestamps property."""
        return self.filtered_pos_df.x.to_numpy()

    @property
    def y(self):
        """The position timestamps property."""
        if (self.ndim > 1):
            return self.filtered_pos_df.y.to_numpy()
        else:
            return None
    @property
    def speed(self):
        """The position timestamps property."""
        if (self.should_smooth_speed and (self.config.smooth is not None) and (self.config.smooth[0] > 0.0)):
            return self.filtered_pos_df.speed_smooth.to_numpy()
        else:
            return self.filtered_pos_df.speed.to_numpy()
        
    @property
    def filtered_spikes_df(self):
        """The filtered_spikes_df property."""
        return self._filtered_spikes_df
    @filtered_spikes_df.setter
    def filtered_spikes_df(self, value):
        self._filtered_spikes_df = value
        
    @property
    def filtered_pos_df(self):
        """The filtered_pos_df property."""
        return self._filtered_pos_df
    @filtered_pos_df.setter
    def filtered_pos_df(self, value):
        self._filtered_pos_df = value
        

    ## ratemap convinence accessors
    @property
    def occupancy(self):
        """The occupancy property."""
        return self.ratemap.occupancy
    @occupancy.setter
    def occupancy(self, value):
        self.ratemap.occupancy = value
    @property
    def neuron_extended_ids(self):
        """The neuron_extended_ids property."""
        return self.ratemap.neuron_extended_ids
    @neuron_extended_ids.setter
    def neuron_extended_ids(self, value):
        self.ratemap.neuron_extended_ids = value
    
    ## self.config convinence accessors. Mostly for compatibility with Pf1D and Pf2D
    @property
    def frate_thresh(self):
        """The frate_thresh property."""
        return self.config.frate_thresh
    @property
    def speed_thresh(self):
        """The speed_thresh property."""
        return self.config.speed_thresh
    
    @property
    def frate_filter_fcn(self):
        """The frate_filter_fcn property."""
        return self._peak_frate_filter_function
    
    @property
    def included_neuron_IDXs(self):
        """The neuron INDEXES, NOT IDs (not 'aclu' values) that were included after filtering by frate and etc. """
        return self._included_thresh_neurons_indx
    
    @property
    def included_neuron_IDs(self):
        """The neuron IDs ('aclu' values) that were included after filtering by frate and etc. """
        return self._filtered_spikes_df.spikes.neuron_ids[self.included_neuron_IDXs]
    
    
    
    def str_for_filename(self, prefix_string=''):
        if self.ndim <= 1:
            return '-'.join(['pf1D', f'{prefix_string}{self.config.str_for_filename(False)}'])
        else:
            return '-'.join(['pf2D', f'{prefix_string}{self.config.str_for_filename(True)}'])
    
    def str_for_display(self, prefix_string=''):
        if self.ndim <= 1:
            return '-'.join(['pf1D', f'{prefix_string}{self.config.str_for_display(False)}', f'cell_{curr_cell_id:02d}'])
        else:
            return '-'.join(['pf2D', f'{prefix_string}{self.config.str_for_display(True)}', f'cell_{curr_cell_id:02d}'])
        

    @staticmethod
    def _build_peak_frate_filter(tuning_maps, frate_thresh, debug_print=False):
        """ Finds the peak value of the tuning map for each cell and compares it to the frate_thresh to see if it should be included.
        
        Returns:
            thresh_neurons_indx: the list of indicies that meet the peak firing rate threshold critiera
            filter_function: a function that takes any list of length n_neurons (original number of neurons) and just indexes its passed list argument by thresh_neurons_indx (including only neurons that meet the thresholding criteria)
        """
        # ---- cells with peak frate abouve thresh ------
        n_neurons = len(tuning_maps)
        # only include the indicies that have a max firing rate greater than frate_thresh
        included_thresh_neurons_indx = [
            neuron_indx
            for neuron_indx in range(n_neurons)
            if np.nanmax(tuning_maps[neuron_indx]) > frate_thresh
        ]
        if debug_print:
            print('_build_peak_frate_filter(...):')
            print('\t frate_thresh: {}'.format(frate_thresh))
            print('\t n_neurons: {}'.format(n_neurons))
            print('\t thresh_neurons_indx: {}'.format(included_thresh_neurons_indx))
        # filter_function: just indexes its passed list argument by thresh_neurons_indx (including only neurons that meet the thresholding criteria)
        filter_function = lambda list_: [list_[_] for _ in included_thresh_neurons_indx] # filter_function: takes any list of length n_neurons (original number of neurons) and returns only the elements that met the firing rate criteria
        
        return included_thresh_neurons_indx, filter_function 

    @staticmethod
    def _bin_pos_nD(x: np.ndarray, y: np.ndarray, num_bins=None, bin_size=None):
        """ Spatially bins the provided x and y vectors into position bins based on either the specified num_bins or the specified bin_size
        Usage:
            ## Binning with Fixed Number of Bins:    
            xbin, ybin, bin_info = _bin_pos(pos_df.x.to_numpy(), pos_df.y.to_numpy(), bin_size=active_config.computation_config.grid_bin) # bin_size mode
            print(bin_info)
            ## Binning with Fixed Bin Sizes:
            xbin, ybin, bin_info = _bin_pos(pos_df.x.to_numpy(), pos_df.y.to_numpy(), num_bins=num_bins) # num_bins mode
            print(bin_info)
            
            
        TODO: 2022-04-22 - Note that I discovered that the bins generated here might cause an error when used with Pandas .cut function, which does not include the left (most minimum) values by default. This would cause the minimumal values not to be included.
        """
        assert (num_bins is None) or (bin_size is None), 'You cannot constrain both num_bins AND bin_size. Specify only one or the other.'
        assert (num_bins is not None) or (bin_size is not None), 'You must specify either the num_bins XOR the bin_size.'
        
        bin_info_out_dict = dict()
        
        if num_bins is not None:
            ## Binning with Fixed Number of Bins:
            mode = 'num_bins'
            if np.isscalar(num_bins):
                num_bins = [num_bins]
            
            xnum_bins = num_bins[0]
            xbin, xstep = np.linspace(np.nanmin(x), np.nanmax(x), num=xnum_bins, retstep=True)  # binning of x position

            if y is not None:
                ynum_bins = num_bins[1]
                ybin, ystep = np.linspace(np.nanmin(y), np.nanmax(y), num=ynum_bins, retstep=True)  # binning of y position       
                
        elif bin_size is not None:
            ## Binning with Fixed Bin Sizes:
            mode = 'bin_size'
            if np.isscalar(bin_size):
                print(f'np.isscalar(bin_size): {bin_size}')
                bin_size = [bin_size]
                
            xstep = bin_size[0]
            xbin = np.arange(np.nanmin(x), (np.nanmax(x) + xstep), xstep)  # binning of x position
            xnum_bins = len(xbin)

            if y is not None:
                ystep = bin_size[1]
                ybin = np.arange(np.nanmin(y), (np.nanmax(y) + ystep), ystep)  # binning of y position
                ynum_bins = len(ybin)
                
        # print('xbin: {}'.format(xbin))
        # print('ybin: {}'.format(ybin))
        bin_info_out_dict = {'mode':mode, 'xstep':xstep, 'xnum_bins':xnum_bins}
        if y is not None:
            # if at least 2D output, add the y-axis properties to the info dictionary
            bin_info_out_dict['ystep'], bin_info_out_dict['ynum_bins']  = ystep, ynum_bins
        else:
            ybin = None
            
        return xbin, ybin, bin_info_out_dict # {'mode':mode, 'xstep':xstep, 'ystep':ystep, 'xnum_bins':xnum_bins, 'ynum_bins':ynum_bins}


    ## Binned Position Columns:
    @staticmethod
    def build_position_df_discretized_binned_positions(active_pos_df, active_computation_config, xbin_values=None, ybin_values=None, debug_print=False):
        """ Adds the 'binned_x' and 'binned_y' columns to the position dataframe """
        # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within.
        if (xbin_values is None) or (ybin_values is None):
            # determine the correct bins to use from active_computation_config.grid_bin:
            if debug_print:
                print(f'active_grid_bin: {active_computation_config.grid_bin}')
                
            if 'y' in active_pos_df.columns:
                xbin, ybin, bin_info = PfND._bin_pos_nD(active_pos_df['x'].values, active_pos_df['y'].values, bin_size=active_computation_config.grid_bin) # bin_size mode
            else:
                # 1D case:
                xbin, ybin, bin_info = PfND._bin_pos_nD(active_pos_df['x'].values, None, bin_size=active_computation_config.grid_bin) # bin_size mode
        else:
            # use the extant values passed in:
            if debug_print:
                print(f'using extant bins passed as arguments: xbin_values.shape: {xbin_values.shape}, ybin_values.shape: {ybin_values.shape}')
            xbin = xbin_values
            ybin = ybin_values
            bin_info = None
        
        active_pos_df['binned_x'] = pd.cut(active_pos_df['x'].to_numpy(), bins=xbin, include_lowest=True, labels=np.arange(start=1, stop=len(xbin))) # same shape as the input data 
        if 'y' in active_pos_df.columns:
            # Only do the y-variables in the 2D case.
            active_pos_df['binned_y'] = pd.cut(active_pos_df['y'].to_numpy(), bins=ybin, include_lowest=True, labels=np.arange(start=1, stop=len(ybin))) 
    
        return active_pos_df, xbin, ybin, bin_info




### Global Placefield Computation Functions
""" Global Placefield perform Computation Functions """

def perform_compute_placefields(active_session_spikes_df, active_pos, computation_config: PlacefieldComputationParameters, active_epoch_placefields1D=None, active_epoch_placefields2D=None, included_epochs=None, should_force_recompute_placefields=True):
    """ Most general computation function. Computes both 1D and 2D placefields.
    active_epoch_session_Neurons: 
    active_epoch_pos: a Position object
    included_epochs: a Epoch object to filter with, only included epochs are included in the PF calculations
    active_epoch_placefields1D (Pf1D, optional) & active_epoch_placefields2D (Pf2D, optional): allow you to pass already computed Pf1D and Pf2D objects from previous runs and it won't recompute them so long as should_force_recompute_placefields=False, which is useful in interactive Notebooks/scripts
    Usage:
        active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_epoch_session_Neurons, active_epoch_pos, active_epoch_placefields1D, active_epoch_placefields2D, active_config.computation_config, should_force_recompute_placefields=True)
    """
    ## Linearized (1D) Position Placefields:
    if ((active_epoch_placefields1D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_placefields...', end=' ')
        # PfND version:
        active_epoch_placefields1D = PfND(deepcopy(active_session_spikes_df), deepcopy(active_pos.linear_pos_obj), epochs=included_epochs,
                                          speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                          grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)

        print('\t done.')
    else:
        print('active_epoch_placefields1D already exists, reusing it.')

    ## 2D Position Placemaps:
    if ((active_epoch_placefields2D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_placefields2D...', end=' ')
        # PfND version:
        active_epoch_placefields2D = PfND(deepcopy(active_session_spikes_df), deepcopy(active_pos), epochs=included_epochs,
                                          speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                          grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)

        print('\t done.')
    else:
        print('active_epoch_placefields2D already exists, reusing it.')
    
    return active_epoch_placefields1D, active_epoch_placefields2D

def compute_placefields_masked_by_epochs(sess, active_config, included_epochs=None, should_display_2D_plots=False):
    """ Wrapps perform_compute_placefields to make the call simpler """
    active_session = deepcopy(sess)
    active_epoch_placefields1D, active_epoch_placefields2D = compute_placefields_as_needed(active_session, active_config.computation_config, active_config, None, None, included_epochs=included_epochs, should_force_recompute_placefields=True, should_display_2D_plots=should_display_2D_plots)
    # Focus on the 2D placefields:
    # active_epoch_placefields = active_epoch_placefields2D
    # Get the updated session using the units that have good placefields
    # active_session, active_config, good_placefield_neuronIDs = process_by_good_placefields(active_session, active_config, active_epoch_placefields)
    # debug_print_spike_counts(active_session)
    return active_epoch_placefields1D, active_epoch_placefields2D


def compute_placefields_as_needed(active_session, computation_config:PlacefieldComputationParameters=None, general_config=None, active_placefields1D = None, active_placefields2D = None, included_epochs=None, should_force_recompute_placefields=True, should_display_2D_plots=False):
    from neuropy.plotting.placemaps import plot_all_placefields
    
    if computation_config is None:
        computation_config = PlacefieldComputationParameters(speed_thresh=9, grid_bin=2, smooth=0.5)
    # active_placefields1D, active_placefields2D = perform_compute_placefields(active_session.neurons, active_session.position, computation_config, active_placefields1D, active_placefields2D, included_epochs=included_epochs, should_force_recompute_placefields=True)
    active_placefields1D, active_placefields2D = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, active_placefields1D, active_placefields2D, included_epochs=included_epochs, should_force_recompute_placefields=should_force_recompute_placefields)
    # Plot the placefields computed and save them out to files:
    if should_display_2D_plots:
        ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(active_placefields1D, active_placefields2D, general_config)
    else:
        print('skipping 2D placefield plots')
    return active_placefields1D, active_placefields2D

