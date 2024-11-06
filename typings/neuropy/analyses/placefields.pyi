"""
This type stub file was generated by pyright.
"""

import neuropy.utils.type_aliases as types
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
from attrs import define
from nptyping import NDArray
from neuropy.core.epoch import Epoch
from neuropy.core.position import Position
from neuropy.core.ratemap import Ratemap
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin
from neuropy.utils.mixins.HDF5_representable import HDFMixin
from neuropy.plotting.mixins.placemap_mixins import PfnDPlottingMixin
from neuropy.utils.mixins.binning_helpers import BinnedPositionsMixin
from neuropy.utils.mixins.diffable import DiffableObject
from neuropy.utils.mixins.dict_representable import DictInitializable, DictlikeOverridableMixin, SubsettableDictRepresentable
from neuropy.utils.debug_helpers import safely_accepts_kwargs
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
from neuropy.utils.mixins.peak_location_representing import ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin
from neuropy.utils.mixins.gettable_mixin import KeypathsAccessibleMixin
from neuropy.utils.mixins.print_helpers import OrderedMeta, SimplePrintable

custom_formatting_dict: Dict[str, Callable] = ...
custom_skip_formatting_display_list: List[str] = ...
class PlacefieldComputationParameters(SimplePrintable, KeypathsAccessibleMixin, SubsettableDictRepresentable, DictlikeOverridableMixin, DictInitializable, DiffableObject, metaclass=OrderedMeta):
	"""A simple wrapper object for parameters used in placefield calcuations
	
	#TODO 2023-07-30 18:18: - [ ] HDFMixin conformance for PlacefieldComputationParameters
	
	"""
	decimal_point_character = ...
	param_sep_char = ...
	variable_names = ...
	variable_inline_names = ...
	variable_inline_names = ...
	float_precision: int = ...
	array_items_threshold: int = ...
	def __init__(self, speed_thresh=..., grid_bin=..., grid_bin_bounds=..., smooth=..., frate_thresh=..., is_directional=..., **kwargs) -> None:
		...
	
	@property
	def grid_bin_1D(self): # -> generic | bool | int | float | complex | str | bytes | memoryview[int]:
		"""The grid_bin_1D property."""
		...
	
	@property
	def grid_bin_bounds_1D(self): # -> generic | bool | int | float | complex | str | bytes | memoryview[int] | None:
		"""The grid_bin_bounds property."""
		...
	
	@property
	def smooth_1D(self): # -> generic | bool | int | float | complex | str | bytes | memoryview[int]:
		"""The smooth_1D property."""
		...
	
	def str_for_filename(self, is_2D: bool): # -> str:
		...
	
	def str_for_display(self, is_2D: bool, extras_join_sep: str = ..., normal_to_extras_line_sep: str = ...): # -> str:
		""" For rendering in a title, etc
		normal_to_extras_line_sep: the separator between the normal and extras lines
		
		#TODO 2023-07-21 16:35: - [ ] The np.printoptions doesn't affect the values that are returned from `extras_string = ', '.join(self._unlisted_parameter_strings())`
		We end up with '(speedThresh_10.00, gridBin_2.00, smooth_2.00, frateThresh_1.00)grid_bin_bounds_((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))' (too many sig-figs on the output grid_bin_bounds)
		
		"""
		...
	
	def str_for_attributes_list_display(self, param_sep_char=..., key_val_sep_char=..., subset_includelist: Optional[list] = ..., subset_excludelist: Optional[list] = ..., override_float_precision: Optional[int] = ..., override_array_items_threshold: Optional[int] = ...): # -> str:
		""" For rendering in attributes list like outputs
		# Default for attributes lists outputs:
		Example Output:
			speed_thresh	2.0
			grid_bin	[3.777 1.043]
			smooth	[1.5 1.5]
			frate_thresh	0.1
			time_bin_size	0.5
		"""
		...
	
	def __hash__(self) -> int:
		""" custom hash function that allows use in dictionary just based off of the values and not the object instance. """
		...
	
	def __eq__(self, other) -> bool:
		"""Overrides the default implementation to allow comparing by value. """
		...
	
	@classmethod
	def compute_grid_bin_bounds(cls, x, y): # -> tuple[Any, ...]:
		...
	


class PfnConfigMixin:
	def str_for_filename(self, is_2D=...):
		...
	


class PfnDMixin(SimplePrintable):
	should_smooth_speed = ...
	should_smooth_spikes_map = ...
	should_smooth_spatial_occupancy_map = ...
	should_smooth_final_tuning_map = ...
	@property
	def spk_pos(self):
		...
	
	@property
	def spk_t(self):
		...
	
	@property
	def cell_ids(self):
		...
	
	@safely_accepts_kwargs
	def plot_raw(self, subplots=..., fignum=..., alpha=..., label_cells=..., ax=..., clus_use=...): # -> Figure | None:
		""" Plots the Placefield raw spiking activity for all cells"""
		...
	
	@safely_accepts_kwargs
	def plotRaw_v_time(self, cellind, speed_thresh=..., spikes_color=..., spikes_alpha=..., ax=..., position_plot_kwargs=..., spike_plot_kwargs=..., should_include_trajectory=..., should_include_spikes=..., should_include_filter_excluded_spikes=..., should_include_labels=..., use_filtered_positions=..., use_pandas_plotting=...):
		""" Builds one subplot for each dimension of the position data
		Updated to work with both 1D and 2D Placefields

		should_include_trajectory:bool - if False, will not try to plot the animal's trajectory/position
			NOTE: Draws the spike_positions actually instead of the continuously sampled animal position

		should_include_labels:bool - whether the plot should include text labels, like the title, axes labels, etc
		should_include_spikes:bool - if False, will not try to plot points for spikes
		use_pandas_plotting:bool = False
		use_filtered_positions:bool = False # If True, uses only the filtered positions (which are missing the end caps) and the default a.plot(...) results in connected lines which look bad.

		"""
		...
	
	@safely_accepts_kwargs
	def plot_all(self, cellind, speed_thresh=..., spikes_color=..., spikes_alpha=..., fig=...): # -> Figure:
		...
	


class Pf1D(PfnConfigMixin, PfnDMixin):
	...


class Pf2D(PfnConfigMixin, PfnDMixin):
	...


class PlacefieldND(PfnConfigMixin, PfnDMixin):
	""" 2023-11-10 ChatGPT-3 Generalized Implementation, UNTESTED
	# 1D:
	def _compute_occupancy(x, xbin, position_srate, smooth, should_return_num_pos_samples_occupancy=False)
	def _compute_spikes_map(spk_x, xbin, smooth)
	def _compute_tuning_map(spk_x, xbin, occupancy, smooth, should_also_return_intermediate_spikes_map=False)

	# 2D:
	def _compute_occupancy(x, y, xbin, ybin, position_srate, smooth, should_return_num_pos_samples_occupancy=False):
	def _compute_spikes_map(spk_x, spk_y, xbin, ybin, smooth)
	def _compute_tuning_map(spk_x, spk_y, xbin, ybin, occupancy, smooth, should_also_return_intermediate_spikes_map=False)

	# 3D:
	position_args: x, y, z
	bin_args: xbin, ybin, zbin
	spike_args: spk_x, spk_y, spk_z
	def _compute_occupancy(x, y, z, xbin, ybin, zbin, position_srate, smooth, should_return_num_pos_samples_occupancy=False):
	def _compute_spikes_map(spk_x, spk_y, spk_z, xbin, ybin, zbin, smooth)
	def _compute_tuning_map(spk_x, spk_y, spk_z, xbin, ybin, zbin, occupancy, smooth, should_also_return_intermediate_spikes_map=False)


	# N-D
	position_args: x, y, z
	bin_args: xbin, ybin, zbin
	spike_args: spk_x, spk_y, spk_z
	def _compute_occupancy(x, y, z, xbin, ybin, zbin, position_srate, smooth, should_return_num_pos_samples_occupancy=False):
	def _compute_spikes_map(spk_x, spk_y, spk_z, xbin, ybin, zbin, smooth)
	def _compute_tuning_map(spk_x, spk_y, spk_z, xbin, ybin, zbin, occupancy, smooth, should_also_return_intermediate_spikes_map=False)


	from neuropy.analyses.placefields import Pf1D, Pf2D, PlacefieldND


	"""
	...


@define(slots=False)
class PfND(HDFMixin, AttrsBasedClassHelperMixin, ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin, NeuronUnitSlicableObjectProtocol, BinnedPositionsMixin, PfnConfigMixin, PfnDMixin, PfnDPlottingMixin):
	"""Represents a collection of placefields over binned,  N-dimensional space. 

		It always computes two place maps with and without speed thresholds.

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


		# Excluded from serialization: ['_included_thresh_neurons_indx', '_peak_frate_filter_function']
	"""
	spikes_df: pd.DataFrame
	position: Position
	epochs: Epoch = ...
	config: PlacefieldComputationParameters = ...
	position_srate: float = ...
	setup_on_init: bool = ...
	compute_on_init: bool = ...
	_save_intermediate_spikes_maps: bool = ...
	_included_thresh_neurons_indx: np.ndarray = ...
	_peak_frate_filter_function: Callable = ...
	_ratemap: Ratemap = ...
	_ratemap_spiketrains: list = ...
	_ratemap_spiketrains_pos: list = ...
	_filtered_pos_df: pd.DataFrame = ...
	_filtered_spikes_df: pd.DataFrame = ...
	ndim: int = ...
	xbin: np.ndarray = ...
	ybin: np.ndarray = ...
	bin_info: dict = ...
	def __attrs_post_init__(self): # -> None:
		""" called after initializer built by `attrs` library. """
		...
	
	@classmethod
	def from_config_values(cls, spikes_df: pd.DataFrame, position: Position, epochs: Epoch = ..., frate_thresh=..., speed_thresh=..., grid_bin=..., grid_bin_bounds=..., smooth=..., setup_on_init: bool = ..., compute_on_init: bool = ...): # -> Self:
		""" initialize from the explicitly listed arguments instead of a specified config. """
		...
	
	def setup(self, position: Position, spikes_df, epochs: Epoch, debug_print=...): # -> None:
		""" do the preliminary setup required to build the placefields

		Adds columns to the spikes and positions dataframes, etc.

		Depends on:
			self.config.smooth
			self.config.grid_bin_bounds

		Assigns:
			self.ndim
			self._filtered_pos_df
			self._filtered_spikes_df

			self.xbin, self.ybin, self.bin_info
		"""
		...
	
	def compute(self): # -> None:
		""" actually compute the placefields after self.setup(...) is complete.


		Depends on:
			self.config.smooth
			self.x, self.y, self.xbin, self.ybin, self.position_srate


		Assigns:

			self.ratemap
			self.ratemap_spiketrains
			self.ratemap_spiketrains_pos

			self._included_thresh_neurons_indx
			self._peak_frate_filter_function

		"""
		...
	
	@property
	def PeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
		""" the variable that the peaks are calculated and returned for """
		...
	
	@property
	def ContinuousPeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
		""" the variable that the peaks are calculated and returned for """
		...
	
	@property
	def t(self) -> NDArray:
		"""The position timestamps property."""
		...
	
	@property
	def x(self) -> NDArray:
		"""The position timestamps property."""
		...
	
	@property
	def y(self) -> Optional[NDArray]:
		"""The position timestamps property."""
		...
	
	@property
	def speed(self) -> NDArray:
		"""The position timestamps property."""
		...
	
	@property
	def xbin_centers(self): # -> NDArray[floating[Any]]:
		...
	
	@property
	def ybin_centers(self): # -> NDArray[floating[Any]]:
		...
	
	@property
	def filtered_spikes_df(self): # -> DataFrame[Any]:
		"""The filtered_spikes_df property."""
		...
	
	@filtered_spikes_df.setter
	def filtered_spikes_df(self, value): # -> None:
		...
	
	@property
	def filtered_pos_df(self): # -> DataFrame[Any]:
		"""The filtered_pos_df property."""
		...
	
	@filtered_pos_df.setter
	def filtered_pos_df(self, value): # -> None:
		...
	
	@property
	def ratemap(self): # -> Ratemap:
		"""The ratemap property."""
		...
	
	@ratemap.setter
	def ratemap(self, value): # -> None:
		...
	
	@property
	def ratemap_spiketrains(self): # -> list[Any]:
		"""The ratemap_spiketrains property."""
		...
	
	@ratemap_spiketrains.setter
	def ratemap_spiketrains(self, value): # -> None:
		...
	
	@property
	def ratemap_spiketrains_pos(self): # -> list[Any]:
		"""The ratemap_spiketrains_pos property."""
		...
	
	@ratemap_spiketrains_pos.setter
	def ratemap_spiketrains_pos(self, value): # -> None:
		...
	
	@property
	def occupancy(self): # -> None:
		"""The occupancy property."""
		...
	
	@occupancy.setter
	def occupancy(self, value): # -> None:
		...
	
	@property
	def never_visited_occupancy_mask(self): # -> NDArray:
		...
	
	@property
	def nan_never_visited_occupancy(self): # -> NDArray:
		...
	
	@property
	def probability_normalized_occupancy(self) -> NDArray:
		...
	
	@property
	def visited_occupancy_mask(self) -> NDArray:
		...
	
	@property
	def neuron_extended_ids(self):
		"""The neuron_extended_ids property."""
		...
	
	@neuron_extended_ids.setter
	def neuron_extended_ids(self, value): # -> None:
		...
	
	@property
	def tuning_curves_dict(self) -> Dict[types.aclu_index, NDArray]:
		""" aclu:tuning_curve_array """
		...
	
	@property
	def normalized_tuning_curves_dict(self) -> Dict[types.aclu_index, NDArray]:
		""" aclu:tuning_curve_array """
		...
	
	@property
	def frate_thresh(self): # -> int:
		"""The frate_thresh property."""
		...
	
	@property
	def speed_thresh(self): # -> int:
		"""The speed_thresh property."""
		...
	
	@property
	def pos_bin_size(self) -> Union[float, Tuple[float, float]]:
		""" extracts pos_bin_size: the size of the x_bin in [cm], from the decoder. 
		
		returns a tuple if 2D or a single float if 1D

		"""
		...
	
	@property
	def frate_filter_fcn(self): # -> Callable[..., Any]:
		"""The frate_filter_fcn property."""
		...
	
	@property
	def included_neuron_IDXs(self): # -> ndarray[Any, Any]:
		"""The neuron INDEXES, NOT IDs (not 'aclu' values) that were included after filtering by frate and etc. """
		...
	
	@property
	def included_neuron_IDs(self):
		"""The neuron IDs ('aclu' values) that were included after filtering by frate and etc. """
		...
	
	def get_by_id(self, ids) -> PfND:
		"""Implementors return a copy of themselves with neuron_ids equal to ids
			Needs to update: copy_pf._filtered_spikes_df, copy_pf.ratemap, copy_pf.ratemap_spiketrains, copy_pf.ratemap_spiketrains_pos, 
		"""
		...
	
	def replacing_computation_epochs(self, epochs: Union[Epoch, pd.DataFrame]) -> PfND:
		"""Implementors return a copy of themselves with their computation epochs replaced by the provided ones. The existing epochs are unrelated and do not need to be related to the new ones.
		"""
		...
	
	def conform_to_position_bins(self, target_pf, force_recompute=...): # -> tuple[Self, bool]:
		""" Allow overriding PfND's bins:
			# 2022-12-09 - We want to be able to have both long/short track placefields have the same spatial bins.
			This function standardizes the short pf1D's xbins to the same ones as the long_pf1D, and then recalculates it.
			Usage:
				short_pf1D, did_update_bins = short_pf1D.conform_to_position_bins(long_pf1D)
		"""
		...
	
	def to_1D_maximum_projection(self) -> PfND:
		...
	
	@classmethod
	def build_1D_maximum_projection(cls, pf2D: PfND) -> PfND:
		""" builds a 1D ratemap from a 2D ratemap
		creation_date='2023-04-05 14:02'

		Usage:
			ratemap_1D = build_1D_maximum_projection(ratemap_2D)
		"""
		...
	
	@classmethod
	def filtered_by_speed(cls, epochs_df: pd.DataFrame, position_df: pd.DataFrame, speed_thresh: Optional[float], speed_column_override_name: Optional[str] = ..., debug_print: bool = ...): # -> DataFrame[Any]:
		""" Filters the position_df by speed and epoch_df correctly, so we can get actual occupancy.
		2023-11-14 - 
		
		
		speed_thresh = a_decoder.config.speed_thresh # 10.0
		position_df = a_decoder.position.to_dataframe()
		
		
		"""
		...
	
	def str_for_filename(self, prefix_string=...): # -> str:
		...
	
	def str_for_display(self, prefix_string=...): # -> str:
		...
	
	def to_dict(self): # -> Dict[str, Any]:
		...
	
	def __getstate__(self): # -> Dict[str, Any]:
		...
	
	def __setstate__(self, state): # -> None:
		""" assumes state is a dict generated by calling self.__getstate__() previously"""
		...
	
	@staticmethod
	def build_position_df_discretized_binned_positions(active_pos_df, active_computation_config, xbin_values=..., ybin_values=..., debug_print=...): # -> tuple[Any, Any, Any | None, dict[str, Any]]:
		""" Adds the 'binned_x' and 'binned_y' columns to the position dataframe

		Assumes either 1D or 2D positions dependent on whether the 'y' column exists in active_pos_df.columns.
		Wraps the build_df_discretized_binned_position_columns and appropriately unwraps the result for compatibility with previous implementations.

		"""
		...
	
	def to_hdf(self, file_path, key: str, **kwargs): # -> None:
		""" Saves the object to key in the hdf5 file specified by file_path
		Usage:
			hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
			_pfnd_obj: PfND = long_one_step_decoder_1D.pf
			_pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
		"""
		...
	
	@classmethod
	def read_hdf(cls, file_path, key: str, **kwargs) -> PfND:
		""" Reads the data from the key in the hdf5 file at file_path
		Usage:
			_reread_pfnd_obj = PfND.read_hdf(hdf5_output_path, key='test_pfnd')
			_reread_pfnd_obj
		"""
		...
	
	@classmethod
	def build_pseduo_2D_directional_placefield_positions(cls, *directional_1D_decoder_list) -> Position:
		""" 2023-11-10 - builds the positions for the directional 1D decoders into a pseudo 2D decoder
		## HACK: this adds the two directions of two separate 1D placefields into a stack with a pseudo-y dimension (with two bins):
		## WARNING: the animal will "teleport" between y-coordinates between the RL/LR laps. This will mean that all velocity_y, or vector-based velocity calculations (that use both x and y) are going to be messed up.
		
		First decoder is assigned virtual y-positions: 1.0
		Second decoder is assigned virtual y-positions: 2.0,
		etc.
		
		"""
		...
	
	@classmethod
	def build_merged_directional_placefields(cls, input_unidirectional_decoder_dict, debug_print=...) -> PfND:
		""" 2024-01-02 - Combine the non-directional PDFs and renormalize to get the directional PDF:

		Builds a manually merged directional pf from a dict of pf1Ds (one for each direction)

		First decoder is assigned virtual y-positions: 1.0
		Second decoder is assigned virtual y-positions: 2.0,
		etc.


		@#TODO 2024-04-05 22:20: - [ ] The returned combined PfND is missing its `spikes_df`, `filtered_spikes_df` properties making `.get_by_id(...)` not work at all. Also `.extended_neuron_ids` is a list instead of a NDArray, making indexing into that fail in certain places.
		
		Usage:
			from neuropy.analyses.placefields import PfND

			## Combine the non-directional PDFs and renormalize to get the directional PDF:
			# Inputs: long_LR_pf1D, long_RL_pf1D
			merged_pf1D = PfND.build_merged_directional_placefields(deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), debug_print = True)
			merged_pf1D 

		"""
		...
	
	@classmethod
	def determine_pf_aclus_filtered_by_frate_and_qclu(cls, pf_dict: Dict[str, PfND], minimum_inclusion_fr_Hz: Optional[float] = ..., included_qclu_values: Optional[List] = ...): # -> tuple[dict[str, PfND], list[NDArray[Any]]]:
		""" Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
		minimum_inclusion_fr_Hz: float = 5.0
		modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

		individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

		Usage:
			filtered_decoder_dict, filtered_direction_shared_aclus_list = PfND.determine_pf_aclus_filtered_by_frate_and_qclu(pf_dict=track_templates.get_pf_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

		"""
		...
	


def perform_compute_placefields(active_session_spikes_df, active_pos, computation_config: PlacefieldComputationParameters, active_epoch_placefields1D=..., active_epoch_placefields2D=..., included_epochs=..., should_force_recompute_placefields=..., progress_logger=...): # -> tuple[PfND | Any, PfND | Any]:
	""" Most general computation function. Computes both 1D and 2D placefields.
	active_epoch_session_Neurons:
	active_epoch_pos: a Position object
	included_epochs: a Epoch object to filter with, only included epochs are included in the PF calculations
	active_epoch_placefields1D (Pf1D, optional) & active_epoch_placefields2D (Pf2D, optional): allow you to pass already computed Pf1D and Pf2D objects from previous runs and it won't recompute them so long as should_force_recompute_placefields=False, which is useful in interactive Notebooks/scripts
	Usage:
		active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_epoch_session_Neurons, active_epoch_pos, active_epoch_placefields1D, active_epoch_placefields2D, active_config.computation_config, should_force_recompute_placefields=True)


	NOTE: 2023-04-07 - Uses only the spikes from PYRAMIDAL cells in `active_session_spikes_df` to perform the placefield computations. 
	"""
	...

def compute_placefields_masked_by_epochs(sess, active_config, included_epochs=..., should_display_2D_plots=...): # -> tuple[PfND | Any, PfND | Any]:
	""" Wrapps perform_compute_placefields to make the call simpler """
	...

def compute_placefields_as_needed(active_session, computation_config: PlacefieldComputationParameters = ..., general_config=..., active_placefields1D=..., active_placefields2D=..., included_epochs=..., should_force_recompute_placefields=..., should_display_2D_plots=...): # -> tuple[PfND | Any, PfND | Any]:
	...

