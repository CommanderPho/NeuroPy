"""
This type stub file was generated by pyright.
"""

import numpy as np
import pandas as pd
from typing import Sequence, Union
from .epoch import Epoch
from .datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicedMixin
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable
from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable
from neuropy.utils.mixins.HDF5_representable import HDFMixin

def build_position_df_time_window_idx(active_pos_df, curr_active_time_windows, debug_print=...):
    """ adds the time_window_idx column to the active_pos_df
    Usage:
        curr_active_time_windows = np.array(pho_custom_decoder.active_time_windows)
        active_pos_df = build_position_df_time_window_idx(sess.position.to_dataframe(), curr_active_time_windows)
    """
    ...

def build_position_df_resampled_to_time_windows(active_pos_df, time_bin_size=...):
    """ Note that this returns a TimedeltaIndexResampler, not a dataframe proper. To get the real dataframe call .nearest() on output.

    Usage:
        time_binned_position_resampler = build_position_df_resampled_to_time_windows(computation_result.sess.position.to_dataframe(), time_bin_size=computation_result.computation_config.pf_params.time_bin_size) # TimedeltaIndexResampler
        time_binned_position_df = time_binned_position_resampler.nearest() # an actual dataframe
    """
    ...

class PositionDimDataMixin:
    """ Implementors gain convenience properties to access .x, .y, and .z variables as properties. 
    Requirements:
        Implement  
            @property
            def df(self):
                return <a dataframe>
    """
    __time_variable_name = ...
    @property
    def df(self):
        """The df property."""
        ...
    
    @df.setter
    def df(self, value):
        ...
    
    @property
    def x(self):
        ...
    
    @x.setter
    def x(self, x): # -> None:
        ...
    
    @property
    def y(self):
        ...
    
    @y.setter
    def y(self, y): # -> None:
        ...
    
    @property
    def z(self):
        ...
    
    @z.setter
    def z(self, z): # -> None:
        ...
    
    @property
    def traces(self):
        """ Compatibility method for the old-style implementation. """
        ...
    
    @property
    def time_variable_name(self): # -> str:
        ...
    
    @property
    def time(self):
        ...
    
    @property
    def t_start(self):
        ...
    
    @t_start.setter
    def t_start(self, t):
        ...
    
    @property
    def t_stop(self):
        ...
    
    @property
    def duration(self):
        ...
    
    @property
    def ndim(self): # -> bool_:
        """ returns the count of the spatial columns that the dataframe has """
        ...
    
    @property
    def dim_columns(self): # -> list[Any]:
        """ returns the labels of the columns that correspond to spatial columns 
            If ndim == 1, returns ['x'], 
            if ndim == 2, returns ['x','y'], etc.
        """
        ...
    
    @property
    def n_frames(self): # -> int:
        ...
    


class PositionComputedDataMixin:
    """ Requires conformance to PositionDimDataMixin as well. Adds computed properties like velocity and acceleration (higher-order derivatives of position) and smoothed values to the dataframe."""
    @property
    def dim_computed_columns(self, include_dt=...): # -> list[str]:
        """ returns the labels for the computed columns
            output: ['dt', 'velocity_x', 'acceleration_x', 'velocity_y', 'acceleration_y']
        """
        ...
    
    def compute_higher_order_derivatives(self): # -> DataFrame[Any]:
        """Computes the higher-order positional derivatives for all spatial dimensional components of self.df. Adds the dt, velocity, and acceleration columns
        """
        ...
    
    @property
    def dim_smoothed_columns(self, non_smoothed_column_labels=...): # -> list[str]:
        """ returns the labels for the smoothed columns
            non_smoothed_column_labels: list can be specified to only get the smoothed labels for the variables in the non_smoothed_column_labels list
            output: ['x_smooth', 'y_smooth', 'velocity_x_smooth', 'acceleration_x_smooth', 'velocity_y_smooth', 'acceleration_y_smooth']
        """
        ...
    
    def compute_smoothed_position_info(self, N: int = ..., non_smoothed_column_labels=...): # -> DataFrame[Any]:
        """Computes smoothed position variables and adds them as columns to the internal dataframe
        Args:
            N (int, optional): Number of previous samples to smooth over. Defaults to 20.
        Returns:
            [type]: [description]
        """
        ...
    
    @property
    def linear_pos_obj(self) -> Position:
        """ returns a Position object containing only the linear_pos as its trace. This is used for compatibility with Bapun's Pf1D function """
        ...
    
    @property
    def linear_pos(self): # -> ndarray[Any, Any]:
        ...
    
    @linear_pos.setter
    def linear_pos(self, linear_pos): # -> None:
        ...
    
    @property
    def has_linear_pos(self): # -> bool:
        ...
    
    def compute_linearized_position(self, method=..., **kwargs) -> Position:
        """ computes and adds the linear position to this Position object """
        ...
    
    @property
    def speed(self): # -> ndarray[Any, Any]:
        ...
    
    @property
    def dt(self): # -> ndarray[Any, Any]:
        ...
    
    @property
    def velocity_x(self): # -> ndarray[Any, Any]:
        ...
    
    @property
    def acceleration_x(self): # -> ndarray[Any, Any]:
        ...
    
    @property
    def velocity_y(self): # -> ndarray[Any, Any]:
        ...
    
    @property
    def acceleration_y(self): # -> ndarray[Any, Any]:
        ...
    


@pd.api.extensions.register_dataframe_accessor("position")
class PositionAccessor(PositionDimDataMixin, PositionComputedDataMixin, TimeSlicedMixin):
    """ A Pandas DataFrame-based Position helper. """
    def __init__(self, pandas_obj) -> None:
        ...
    
    @property
    def df(self): # -> Any:
        ...
    
    @df.setter
    def df(self, value): # -> None:
        ...
    
    def to_Position_obj(self, metadata=...): # -> Position:
        """ builds a Position object from the PositionAccessor's dataframe 
        Usage:
            pos_df.position.to_Position_obj()
        """
        ...
    
    def drop_dimensions_above(self, desired_ndim: int, inplace: bool = ...): # -> None:
        """ drops all columns related to dimensions above `desired_ndim`.

        e.g. desired_ndim = 1:
            would drop 'y' related columns

        if inplace is True, None is returned and the dataframe is modified in place

        """
        ...
    


class Position(HDFMixin, PositionDimDataMixin, PositionComputedDataMixin, ConcatenationInitializable, StartStopTimesMixin, TimeSlicableObjectProtocol, DataFrameRepresentable, DataWriter):
    def __init__(self, pos_df: pd.DataFrame, metadata=...) -> None:
        """[summary]
        Args:
            pos_df (pd.DataFrame): Each column is a pd.Series(["t", "x", "y"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        ...
    
    def time_slice_indicies(self, t_start, t_stop): # -> Index:
        ...
    
    @classmethod
    def init(cls, traces: np.ndarray, computed_traces: np.ndarray = ..., t_start=..., sampling_rate=..., metadata=...): # -> Position:
        """ Comatibility initializer """
        ...
    
    @classmethod
    def legacy_from_dict(cls, dict_rep: dict): # -> Position:
        """ Tries to load the dict using previous versions of this code. """
        ...
    
    @property
    def df(self): # -> DataFrame[Any]:
        ...
    
    @df.setter
    def df(self, value): # -> None:
        ...
    
    @property
    def sampling_rate(self): # -> floating[Any]:
        ...
    
    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        ...
    
    def to_dict(self): # -> dict[str, Any]:
        ...
    
    @staticmethod
    def from_dict(d): # -> Position:
        ...
    
    def to_dataframe(self): # -> DataFrame[Any]:
        ...
    
    def speed_in_epochs(self, epochs: Epoch): # -> None:
        ...
    
    def time_slice(self, t_start, t_stop): # -> Position:
        ...
    
    @classmethod
    def from_separate_arrays(cls, t, x, y=..., z=..., lin_pos=..., metadata=...): # -> Self:
        ...
    
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]): # -> Self:
        """ Concatenates the object list """
        ...
    
    def drop_dimensions_above(self, desired_ndim: int):
        """ modifies the internal dataframe to drop dimensions above a certain number. Always done in place, and returns None. """
        ...
    
    def print_debug_str(self): # -> None:
        ...
    
    def to_hdf(self, file_path, key: str, **kwargs): # -> None:
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pos_obj: Position = long_one_step_decoder_1D.pf.position
            _pos_obj.to_hdf(hdf5_output_path, key='pos')
        """
        ...
    
    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> Position:
        """ Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = Position.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        ...
    


