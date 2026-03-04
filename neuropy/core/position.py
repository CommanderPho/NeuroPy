from copy import deepcopy
from typing import Sequence, Union, Optional, Tuple, Dict, List, Any
import itertools # for flattening lists with itertools.chain.from_iterable()
from nptyping import NDArray
import numpy as np
import h5py # for to_hdf and read_hdf definitions
from pandas.core.indexing import IndexingError

import pandas as pd
from .epoch import Epoch
from .datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin, TimeSlicedMixin
from neuropy.utils.mixins.concatenatable import ConcatenationInitializable
from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable, ensure_dataframe
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.time_slicing import TimePointEventAccessor
from neuropy.utils.mixins.position_slicing import PositionSlicedMixin

    
""" --- Helper FUNCTIONS """
def build_position_df_time_window_idx(active_pos_df: pd.DataFrame, curr_active_time_windows, debug_print=False):
    """ adds the time_window_idx column to the active_pos_df
    Usage:
        curr_active_time_windows = np.array(pho_custom_decoder.active_time_windows)
        active_pos_df = build_position_df_time_window_idx(sess.position.to_dataframe(), curr_active_time_windows)
    """
    active_pos_df['time_window_idx'] = np.full_like(active_pos_df['t'], -1, dtype='int')
    starts = curr_active_time_windows[:,0]
    stops = curr_active_time_windows[:,1]
    num_slices = len(starts)
    if debug_print:
        print(f'starts: {np.shape(starts)}, stops: {np.shape(stops)}, num_slices: {num_slices}')
    for i in np.arange(num_slices):
        active_pos_df.loc[active_pos_df[active_pos_df.position.time_variable_name].between(starts[i], stops[i], inclusive='both'), ['time_window_idx']] = int(i) # set the 'time_window_idx' identifier on the object
    active_pos_df['time_window_idx'] = active_pos_df['time_window_idx'].astype(int) # ensure output is the correct datatype
    return active_pos_df


def build_position_df_resampled_to_time_windows(active_pos_df: pd.DataFrame, time_bin_size=0.02):
    """ Note that this returns a TimedeltaIndexResampler, not a dataframe proper. To get the real dataframe call .nearest() on output.

    Usage:
        time_binned_position_resampler = build_position_df_resampled_to_time_windows(computation_result.sess.position.to_dataframe(), time_bin_size=computation_result.computation_config.pf_params.time_bin_size) # TimedeltaIndexResampler
        time_binned_position_df = time_binned_position_resampler.nearest() # an actual dataframe
    """
    position_time_delta = pd.to_timedelta(active_pos_df[active_pos_df.position.time_variable_name], unit="sec")
    active_pos_df['time_delta_sec'] = position_time_delta
    active_pos_df = active_pos_df.set_index('time_delta_sec')
    window_resampled_pos_df = active_pos_df.resample(f'{time_bin_size}S', base=0) #.nearest() # '0.02S' 0.02 second bins
    # window_resampled_pos_df = active_pos_df.resample(f'{time_bin_size}S') # , origin='start'
    # What happened to the `base` parameter in Pandas v2? I used to `window_resampled_pos_df = active_pos_df.resample(f'{time_bin_size}S', base=0)` and now I get `TypeError: resample() got an unexpected keyword argument 'base'`

    return window_resampled_pos_df



""" --- Helper MIXINS """
class PositionDimDataMixin:
    """ Implementors gain convenience properties to access .x, .y, and .z variables as properties. 
    Requirements:
        Implement  
            @property
            def df(self):
                return <a dataframe>
    """
    __time_variable_name = 't' # currently hardcoded
    
    @property
    def df(self):
        """The df property."""
        raise NotImplementedError # must be overriden by implementor 
        # return self._obj # for PositionAccessor
        # return self._df # for Position
    @df.setter
    def df(self, value):
        raise NotImplementedError # must be overriden by implementor 
        # self._obj = value # for PositionAccessor
        # self._df = value # for Position
    
    
    # Position
    @property
    def x(self):
        return self.df['x'].to_numpy()

    @x.setter
    def x(self, x):
        self.df.loc[:, 'x'] = x

    @property
    def y(self):
        assert self.ndim > 1, "No y for one-dimensional position"
        return self.df['y'].to_numpy()

    @y.setter
    def y(self, y):
        assert self.ndim > 1, "Position data has only one dimension"
        self.df.loc[:, 'y'] = y

    @property
    def z(self):
        assert self.ndim == 3, "Position data is not three-dimensional"
        return self.df['z'].to_numpy()

    @z.setter
    def z(self, z):
        self.df.loc[:, 'z'] = z
    
    @property
    def traces(self):
        """ Compatibility method for the old-style implementation. """
        # print('traces accessed with self.ndim of {}'.format(self.ndim))
        if self.ndim == 1:
            return self.df[['x']].to_numpy().T
        elif self.ndim >= 2:
            return self.df[['x','y']].to_numpy().T
        elif self.ndim >= 3:
            return self.df[['x','y','z']].to_numpy().T
        else:
            raise IndexingError
        
    # Time Properties:
    @property
    def time_variable_name(self):
        return PositionDimDataMixin.__time_variable_name
    @property
    def time(self):
        return self.df[self.time_variable_name].to_numpy()
    @property
    def t_start(self):
        return self.df[self.time_variable_name].iloc[0]
    @t_start.setter
    def t_start(self, t):
        raise NotImplementedError
        # self._t_start = t
    @property
    def t_stop(self):
        return self.df[self.time_variable_name].iloc[-1]
    @property
    def duration(self):
        return self.t_stop - self.t_start
        # return float(self.n_frames) / float(self.sampling_rate)

    # Dimension Properties:
    @property
    def ndim(self):
        """ returns the count of the spatial columns that the dataframe has """
        return np.sum(np.isin(['x','y','z'], self.df.columns))
    @property
    def dim_columns(self):
        """ returns the labels of the columns that correspond to spatial columns 
            If ndim == 1, returns ['x'], 
            if ndim == 2, returns ['x','y'], etc.
        """
        spatial_column_labels = np.array(['x','y','z'])
        return list(spatial_column_labels[np.isin(spatial_column_labels, self.df.columns)])
    
    @property
    def n_frames(self):
        return len(self.df.index)




class PositionComputedDataMixin(PositionSlicedMixin):
    """ Requires conformance to PositionDimDataMixin as well. Adds computed properties like velocity and acceleration (higher-order derivatives of position) and smoothed values to the dataframe."""
    
    ## Computed Variable Labels Properties:
    @staticmethod
    def _computed_column_component_labels(component_label):
        return [f'velocity_{component_label}', f'acceleration_{component_label}']
    @property
    def dim_computed_columns(self, include_dt=False, include_optional_columns:bool=False):
        """ returns the labels for the computed columns
            output: ['dt', 'velocity_x', 'acceleration_x', 'velocity_y', 'acceleration_y']
        """
        computed_column_labels = [PositionComputedDataMixin._computed_column_component_labels(a_dim_label) for a_dim_label in self.dim_columns]
        if include_dt:
            computed_column_labels.insert(0, ['dt']) # insert 'dt' at the start of the list

        if include_optional_columns:
            computed_column_labels.extend([k for k in self.dim_optional_additional_columns if k in self.df.columns]) ## if we have these optional columns
            
        # itertools.chain.from_iterable converts ['dt', ['velocity_x', 'acceleration_x'], ['velocity_y', 'acceleration_y']] to ['dt', 'velocity_x', 'acceleration_x', 'velocity_y', 'acceleration_y']
        computed_column_labels = list(itertools.chain.from_iterable(computed_column_labels)) # ['dt', 'velocity_x', 'acceleration_x', 'velocity_y', 'acceleration_y']
        return computed_column_labels
    
    @staticmethod
    def _perform_compute_higher_order_derivatives(pos_df: pd.DataFrame, component_label: str, time_variable_name: str = 't') -> pd.DataFrame:
        """Computes the higher-order positional derivatives for a single component (given by component_label) of the pos_df
        Args:
            pos_df (pd.DataFrame): [description]
            component_label (str): [description]
        Returns:
            pd.DataFrame: The updated dataframe with the dt, velocity, and acceleration columns added.
        """
        # compute each component separately:
        velocity_column_key = f'velocity_{component_label}'
        acceleration_column_key = f'acceleration_{component_label}'
                
        dt = np.insert(np.diff(pos_df[time_variable_name]), 0, np.nan)
        velocity_comp = np.insert(np.diff(pos_df[component_label]), 0, 0.0) / dt
        velocity_comp[np.isnan(velocity_comp)] = 0.0 # replace NaN components with zero
        acceleration_comp = np.insert(np.diff(velocity_comp), 0, 0.0) / dt
        acceleration_comp[np.isnan(acceleration_comp)] = 0.0 # replace NaN components with zero
        dt[np.isnan(dt)] = 0.0 # replace NaN components with zero
        
        # add the columns to the dataframe:
        # pos_df.loc[:, 'dt'] = dt
        pos_df['dt'] = dt
        pos_df[velocity_column_key] = velocity_comp
        pos_df[acceleration_column_key] = acceleration_comp
        
        return pos_df
    
    def compute_higher_order_derivatives(self) -> pd.DataFrame:
        """Computes the higher-order positional derivatives for all spatial dimensional components of self.df. Adds the dt, velocity, and acceleration columns
        """
        for dim_i in np.arange(self.ndim):
            curr_column_label = self.dim_columns[dim_i]
            self.df = PositionComputedDataMixin._perform_compute_higher_order_derivatives(self.df, curr_column_label, time_variable_name=self.time_variable_name)
        return self.df
    
    
    ## Smoothed Computed Variables:
    @staticmethod
    def _smoothed_column_labels(non_smoothed_column_labels):
        """ returns the smoothed_column_labels given the specified non_smoothed_column_labels (which is a list of str) """
        return [f'{a_label}_smooth' for a_label in non_smoothed_column_labels]
    @property
    def dim_smoothed_columns(self, non_smoothed_column_labels=None):
        """ returns the labels for the smoothed columns
            non_smoothed_column_labels: list can be specified to only get the smoothed labels for the variables in the non_smoothed_column_labels list
            output: ['x_smooth', 'y_smooth', 'velocity_x_smooth', 'acceleration_x_smooth', 'velocity_y_smooth', 'acceleration_y_smooth']
        """
        if non_smoothed_column_labels is None:
            non_smoothed_column_labels = (self.dim_columns + self.dim_computed_columns)
        smoothed_column_labels = PositionComputedDataMixin._smoothed_column_labels(non_smoothed_column_labels)
        # smoothed_column_labels = list(itertools.chain.from_iterable(smoothed_column_labels)) # flattens list, but I don't think we need this
        return smoothed_column_labels
    
    @staticmethod
    def _perform_compute_smoothed_position_info(pos_df: pd.DataFrame, non_smoothed_column_labels,  N: int = 20) -> pd.DataFrame:
        """Computes the smoothed quantities for a single component (given by component_label) of the pos_df
        Args:
            pos_df (pd.DataFrame): [description]
            non_smoothed_column_labels (list(str)): a list of the columns to be smoothed
            N (int): 20 # roll over the last N samples
            
        Returns:
            pd.DataFrame: The updated dataframe with the dt, velocity, and acceleration columns added.
            
        Usage:
            smoothed_pos_df = Position._perform_compute_smoothed_position_info(pos_df, (position_obj.dim_columns + position_obj.dim_computed_columns), N=20)
        
        """
        smoothed_column_names = PositionComputedDataMixin._smoothed_column_labels(non_smoothed_column_labels)
        pos_df[smoothed_column_names] = pos_df[non_smoothed_column_labels].rolling(window=N).mean()
        return pos_df
    
    def compute_smoothed_position_info(self, N: int = 20, non_smoothed_column_labels=None) -> pd.DataFrame:
        """Computes smoothed position variables and adds them as columns to the internal dataframe
        Args:
            N (int, optional): Number of previous samples to smooth over. Defaults to 20.
        Returns:
            [type]: [description]
        """
        if non_smoothed_column_labels is None:
            non_smoothed_column_labels = (self.dim_columns + self.dim_computed_columns)
        # smoothed_column_names = self.dim_smoothed_columns(non_smoothed_column_labels)
        self.df = PositionComputedDataMixin._perform_compute_smoothed_position_info(self.df, non_smoothed_column_labels, N=N)
        return self.df        
    

    def compute_speed_info(self) -> pd.DataFrame:
        """ explicitly recomputes the speed """
        if 'speed' not in self.df:
            dt = np.mean(np.diff(self.time))
            self.df['speed'] = np.insert((np.sqrt(((np.abs(np.diff(self.traces, axis=1))) ** 2).sum(axis=0)) / dt), 0, 0.0) # prepends a 0.0 value to the front of the result array so it's the same length as the other position vectors (x, y, etc)  

        if (self.ndim > 1) and ('speed_xy' not in self.df):
            if ('velocity_x_smooth' not in self.df) or ('velocity_y_smooth' not in self.df):
                self.compute_higher_order_derivatives()
                self.compute_smoothed_position_info()
            self.df['speed_xy'] = np.hypot(self.df['velocity_x_smooth'], self.df['velocity_y_smooth'])
        return self.df
    

    ## Linear Position Properties:
    @property
    def linear_pos_obj(self) -> "Position":
        """ returns a Position object containing only the linear_pos as its trace. This is used for compatibility with Bapun's Pf1D function """ 
        if not self.has_linear_pos:
            # Uses`self.pf_params.linearization_method`
            linearization_method: str = 'isomap'
            try:
                linearization_method = self.pf_params.linearization_method
                print(f'linearization_method: {linearization_method}')
            except (ValueError, AttributeError) as e:
                linearization_method = 'isomap' ## fallback to isomap                
            except Exception as e:
                linearization_method = 'isomap' ## fallback to isomap
                raise e
            
            self.compute_linearized_position(method=linearization_method)
            assert self.has_linear_pos, "Doesn't have linear position even after `self.compute_linearized_position()` was called!"
            
        extra_col_names = ['lap', 'lap_dir']
        active_extra_col_names = [v for v in extra_col_names if v in self.df.columns]
        
        # linear_pos_df[extra_col_names] = deepcopy(pos_df[extra_col_names])

        lin_pos_df = deepcopy(self.df[[self.time_variable_name, 'lin_pos', *active_extra_col_names]])
        # lin_pos_df.rename({'lin_pos':'x'}, axis='columns', errors='raise', inplace=True)
        lin_pos_df['x'] = lin_pos_df['lin_pos'].copy() # duplicate the lin_pos column to the 'x' column
        out_obj = Position(lin_pos_df, metadata=None) ## build position object out of the dataframe
        out_obj.compute_higher_order_derivatives()
        out_obj.compute_smoothed_position_info()
        out_obj.speed; # ensure speed is calculated for the new object
        return out_obj
    @property
    def linear_pos(self):
        assert 'lin_pos' in self.df.columns, "Linear Position data has not yet been computed."
        return self.df['lin_pos'].to_numpy()
    @linear_pos.setter
    def linear_pos(self, linear_pos):
        self.df.loc[:, 'lin_pos'] = linear_pos
    @property
    def has_linear_pos(self):
        if 'lin_pos' in self.df.columns:
            return not np.isnan(self.df['lin_pos'].to_numpy()).all() # check if all are nan
        else:
            # Linear Position data has not yet been computed.
            return False


    def compute_linearized_position(self, method='isomap', **kwargs) -> "Position":
        """ computes and adds the linear position to this Position object """
        from neuropy.utils import position_util
        # out_linear_position_obj = position_util.linearize_position(self, method=method, **kwargs)
        # self._df['lin_pos'] = out_linear_position_obj.to_dataframe()['lin_pos'] # add the `lin_pos` column to the pos_df
        self.df = position_util.linearize_position_df(self.df, method=method, **kwargs) # adds 'lin_pos' column to `self.df`
        return self
    

    ## Optional Computed Variables:
    @property
    def dim_optional_additional_columns(self) -> List[str]:
        """ returns the labels for the optional columns """
        all_segment_transfer_col_names = ['segment_idx', 'Vp']
        return ['normal_dir_unit_t', 'normal_dir_unit_x', 'approx_head_dir_degrees', *all_segment_transfer_col_names]
    

    def adding_approx_head_dir_columns(self, N:int=15, n_dir_angular_bins: int = 8) -> pd.DataFrame:
        """ adds a one or more binned position columns (depending on whether 2D position is available) - given the `xbin_edges` and (optionally `ybin_edges`) or a `active_computation_config` config provided 
        `active_computation_config` is not used/needed if the appropriate xbin_edges/ybin_edges are provided.
        Internally uses: `cls.perform_add_binned_position_columns(...)`
        adds columns: ['approx_head_dir_degrees', 'head_dir_angle_binned']

        Usage:
            global_pos_obj: Position = deepcopy(global_session.position)
            global_pos_df: pd.DataFrame = global_pos_obj.adding_approx_head_dir_columns() ## also `global_pos_obj` 
            
        """
        self.df = self.compute_higher_order_derivatives().position.compute_smoothed_position_info(N=N)
        self.df['approx_head_dir_degrees'] = ((np.rad2deg(np.arctan2(self.df['velocity_y_smooth'], self.df['velocity_x_smooth'])) + 360) % 360) # arctan2 is required to get the angle right
        self.df = self.df.dropna(axis='index', subset=['approx_head_dir_degrees'])
        
        self.df = self.bin_segment_direction_angles(self.df, n_dir_angular_bins=n_dir_angular_bins, angle_degrees_col_name='approx_head_dir_degrees', output_col_name='head_dir_angle_binned')

        return self.df



    # @function_attributes(short_name=None, tags=['position', 'hairy', 'hair', 'normal', 'direction'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-30 12:24', related_items=[])
    def adding_hairy_curve_normal_dir_columns(self, N:int=15, x_column_name: str = 'x_smooth', time_column_name: str = 't', replace_existing:bool=False) -> pd.DataFrame:
        """ adds two columns indicating the direction orthogonal (normal) to the direction of position change at each point.        
        Adds: ['normal_dir_unit_t', 'normal_dir_unit_x']
        
        Usage:
            global_pos_obj: Position = deepcopy(global_session.position)

        """
        if ('normal_dir_unit_t' in self.df.columns) and ('normal_dir_unit_x' in self.df.columns) and (not replace_existing):
            return self.df
        else:       
            self.df = self.compute_higher_order_derivatives().position.compute_smoothed_position_info(N=N)
            # 1. Tangent slope  m = dx/dt  (use central differences with np.gradient)
            self.df['m_tan'] = np.gradient(self.df[x_column_name].values, self.df[time_column_name].values)   # dx/dt

            # 2a. Slope of the normal line
            # self.df['m_normal'] = -1.0 / self.df['m_tan'] # −1/m

            # 2b. Raw 2-D normal vector  n = (−m , 1)
            # self.df['n_vec'] = list(zip(-self.df['m_tan'], np.ones(len(self.df))))

            # 2c. Unit-length normal vector
            n_x = -self.df['m_tan'].values
            n_t = np.ones_like(n_x)
            norm = np.sqrt(n_x**2 + n_t**2) # Normalise it
            self.df['normal_dir_unit_t'] = n_t / norm
            self.df['normal_dir_unit_x'] = n_x / norm
            
            # Optional: drop helper columns if you like
            self.df = self.df.drop(columns=['m_tan'])
            # self.df = self.df.dropna(axis='index', subset=['normal_dir_unit_t', 'normal_dir_unit_x']) ## should we drop undefined columns?
            return self.df


            
    
    ## Computed Variable Properties:
    @property
    def speed(self):
        if 'speed' in self.df.columns:
            return self.df['speed'].to_numpy()
        else:
            # Compute the speed if not already done upon first access
            dt = np.mean(np.diff(self.time))
            self.df['speed'] = np.insert((np.sqrt(((np.abs(np.diff(self.traces, axis=1))) ** 2).sum(axis=0)) / dt), 0, 0.0) # prepends a 0.0 value to the front of the result array so it's the same length as the other position vectors (x, y, etc)        
        return self.df['speed'].to_numpy()
    
    @property
    def speed_xy(self):
        if 'speed_xy' in self.df.columns:
            return self.df['speed_xy'].to_numpy()
        else:
            # Compute the speed if not already done upon first access
            self.compute_speed_info()

        return self.df['speed_xy'].to_numpy()
    



    @property
    def dt(self):
        if 'dt' in self.df.columns:
            return self.df['dt'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['dt'].to_numpy()
    
    @property
    def velocity_x(self):
        if 'velocity_x' in self.df.columns:
            return self.df['velocity_x'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['velocity_x'].to_numpy()
    
    @property
    def acceleration_x(self):
        if 'acceleration_x' in self.df.columns:
            return self.df['acceleration_x'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['acceleration_x'].to_numpy()
    
    @property
    def velocity_y(self):
        if 'velocity_y' in self.df.columns:
            return self.df['velocity_y'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['velocity_y'].to_numpy()
    
    @property
    def acceleration_y(self):
        if 'acceleration_y' in self.df.columns:
            return self.df['acceleration_y'].to_numpy()
        else:
            # compute the higher_order_derivatives if not already done upon first access
            self.df = self.compute_higher_order_derivatives()   
        return self.df['acceleration_y'].to_numpy()


    # @function_attributes(short_name=None, tags=['BCPA', 'change-point-detection', 'segment'], input_requires=[], output_provides=[], uses=[], used_by=['cls.segment_trajectories], creation_date='2026-01-14 09:27', related_items=[])
    @classmethod
    def calculate_persistence_velocity(cls, pos_df: pd.DataFrame, t_col_name: str = 't', overwrite_existing:bool=True) -> pd.DataFrame:
        """ Calculate Persistence Velocity (Behavioral Metric)
        Assumes dataframe has 'x', 'y', and 'time' columns.
        Uses smoothed columns and existing computed columns (velocity_x_smooth, velocity_y_smooth, speed_xy, dt) if available.
        Returns dataframe with Persistence Velocity (Vp).
        """
        # Track original columns to identify any temporary columns we might create
        original_columns = set(pos_df.columns)
        
        # Small epsilon to prevent division by zero (using machine epsilon for float64)
        eps = np.finfo(np.float64).eps * 10  # Small but not too small to avoid numerical issues
        
        if overwrite_existing or ('Vp' not in pos_df):
            # Use existing computed velocity columns if available, otherwise calculate from position
            if 'velocity_x_smooth' in pos_df.columns and 'velocity_y_smooth' in pos_df.columns:
                # Use existing smoothed velocity components
                velocity_x = pos_df['velocity_x_smooth']
                velocity_y = pos_df['velocity_y_smooth']
                
                # Calculate speed from existing components or use speed_xy if available
                if 'speed_xy' in pos_df.columns:
                    velocity = pos_df['speed_xy']
                else:
                    velocity = np.hypot(velocity_x, velocity_y)
                
                # Heading (theta) from velocity components
                theta = np.arctan2(velocity_y, velocity_x)
            else:
                # Fallback: calculate from position columns (use smoothed if available)
                x_col = 'x_smooth' if 'x_smooth' in pos_df.columns else 'x'
                y_col = 'y_smooth' if 'y_smooth' in pos_df.columns else 'y'
                
                # Calculate step lengths
                dx = pos_df[x_col].diff()
                dy = pos_df[y_col].diff()
                
                # Use existing dt if available, otherwise calculate
                if 'dt' in pos_df.columns:
                    dt = pos_df['dt'].copy()
                else:
                    dt = pos_df[t_col_name].diff()
                
                # Replace zeros and negative values in dt with epsilon to prevent division by zero
                # Also handle NaN values (first row from diff()) by keeping them as NaN
                dt_safe = dt.copy()
                # Replace zeros and negative/very small values with epsilon, but keep NaN values as NaN
                dt_safe = dt_safe.where((dt_safe > eps) | dt_safe.isna(), eps)
                
                # Calculate distance moved
                distance = np.sqrt(dx**2 + dy**2)
                
                # Velocity with safe division (NaN in dt will produce NaN in velocity, which is correct)
                velocity = distance / dt_safe
                
                # Heading (theta) - handle case where both dx and dy are zero (or very small)
                # Use epsilon to prevent atan2(0, 0) which is undefined but atan2 handles it gracefully
                theta = np.arctan2(dy, dx)
            
            # Turning angle (difference in heading)
            # Ensure theta is a Series for .diff() method (np.arctan2 on Series returns Series, but type checker may not infer this)
            turning_angle = pd.Series(theta, index=pos_df.index).diff()
            
            # Persistence Velocity = Velocity * cos(Turning Angle)
            # This measures how much of the movement is "directed" vs "tortuous"
            # Fill NaN in turning_angle (first row) with 0, which is correct (no previous angle to compare)
            pos_df['Vp'] = velocity * np.cos(turning_angle.fillna(0))
        
        # # Drop any temporary columns that were created (columns that weren't in original set, except 'Vp')
        # new_columns = set(pos_df.columns) - original_columns
        # temp_columns_to_drop = [col for col in new_columns if col != 'Vp']
        # if temp_columns_to_drop:
        #     pos_df = pos_df.drop(columns=temp_columns_to_drop)
        
        return pos_df #pos_df.dropna(subset=['Vp'], inplace=False)

    @classmethod
    def angular_diff_deg(cls, a, b):
        """ correct (circular) mean giving the minimum angle subtending two other angles in degrees """
        return np.abs((a - b + 180.0) % 360.0 - 180.0)


    @classmethod
    def circular_mean_deg(cls, angle_rad: NDArray) -> NDArray:
        """Returns the mean of angles (in radians), in degrees, handling wrapping.

        segment_angles = pos_df.groupby('segment_idx')['Vp_rad'].agg(lambda x: np.angle(np.mean(np.exp(1j * x.dropna())))) # drop NaNs per segment
        
        segment_angles_deg = pos_df.groupby('segment_idx')['Vp_rad'].agg(cls.circular_mean_deg)
        segment_R = pos_df.groupby('segment_idx')['Vp_rad'].agg(lambda x: np.abs(np.mean(np.exp(1j * x.dropna())))) # R = 1 → perfectly aligned, R → 0 → very scattered

        """
        return np.mod(np.rad2deg(np.angle(np.mean(np.exp(1j * angle_rad.dropna())))), 360)


    @classmethod
    def circular_mean_scatteredness_R(cls, angle_rad: NDArray) -> NDArray:
        """Returns the scatteredness of the angles R: R = 1 → perfectly aligned, R → 0 → very scattered
        segment_R = pos_df.groupby('segment_idx')['Vp_rad'].agg(cls.circular_mean_scatteredness_R)
        """
        return np.abs(np.mean(np.exp(1j * angle_rad.dropna()))) # R = 1 → perfectly aligned, R → 0 → very scattered




    @classmethod
    def bin_segment_direction_angles(cls, pos_df: pd.DataFrame, n_dir_angular_bins: Optional[int] = 8, angle_degrees_col_name: str = 'segment_Vp_deg', output_col_name: str = 'segment_dir_angle_binned') -> pd.DataFrame:
        """Bins segment direction angles into angular bins.
        
        Args:
            pos_df: Position dataframe with angle column specified by angle_degrees_col_name
            n_dir_angular_bins: Number of angular bins to create. If None, returns pos_df unchanged.
            angle_degrees_col_name: Name of the column containing angle values in degrees. Defaults to 'segment_Vp_deg'.
            output_col_name: Name of the output column to create. Defaults to 'segment_dir_angle_binned'.
            
        Returns:
            pd.DataFrame: The dataframe with output column added and NaN rows dropped.
        """
        if n_dir_angular_bins is not None:
            angle_dir_bin_edges = np.linspace(0, 360, (n_dir_angular_bins + 1))
            n_dir_angular_bins: int = len(angle_dir_bin_edges) - 1
            # Use pd.cut with the explicit bin edges
            pos_df[output_col_name] = pd.cut(pos_df[angle_degrees_col_name], bins=angle_dir_bin_edges, labels=False, include_lowest=True)
            # Only drop NaN rows if there are valid rows remaining (prevent empty dataframe)
            if pos_df[output_col_name].notna().any():
                pos_df = pos_df.dropna(axis='index', subset=[output_col_name])
            # Otherwise, keep the dataframe even if all values are NaN (preserve structure)
        return pos_df


    @classmethod
    def compute_segment_representative_angles(cls, pos_df: pd.DataFrame, n_dir_angular_bins: int = 8) -> pd.DataFrame:
        """Compute the representative angle ("Vp" mean direction) for each segment and assign to dataframe.
        
        Args:
            pos_df: Position dataframe with 'Vp' column. If 'segment_idx' column is missing, all rows will be treated as segment 0.
            n_dir_angular_bins: Number of angular bins for direction binning. Defaults to 8.
            
        Returns:
            pd.DataFrame: The dataframe with 'segment_Vp_deg', 'segment_Vp_scatteredness', and binned direction columns added.
        """
        assert 'Vp' in pos_df.columns, f"pos_df.columns: {list(pos_df.columns)}"

        # If segment_idx doesn't exist, create it with all zeros (treat all rows as one segment)
        if 'segment_idx' not in pos_df.columns:
            pos_df['segment_idx'] = 0

        # For each segment_idx, compute the circular mean of angle Vp (in radians)
        if len(pos_df) > 0:
            # Convert Vp from degrees to radians (if not already radians)
            # If Vp values can be > 2pi, assume they are in degrees and convert to radians
            if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
                vp_rad = np.deg2rad(pos_df['Vp'])
            else:
                vp_rad = pos_df['Vp'].astype(np.float64)
            pos_df['Vp_rad'] = vp_rad

            # Compute the mean direction for each segment
            segment_angles_deg = pos_df.groupby('segment_idx')['Vp_rad'].agg(cls.circular_mean_deg)
            segment_R = pos_df.groupby('segment_idx')['Vp_rad'].agg(cls.circular_mean_scatteredness_R) # R = 1 → perfectly aligned, R → 0 → very scattered

            # Map back onto df
            pos_df['segment_Vp_deg'] = pos_df['segment_idx'].map(segment_angles_deg)
            pos_df['segment_Vp_scatteredness'] = pos_df['segment_idx'].map(segment_R)
            pos_df = cls.bin_segment_direction_angles(pos_df, n_dir_angular_bins, angle_degrees_col_name='segment_Vp_deg')

            # Optionally, can drop the helper rad column
            pos_df.drop(columns=['Vp_rad'], inplace=True)

        return pos_df


    # @function_attributes(short_name=None, tags=['BCPA', 'change-point-detection', 'segment'], input_requires=[], output_provides=[], uses=['cls.calculate_persistence_velocity'], used_by=[], creation_date='2026-01-14 09:27', related_items=[])
    @classmethod
    def perform_segment_trajectories(cls, pos_df: pd.DataFrame, should_plot_result: bool=False, min_signal_length: int=4, pen: float=10.0, n_dir_angular_bins: int = 8, overwrite_existing:bool=True, disable_segmentation: bool = True, **kwargs):
        """ BCPA: (Behavioral Change Point Analysis by Gurarie et al.).
        1. Preprocessing: Decomposing movement data into Persistence Velocity (continuous movement) and Turning Velocity.
        2. Analysis: Running a structural change point detection algorithm on those time series.

        Args:
            pos_df: Position dataframe with 'x', 'y', and time columns
            should_plot_result: Whether to plot the change point detection results. Defaults to False.
            min_signal_length: Minimum number of data points required for segmentation. Defaults to 4.
            pen: Penalty parameter for change point detection. Defaults to 10.0. Higher values = fewer change points.
            n_dir_angular_bins: Number of angular bins for direction binning. Defaults to 8.
            overwrite_existing: If False and all required columns exist, return dataframe unchanged. Defaults to True.
            disable_segmentation: If True, skip change point detection and treat all rows as a single segment. Defaults to True.
            **kwargs: Additional arguments passed to calculate_persistence_velocity

        Returns:
            pd.DataFrame: The dataframe with ['segment_idx', 'Vp', 'segment_Vp_deg', 'segment_dir_angle_binned', 'segment_Vp_scatteredness'] column added.

        """
        if (not overwrite_existing) and np.all(np.isin(['segment_idx', 'Vp', 'segment_Vp_deg', 'segment_dir_angle_binned', 'segment_Vp_scatteredness'], pos_df.columns)):
            return pos_df ## return with no modifications
        if len(pos_df) == 0:
            return pos_df

        initial_len: int = len(pos_df)
        # Calculate persistence velocity (Vp) - the direction of movement
        pos_df = cls.calculate_persistence_velocity(pos_df=pos_df, overwrite_existing=overwrite_existing, **kwargs)
        assert len(pos_df) == initial_len, f"initial_len: {initial_len}, len(pos_df): {len(pos_df)}"
        assert 'Vp' in pos_df.columns
        assert len(pos_df) > 0
        
        # Signal to analyze (Persistence Velocity)
        signal = pos_df['Vp'].values
        
        needs_segmentation: bool = True

        # If segmentation is disabled, skip changepoint detection and treat all as single segment
        if disable_segmentation:
            pos_df['segment_idx'] = np.zeros(len(pos_df), dtype=int)
            needs_segmentation = False

        # Validate signal before segmentation
        if len(signal) < min_signal_length:
            # Signal too short for segmentation - assign all to single segment
            pos_df['segment_idx'] = np.zeros(len(pos_df), dtype=int)
            needs_segmentation = False
        
        # Check if signal has variation (not constant)
        if np.std(signal) == 0 or np.isnan(np.std(signal)):
            # Constant signal - no change points possible
            pos_df['segment_idx'] = np.zeros(len(pos_df), dtype=int)
            needs_segmentation = False
        
        # Check for invalid values
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            # Remove invalid values and re-index
            valid_mask = ~(np.isnan(signal) | np.isinf(signal))
            num_valid = np.sum(valid_mask)
            if num_valid < min_signal_length:
                # Not enough valid points after filtering
                pos_df['segment_idx'] = np.zeros(len(pos_df), dtype=int)
                needs_segmentation = False
            elif num_valid == 0:
                # All values are invalid - keep original dataframe but mark all as single segment
                pos_df['segment_idx'] = np.zeros(len(pos_df), dtype=int)
                needs_segmentation = False
            else:
                # Only filter if we have enough valid points and need segmentation
                if needs_segmentation:
                    # Use only valid portion of signal for segmentation
                    signal = signal[valid_mask]
                    pos_df = pos_df.iloc[valid_mask].reset_index(drop=True)
                # If not segmenting, keep all rows (invalid values will be handled elsewhere)

        if needs_segmentation:
            # PELT is a common algorithm for unknown number of change points
            # 'rbf' cost function is non-parametric and robust for behavioral data
            try:
                import ruptures as rpt
                
                model = rpt.Pelt(model="rbf").fit(signal)
                result = model.predict(pen=pen) # 'pen' is the penalty (sensitivity)
                
                if should_plot_result:
                    import matplotlib.pyplot as plt
                    # --- 3. Visualization ---
                    rpt.display(signal, result)
                    plt.title("Behavioral Change Points (Persistence Velocity)")
                    plt.show()
                
                # Assign segment indices based on change points
                # result contains the indices where segments end (excluding the final index)
                # Convert to segment indices: each row gets assigned to segment 0, 1, 2, etc.
                change_points = np.array(result[:-1]) if len(result) > 1 else np.array([])  # Exclude last index (end of data)
                if len(change_points) > 0:
                    # Use searchsorted to find which segment each index belongs to
                    segment_idx = np.searchsorted(change_points, np.arange(len(pos_df)), side='right')
                else:
                    # No change points detected, all data is one segment
                    segment_idx = np.zeros(len(pos_df), dtype=int)
                
                pos_df['segment_idx'] = segment_idx
                
            except Exception as e:
                # Handle ruptures exceptions (BadSegmentationParameters, etc.)
                # Fallback: assign all data to single segment
                import warnings
                warnings.warn(f"Change point detection failed: {e}. Assigning all data to a single segment.", UserWarning)
                pos_df['segment_idx'] = np.zeros(len(pos_df), dtype=int)
                needs_segmentation = False

        # Compute representative angles for each segment
        pos_df = cls.compute_segment_representative_angles(pos_df, n_dir_angular_bins)

        return pos_df


    # @function_attributes(short_name=None, tags=['segment, 'BCPA'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 09:32', related_items=[])
    def adding_segmented_trajectories_columns(self, plot_result: bool=False, disable_segmentation: bool = True, **kwargs) -> pd.DataFrame:
        """ BCPA: (Behavioral Change Point Analysis by Gurarie et al.).
        Instance method version that operates on self.df.
        Internally uses: `cls.perform_segment_trajectories(...)`
        
        Returns:
            pd.DataFrame: The updated dataframe with 'segment_idx' column added.
        """
        self.df = self.perform_segment_trajectories(pos_df=self.df, should_plot_result=plot_result, disable_segmentation=disable_segmentation, **kwargs)
        return self.df




    @classmethod
    def perform_add_representitive_trajectories_angles_columns(cls, pos_df: pd.DataFrame, n_dir_angular_bins: int = 8, overwrite_existing:bool=True, **kwargs) -> pd.DataFrame:
        """Add representative trajectory angle columns without performing segmentation.
        
        Computes persistence velocity (Vp) and representative angles for each segment. Unlike 
        `perform_segment_trajectories`, this function does NOT perform change point detection 
        or segmentation. If 'segment_idx' column is missing, all rows are treated as a single 
        segment (segment 0).
        
        Internally uses: `cls.calculate_persistence_velocity(...)` and `cls.compute_segment_representative_angles(...)`
        
        Args:
            pos_df: Position dataframe with 'x', 'y', and time columns
            n_dir_angular_bins: Number of angular bins for direction binning. Defaults to 8.
            overwrite_existing: If False and all required columns exist, return dataframe unchanged. Defaults to True.
            **kwargs: Additional arguments passed to `calculate_persistence_velocity`
        
        Returns:
            pd.DataFrame: The updated dataframe with columns added:
                - 'Vp': Persistence velocity (direction of movement)
                - 'segment_Vp_deg': Mean direction angle in degrees for each segment
                - 'segment_Vp_scatteredness': Scatteredness measure (R) for each segment (1 = aligned, 0 = scattered)
                - 'segment_dir_angle_binned': Binned direction angle
                - 'segment_idx': Segment index (created as all zeros if missing)
        """
        if (not overwrite_existing) and np.all(np.isin(['Vp', 'segment_Vp_deg', 'segment_dir_angle_binned', 'segment_Vp_scatteredness'], pos_df.columns)):
            return pos_df ## return with no modifications
        if len(pos_df) == 0:
            return pos_df

        initial_len: int = len(pos_df)
        # Calculate persistence velocity (Vp) - the direction of movement
        pos_df = cls.calculate_persistence_velocity(pos_df=pos_df, overwrite_existing=overwrite_existing, **kwargs)
        assert len(pos_df) == initial_len, f"initial_len: {initial_len}, len(pos_df): {len(pos_df)}"
        assert 'Vp' in pos_df.columns
        assert len(pos_df) > 0
        
        # Compute representative angles for each segment (creates segment_idx if missing)
        pos_df = cls.compute_segment_representative_angles(pos_df, n_dir_angular_bins=n_dir_angular_bins)
        return pos_df

    # @function_attributes(short_name=None, tags=['trajectory_angles', 'angle', 'direction'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-15 15:22', related_items=[])
    def adding_representitive_trajectories_angles_columns(self, n_dir_angular_bins: int = 8, overwrite_existing:bool=True, **kwargs) -> pd.DataFrame:
        """Add representative trajectory angle columns without performing segmentation.
        
        Instance method version that operates on self.df. Computes persistence velocity (Vp) 
        and representative angles for each segment. Unlike `adding_segmented_trajectories_columns`, 
        this function does NOT perform change point detection or segmentation. If 'segment_idx' 
        column is missing, all rows are treated as a single segment (segment 0).
        
        Internally uses: `cls.perform_add_representitive_trajectories_angles_columns(...)`
        
        Args:
            n_dir_angular_bins: Number of angular bins for direction binning. Defaults to 8.
            overwrite_existing: If False and all required columns exist, return dataframe unchanged. Defaults to True.
            **kwargs: Additional arguments passed to `calculate_persistence_velocity`
        
        Returns:
            pd.DataFrame: The updated dataframe with columns added:
                - 'Vp': Persistence velocity (direction of movement)
                - 'segment_Vp_deg': Mean direction angle in degrees for each segment
                - 'segment_Vp_scatteredness': Scatteredness measure (R) for each segment (1 = aligned, 0 = scattered)
                - 'segment_dir_angle_binned': Binned direction angle
                - 'segment_idx': Segment index (created as all zeros if missing)
        """
        self.df = self.perform_add_representitive_trajectories_angles_columns(pos_df=self.df, n_dir_angular_bins=n_dir_angular_bins, overwrite_existing=overwrite_existing, **kwargs)
        return self.df



    # ==================================================================================================================== #
    # Position Discretization/Binning                                                                                      #
    # ==================================================================================================================== #
    @classmethod
    def perform_add_binned_position_columns(cls, pos_df: pd.DataFrame, xbin_edges=None, ybin_edges=None, active_computation_config=None, debug_print:bool=False) -> pd.DataFrame:
        """ adds a one or more binned position columns (depending on whether 2D position is available) - given the `xbin_edges` and (optionally `ybin_edges`) or a `active_computation_config` config provided 
        
        PURE: does not modify `pos_df`, returns a copy
        
        `active_computation_config` is not used/needed if the appropriate xbin_edges/ybin_edges are provided.
        
        Internally uses: `build_df_discretized_binned_position_columns`
        
        """
        from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns # for perform_time_range_computation only
        
        pos_df = deepcopy(pos_df) ## does not modify pos_df
        
        if 'y' not in pos_df.columns:
            # Assume 1D:
            ndim = 1
            pos_col_names = ('x',)
            binned_col_names = ('binned_x',)
            bin_values = (xbin_edges,)
        else:
            # otherwise assume 2D:
            ndim = 2
            pos_col_names = ('x', 'y')
            binned_col_names = ('binned_x', 'binned_y')
            bin_values = (xbin_edges, ybin_edges)        

        # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within.
        pos_df, out_bins, bin_info = build_df_discretized_binned_position_columns(pos_df, bin_values=bin_values, position_column_names=pos_col_names, binned_column_names=binned_col_names, active_computation_config=active_computation_config, force_recompute=False, debug_print=debug_print)

        return pos_df

    
    def adding_binned_position_columns(self, xbin_edges=None, ybin_edges=None, active_computation_config=None, debug_print:bool=False) -> pd.DataFrame:
        """ adds a one or more binned position columns (depending on whether 2D position is available) - given the `xbin_edges` and (optionally `ybin_edges`) or a `active_computation_config` config provided 
        `active_computation_config` is not used/needed if the appropriate xbin_edges/ybin_edges are provided.
        Internally uses: `cls.perform_add_binned_position_columns(...)`
        """
        self.df = self.perform_add_binned_position_columns(pos_df=self.df, xbin_edges=xbin_edges, ybin_edges=ybin_edges, active_computation_config=active_computation_config, debug_print=debug_print)
        return self.df
    

    # ==================================================================================================================== #
    # grid_bin_bounds filtering                                                                                            #
    # ==================================================================================================================== #
    def filtered_by_grid_bin_bounds(self, xmin: Optional[float]=None, xmax: Optional[float]=None, ymin: Optional[float]=None, ymax: Optional[float]=None, xmin_xmax_tuple: Optional[Tuple[float, float]]=None, ymin_ymax_tuple: Optional[Tuple[float, float]]=None, grid_bin_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]=None) -> pd.DataFrame:
        """Instance method version that uses position_sliced"""
        return self.position_sliced(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xmin_xmax_tuple=xmin_xmax_tuple, ymin_ymax_tuple=ymin_ymax_tuple, grid_bin_bounds=grid_bin_bounds)

    def find_percent_pos_samples_within_grid_bin_bounds(self, xmin: Optional[float]=None, xmax: Optional[float]=None, ymin: Optional[float]=None, ymax: Optional[float]=None, xmin_xmax_tuple: Optional[Tuple[float, float]]=None, ymin_ymax_tuple: Optional[Tuple[float, float]]=None, grid_bin_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]=None, debug_print: bool=False) -> Tuple[float, pd.DataFrame]:
        """Instance method version using position_sliced
        
        percentage_within_ranges, filtered_df = self.find_percent_pos_samples_within_grid_bin_bounds(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xmin_xmax_tuple=xmin_xmax_tuple, ymin_ymax_tuple=ymin_ymax_tuple, grid_bin_bounds=grid_bin_bounds, debug_print=debug_print)
        """
        filtered_df = self.filtered_by_grid_bin_bounds(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xmin_xmax_tuple=xmin_xmax_tuple, ymin_ymax_tuple=ymin_ymax_tuple, grid_bin_bounds=grid_bin_bounds)
        percentage_within_ranges = (len(filtered_df) / len(self.df)) * 100
        if debug_print:
            print(f'percentage_within_ranges: {percentage_within_ranges}%')
        return percentage_within_ranges, filtered_df




def adding_lap_info_to_position_df(position_df: pd.DataFrame, laps_df: pd.DataFrame, laps_df_lap_id_col_name:str='lap_id', debug_print:bool=False):
    """ Adds a 'lap' column to the position dataframe:
        Also adds a 'lap_dir' column, containing 0 if it's an outbound trial, 1 if it's an inbound trial, and -1 if it's neither.
    Usage:
    
        from neuropy.core.position import adding_lap_info_to_position_df
        
        curr_position_df = self.position.to_dataframe() # get the position dataframe from the session
        curr_laps_df = self.laps.to_dataframe()
        curr_position_df = adding_lap_info_to_position_df(position_df=curr_position_df, laps_df=curr_laps_df)
        
        # update:
        self.position._df['lap'] = curr_position_df['lap']
        self.position._df['lap_dir'] = curr_position_df['lap_dir']
        
    """
    assert laps_df_lap_id_col_name in laps_df, f"laps_df_lap_id_col_name: '{laps_df_lap_id_col_name}' is not in laps_df: {list(laps_df.columns)}"
    
    possible_lap_columns_to_add = ['maze_id', 'lap_dir', 'is_LR_dir', 'truth_decoder_name']
    possible_lap_columns_to_add_not_found_values = {'maze_id':-1, 'lap_dir':-1, 'is_LR_dir':False, 'truth_decoder_name':''}
    lap_columns_to_add = [v for v in possible_lap_columns_to_add if v in laps_df.columns] ## can only add the columns if they're present in laps_df
    #TODO 2025-07-16 06:25: - [ ] `lap_columns_to_add` NOT YET USED
    
    position_df['lap'] = np.NaN # set all 'lap' column to NaN

    unique_lap_ids = np.unique(laps_df[laps_df_lap_id_col_name])
    
    lap_id_to_dir_dict = None
    has_valid_lap_dir: bool = ('lap_dir' in laps_df.columns)
    if has_valid_lap_dir:
        ## initialize pos_df's 'lap_dir' column to -1
        position_df['lap_dir'] = np.full_like(position_df['lap'], possible_lap_columns_to_add_not_found_values['lap_dir']) # set all 'lap_dir' to -1
        lap_id_to_dir_dict = {a_row.lap_id:a_row.lap_dir for a_row in laps_df[[laps_df_lap_id_col_name, 'lap_dir']].itertuples(index=False)}
        
        if debug_print:
            print(f'lap_id_to_dir_dict: {lap_id_to_dir_dict}')
        assert len(unique_lap_ids) == len(lap_id_to_dir_dict)
    else:
        if debug_print:
            print(f'WARN: laps_df is missing the "lap_dir" columnm, so no "lap_dir" will be added to the pos_df')

    n_laps: int = len(unique_lap_ids)
    for i in np.arange(n_laps):
        curr_lap_id = laps_df.loc[laps_df.index[i], 'lap_id'] # The second epoch in a session doesn't start with indicies of the first lap, so instead we need to get laps_df.index[i] to get the correct index
        curr_lap_t_start, curr_lap_t_stop = laps_df.loc[laps_df.index[i], 'start'], laps_df.loc[laps_df.index[i], 'stop']
        # curr_lap_t_start, curr_lap_t_stop = self.laps.get_lap_times(i)
        if debug_print:
            print('lap[{}]: ({}, {}): '.format(curr_lap_id, curr_lap_t_start, curr_lap_t_stop))
        curr_lap_position_df_is_included = position_df['t'].between(curr_lap_t_start, curr_lap_t_stop, inclusive='both') # returns a boolean array indicating inclusion in teh current lap
        position_df.loc[curr_lap_position_df_is_included, ['lap']] = curr_lap_id # set the 'lap' identifier on the object
        
        if has_valid_lap_dir:
            curr_lap_dir = lap_id_to_dir_dict[curr_lap_id]        
            position_df.loc[curr_lap_position_df_is_included, ['lap_dir']] = curr_lap_dir # set the 'lap' identifier on the object


    # update the lap_dir variable:
    # position_df.loc[np.logical_not(np.isnan(position_df.lap.to_numpy())), 'lap_dir'] = np.mod(position_df.loc[np.logical_not(np.isnan(position_df.lap.to_numpy())), 'lap'], 2.0)
    # position_df['lap_dir'] = position_df['lap'].map(lambda v: lap_id_to_dir_dict.get(v, -1))
    
    # return the extracted traces and the updated curr_position_df
    return position_df


""" --- """
@pd.api.extensions.register_dataframe_accessor("position")
class PositionAccessor(PositionDimDataMixin, PositionComputedDataMixin, TimeSlicedMixin, TimePointEventAccessor):
    """ A Pandas DataFrame-based Position helper. """
    
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column for timestamps ('t') and a column for at least 1D positions ('x')
        if "t" not in obj.columns:
            raise AttributeError("Must have at least one time variable: specifically 't' for PositionAccessor.")
        if "x" not in obj.columns:
            raise AttributeError("Must have at least one position dimension column 'x'.")
        # if "lin_pos" not in obj.columns or "speed" not in obj.columns:
        #     raise AttributeError("Must have 'lin_pos' column and 'x'.")

    # for PositionDimDataMixin & PositionComputedDataMixin
    @property
    def df(self):
        return self._obj # for PositionAccessor
    @df.setter
    def df(self, value):
        self._obj = value # for PositionAccessor
    
    def to_Position_obj(self, metadata=None):
        """ builds a Position object from the PositionAccessor's dataframe 
        Usage:
            pos_df.position.to_Position_obj()
        """
        return Position(self._obj, metadata=metadata)

    def drop_dimensions_above(self, desired_ndim:int, inplace:bool=False):
        """ drops all columns related to dimensions above `desired_ndim`.

        e.g. desired_ndim = 1:
            would drop 'y' related columns

        if inplace is True, None is returned and the dataframe is modified in place

        """
        z_related_column_names = [str(c) for c in self._obj.columns if str(c).endswith('z')] # Find z (3D) related columns
        y_related_column_names = [str(c) for c in self._obj.columns if str(c).endswith('y')] # Find y (2D) related columns
        if inplace:
            out_df = None
        else:
            out_df = self._obj.copy()

        if desired_ndim < 3:
            if inplace:
                self._obj.drop(columns=z_related_column_names, inplace=inplace)
            else:
                out_df = out_df.drop(columns=z_related_column_names, inplace=inplace)
        if desired_ndim < 2:
            if inplace:
                self._obj.drop(columns=y_related_column_names, inplace=inplace)
            else:
                out_df = out_df.drop(columns=y_related_column_names, inplace=inplace)

        return out_df
    


    def adding_lap_info(self, laps_df: pd.DataFrame, inplace:bool=False, debug_print:bool=False):
        """ Adds the ['lap', 'lap_dir'] columns to the position dataframe:
            - 'lap_dir' column, containing 0 if it's an outbound trial, 1 if it's an inbound trial, and -1 if it's neither.
        Usage:
        
            from neuropy.core.position import adding_lap_info_to_position_df
            
            curr_position_df = self.position.to_dataframe() # get the position dataframe from the session
            curr_laps_df = self.laps.to_dataframe()
            curr_position_df = curr_position_df.position.adding_lap_info(laps_df=curr_laps_df, inplace=False)
            
            # update:
            self.position._df['lap'] = curr_position_df['lap']
            self.position._df['lap_dir'] = curr_position_df['lap_dir']
            
        """
        if inplace:
            self._obj = adding_lap_info_to_position_df(position_df=self._obj, laps_df=laps_df, debug_print=debug_print)
            return self._obj
        else:
            out_pos_df = self._obj.copy()
            out_pos_df = adding_lap_info_to_position_df(position_df=out_pos_df, laps_df=laps_df, debug_print=debug_print)
            return out_pos_df


    def detect_general_run_epochs(self, minimum_run_speed: float = 10.0, minimum_epoch_duration: float = 0.5, merging_adjacent_max_separation_sec: float = 2.0, speed_col_name: str ='speed') -> pd.DataFrame:
        """ 
        Returns an Epochs objects describe time frames where the animal is above a certain speed
        
        """
        from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor
        
        a_pos_df: pd.DataFrame = self._obj
        if speed_col_name not in a_pos_df:
            ## compute linear speed
            raise NotImplementedError(f'missing requested speed column: "{speed_col_name}". Present columns: {list(a_pos_df.columns)}')
        
        movement_speed_variable = a_pos_df[speed_col_name].abs().values
        lap_epochs_df = a_pos_df.neuropy.detect_epoch_satisfying_condition(is_condition_satisfied = (movement_speed_variable > minimum_run_speed),
                                                                           minimum_epoch_duration=minimum_epoch_duration, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec,
                                                                           time_col_name='t')
        return lap_epochs_df


    def detect_general_non_running_epochs(self, max_run_speed: float = 2.0, minimum_epoch_duration: float = 0.5, merging_adjacent_max_separation_sec: float = 0.01, speed_col_name: str ='speed_xy') -> pd.DataFrame:
        """ 
        Returns an Epochs objects describe time frames where the animal is above a certain speed
        
        Usage:
            a_sess = curr_active_pipeline.filtered_sessions[k]
            non_running_epochs_df: pd.DataFrame = a_sess.position.compute_speed_info().position.detect_general_non_running_epochs(max_run_speed = 2.0)
            non_running_epochs_df

        """
        from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor
        
        a_pos_df: pd.DataFrame = self.compute_speed_info()

        if speed_col_name not in a_pos_df:
            ## compute linear speed
            raise NotImplementedError(f'missing requested speed column: "{speed_col_name}". Present columns: {list(a_pos_df.columns)}')
        
        movement_speed_variable = a_pos_df[speed_col_name].abs().values
        non_running_epochs_df: pd.DataFrame = a_pos_df.neuropy.detect_epoch_satisfying_condition(is_condition_satisfied = (movement_speed_variable <= max_run_speed),
                                                                            minimum_epoch_duration=minimum_epoch_duration, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec,
                                                                            time_col_name='t')
        return non_running_epochs_df



    
def _subfn_build_or_add_traces_df(df: Optional[pd.DataFrame], traces: NDArray, col_suffix: str='') -> pd.DataFrame:
    if traces.ndim == 1:
        traces = traces.reshape(1, -1) # required before setting ndim
        
    ndim = traces.shape[0]
    assert ndim <= 3, "Maximum possible dimension of position is 3"
         
    if df is None:
        x = traces[0].flatten().copy()
        df = pd.DataFrame({f'x{col_suffix}': x})
    else:
        x = traces[0]
        df[f"x{col_suffix}"] = x.flatten().copy()
        
    if ndim >= 2:
        y = traces[1]
        df[f"y{col_suffix}"] = y.flatten().copy()
    if ndim >= 3:
        z = traces[2]
        df[f"z{col_suffix}"] = z.flatten().copy()
    return df



""" --- """
class Position(HDFMixin, PositionDimDataMixin, PositionComputedDataMixin, ConcatenationInitializable, StartStopTimesMixin, TimeSlicableObjectProtocol, DataFrameRepresentable, DataWriter):
    
    def __init__(self, pos_df: pd.DataFrame, metadata=None) -> None:
        """[summary]
        Args:
            pos_df (pd.DataFrame): Each column is a pd.Series(["t", "x", "y"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        super().__init__(metadata=metadata)
        self._df = pos_df # set to the laps dataframe
        self._df = self._df.sort_values(by=[self.time_variable_name]) # sorts all values in ascending order
        
    def time_slice_indicies(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        included_indicies = self._df[self.time_variable_name].between(t_start, t_stop, inclusive='both') # returns a boolean array indicating inclusion in teh current lap
        return self._df.index[included_indicies]# note that this currently returns a Pandas.Series object. I could get the normal indicis by using included_indicies.to_numpy()
        
    @classmethod
    def init(cls, traces: np.ndarray, computed_traces: np.ndarray=None, t_start=0, sampling_rate=120, metadata=None, traces_rot=None):
        """ Comatibility initializer """
        if traces.ndim == 1:
            traces = traces.reshape(1, -1) # required before setting ndim
            
        ndim = traces.shape[0]
        assert ndim <= 3, "Maximum possible dimension of position is 3"
        
        # generate time vector:
        n_frames = traces.shape[1]
        duration = float(n_frames) / float(sampling_rate)
        t_stop = t_start + duration
        time = np.linspace(t_start, t_stop, n_frames)

        x = traces[0].flatten().copy()       
        df = pd.DataFrame({'t': time, 'x': x})
        if computed_traces is not None:
            if computed_traces.ndim >= 1:
                df["lin_pos"] = computed_traces[0].flatten().copy()        
        if ndim >= 2:
            y = traces[1]
            df["y"] = y.flatten().copy()
        if ndim >= 3:
            z = traces[2]
            df["z"] = z.flatten().copy()
            
        # df = _subfn_build_or_add_traces_df(df=None, traces=traces, col_suffix='')

        if traces_rot is not None:
            # print(f'WARN: traces_rot was passed in {traces_rot} but is UNUSED, part of the legacy protocol. It will be IGNORED.')
            # df = _subfn_build_or_add_traces_df(df=None, traces=traces, col_suffix='')
            df = _subfn_build_or_add_traces_df(df=df, traces=traces_rot, col_suffix='_rot')

        df = df.dropna(how='any', subset=['t', 'x', 'y'], inplace=False) ## drop any NaN values
        
        if metadata is None:
            metadata = {}

        _potential_metadata = {'t_start': t_start, 't_stop': t_stop, 'sampling_rate': sampling_rate}
        for k, v in _potential_metadata.items():
            if (k not in metadata) and (v is not None):
                metadata[k] = v

        return Position(df, metadata=metadata)


    ## Compatibility:
    @classmethod
    def legacy_from_dict(cls, dict_rep: dict):
        """ Tries to load the dict using previous versions of this code. """
        # Legacy fallback:
        print(f'Position falling back to legacy loading protocol...: dict_rep: {dict_rep}')
        traces_rot = dict_rep.pop('traces_rot', None)
        if traces_rot is not None:
            print(f'WARNING: Position class does not currently suppport "traces_rot", but they were passed in. Easy to implement.')

        speed = dict_rep.pop('speed', None)
        if speed is not None:
            print(f'WARNING: Position class does not currently suppport "speed", but they were passed in. Easy to implement.')

        return Position.init(**({'computed_traces': None, 't_start': 0, 'sampling_rate': 120, 'metadata': None} | dict_rep))
        
    # for PositionDimDataMixin
    @property
    def df(self):
        return self._df # for Position
    @df.setter
    def df(self, value):
        self._df = value # for Position

    @property
    def sampling_rate(self):
        # raise NotImplementedError
        return 1.0/np.nanmean(np.diff(self.time))

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        raise NotImplementedError
    

    def to_dict(self):
        data = {
            "df": self._df,
            "metadata": self.metadata,
        }
        return data

    @staticmethod
    def from_dict(d):
        return Position(
            d["df"],
            metadata=d["metadata"],
        )

    # @staticmethod
    # def is_fixed_sampling_rate(time):
    #     dt = np.diff(time)
        
    
    def to_dataframe(self):
        return self._df.copy()


    def __getstate__(self):
        state = self.__dict__.copy()
        return state


    def __setstate__(self, state):
        if '_df' not in state and '_data' in state:
            state['_df'] = state.pop('_data')
        self.__dict__.update(state)

    def speed_in_epochs(self, epochs: Epoch):
        assert isinstance(epochs, Epoch), "epochs must be neuropy.Epoch object"
        pass

    # for TimeSlicableObjectProtocol:
    def time_slice(self, t_start, t_stop):
        t_start, t_stop = self.safe_start_stop_times(t_start, t_stop)
        included_df = deepcopy(self._df)
        included_df = included_df[((included_df[self.time_variable_name] >= t_start) & (included_df[self.time_variable_name] <= t_stop))]
        return Position(included_df, metadata=deepcopy(self.metadata))
        

    @classmethod
    def from_separate_arrays(cls, t, x, y=None, z=None, lin_pos=None, metadata=None):
        temp_dict = {'t':t,'x':x}
        if y is not None:
            temp_dict['y'] = y
        if z is not None:
            temp_dict['z'] = z
        if lin_pos is not None:
            temp_dict['lin_pos'] = lin_pos
        return cls(pd.DataFrame(temp_dict), metadata=metadata)                            
        # return cls(traces=np.vstack((x, y)))
    
    
    # ConcatenationInitializable protocol:
    @classmethod
    def concat(cls, objList: Union[Sequence, np.array]):
        """ Concatenates the object list """
        if isinstance(objList, dict):
            objList = list(objList.values()) ## convert to a list

        # objList = np.array(objList)
        # concat_df = pd.concat([obj._df for obj in objList])
        # return cls(concat_df)
        merged_pos_df: pd.DataFrame = pd.concat([ensure_dataframe(p) for p in objList], ignore_index=True).drop_duplicates(subset=['t'], keep='first', inplace=False, ignore_index=True).sort_values(by='t', axis='index', ascending=True, inplace=False, ignore_index=True)
        merged_metadata = None
        try:                
            ## Build merged metadata:
            merged_metadata = pd.DataFrame.from_records([(p.metadata or {}) for p in objList])
            #TODO 2025-09-11 15:33: - [ ] Assert.all_equal(merged_metadata['sampling_rate'].tolist())
            merged_metadata = {'t_start': merged_metadata['t_start'].min(),
            't_stop': merged_metadata['t_stop'].max(),
            'sampling_rate': merged_metadata['sampling_rate'].min(),
            }
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            print(f'could not merge metadata with error: {e}')
            merged_metadata = None
            pass
        except Exception as e:
            raise
        
        return cls(pos_df=merged_pos_df, metadata=merged_metadata)


        

        
    def drop_dimensions_above(self, desired_ndim:int) -> None:
        """ modifies the internal dataframe to drop dimensions above a certain number. Always done in place, and returns None. """
        return self.df.position.drop_dimensions_above(desired_ndim, inplace=True)


    def print_debug_str(self):
        print('<core.Position :: np.shape(traces): {}\t time: {}\n duration: {}\n time[-1]: {}\n time[0]: {}\n sampling_rate: {}\n t_start: {}\n t_stop: {}\n>\n'.format(np.shape(self.traces), self.time,
            self.duration,
            self.time[-1],
            self.time[0],
            self.sampling_rate,
            self.t_start,
            self.t_stop)
        )
         

    def adding_lap_info(self, laps_df: pd.DataFrame):
        """ Updates the internal dataframe with the lap info 
        """
        curr_position_df: pd.DataFrame = self.to_dataframe() # get the position dataframe from the session
        curr_position_df = curr_position_df.position.adding_lap_info(laps_df=laps_df, inplace=False)

        # update:
        if 'lap' in curr_position_df:
            self._df['lap'] = curr_position_df['lap']
        if 'lap_dir' in curr_position_df:
            self._df['lap_dir'] = curr_position_df['lap_dir']


    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pos_obj: Position = long_one_step_decoder_1D.pf.position
            _pos_obj.to_hdf(hdf5_output_path, key='pos')
        """
        _df = self.to_dataframe()
        
        # Save the DataFrame using pandas
        # Unable to open/create file '/media/MAX/Data/KDIBA/gor01/one/2006-6-12_15-55-31/output/pipeline_results.h5'
        with pd.HDFStore(file_path) as store:
            _df.to_hdf(path_or_buf=store, key=key, format=kwargs.pop('format', 'table'), data_columns=kwargs.pop('data_columns',True), **kwargs)

        # Open the file with h5py to add attributes to the dataset
        with h5py.File(file_path, 'r+') as f:
            dataset = f[key]
            metadata = {
                'time_variable_name': self.time_variable_name,
                'sampling_rate': self.sampling_rate,
                't_start': self.t_start,
                't_stop': self.t_stop,
            }
            for k, v in metadata.items():
                dataset.attrs[k] = v

    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "Position":
        """ Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = Position.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        # Read the DataFrame using pandas
        pos_df = pd.read_hdf(file_path, key=key)

        # Open the file with h5py to read attributes
        with h5py.File(file_path, 'r') as f:
            dataset = f[key]
            metadata = {
                'time_variable_name': dataset.attrs['time_variable_name'],
                'sampling_rate': dataset.attrs['sampling_rate'],
                't_start': dataset.attrs['t_start'],
                't_stop': dataset.attrs['t_stop'],
            }

        # Reconstruct the object using the class constructor
        _out = cls(pos_df=pos_df, metadata=metadata)
        _out.filename = file_path # set the filename it was loaded from
        return _out



    def fixup_legacy(self):
        """ replaces the legacy properties with the modern ones

        """
        state = self.__getstate__()
        self.__setstate__(state)
        return self

