from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
from warnings import warn
from copy import deepcopy
from typing import Optional, Union, List, Dict, Any, Tuple
from nptyping import NDArray
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch, EpochsAccessor


if TYPE_CHECKING:
    from neuropy.core import Position
    from neuropy.core.session.dataSession import DataSession

from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable
from neuropy.utils.mixins.print_helpers import SimplePrintable

from .datawriter import DataWriter
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeSlicableObjectProtocol, TimeSlicableIndiciesMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, deduplicate_epochs # for EpochsAccessor's .get_non_overlapping_df()

## Import:
# from neuropy.core.laps import Laps


@pd.api.extensions.register_dataframe_accessor("laps_accessor")
class LapsAccessor(EpochsAccessor):
    """ A Pandas DataFrame-based Laps helper.
    
    from neuropy.core.laps import LapsAccessor
    
    
    """
    
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj = self.update_column_datatypes()
        

    @staticmethod
    def _validate(obj):
        # verify there is a column for timestamps ('t') and a column for at least 1D positions ('x')        
        # if "lap_id" not in obj.columns:
        #     raise AttributeError("Must have at least one lap identity variable: specifically 'lap_id' for LapsAccessor.")
        return EpochsAccessor._validate(obj)


    # for PositionDimDataMixin & PositionComputedDataMixin
    @property
    def df(self):
        return self._obj 
    @df.setter
    def df(self, value):
        self._obj = value
        
    # ==================================================================================================================== #
    # Properties moved from Laps class                                                                                    #
    # ==================================================================================================================== #

    @property
    def lap_id(self) -> NDArray:
        return self._obj['lap_id'].to_numpy()

    @property
    def maze_id(self) -> NDArray:
        return self._obj['maze_id'].to_numpy()

    @property
    def n_laps(self) -> NDArray:
        return len(self.lap_id)

    @property
    def starts(self) -> NDArray:
        return self._obj['start'].to_numpy()

    @property
    def stops(self) -> NDArray:
        return self._obj['stop'].to_numpy()

    # ==================================================================================================================== #
    # Instance methods moved from Laps class                                                                              #
    # ==================================================================================================================== #

    def filtered_by_lap_flat_index(self, lap_indicies) -> pd.DataFrame:
        """ Returns a dataframe filtered by lap flat indices """
        return self.filtered_by_lap_id(self.lap_id[lap_indicies])

    def filtered_by_lap_id(self, lap_ids) -> pd.DataFrame:
        """ Returns a dataframe filtered by lap IDs """
        return self._obj[np.isin(self.lap_id, lap_ids)]

    def update_maze_id_if_needed(self, t_start: float, t_delta: float, t_end: float) -> None:
        """ adds the 'maze_id' column to the internal dataframe if needed.

        Usage:
            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            laps_df = laps_df.laps_accessor.update_maze_id_if_needed(t_start, t_delta, t_end)
        """
        updated_df = self._obj.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end, replace_existing=True, labels_column_name='lap_id')
        self._obj = updated_df
        return None

    def update_lap_dir_from_smoothed_velocity(self, pos_input: Union[Position, DataSession]) -> None:
        """ Updates lap direction from smoothed velocity """
        self._obj = self._perform_compute_lap_dir_from_smoothed_velocity(self._obj, pos_input, replace_existing=True)
        assert 'is_LR_dir' in self._obj.columns, f"'is_LR_dir' is still missing even after adding it?!?"

    def adding_true_decoder_identifier(self, t_start: float, t_delta: float, t_end: float, labels_column_name: str = 'lap_id') -> pd.DataFrame:
        """ adds the 'maze_id' column to the internal dataframe if needed.

        Usage:
            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            laps_df = laps_df.laps_accessor.adding_true_decoder_identifier(t_start, t_delta, t_end)
        """
        filter_epochs: pd.DataFrame = self._obj.epochs.get_valid_df()
        filter_epochs = filter_epochs.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end, replace_existing=True, labels_column_name=labels_column_name)

        assert 'maze_id' in filter_epochs
        assert 'lap_dir' in filter_epochs
        # Creates Columns: 'truth_decoder_name':
        lap_dir_keys = ['LR', 'RL']
        maze_id_keys = ['long', 'short']
        filter_epochs['truth_decoder_name'] = filter_epochs['maze_id'].map(dict(zip(np.arange(len(maze_id_keys)), maze_id_keys))) + '_' + filter_epochs['lap_dir'].map(dict(zip(np.arange(len(lap_dir_keys)), lap_dir_keys)))

        self._obj[['maze_id', 'truth_decoder_name']] = filter_epochs[['maze_id', 'truth_decoder_name']]
        return filter_epochs

    def filter_to_valid(self) -> pd.DataFrame:
        """ Returns a dataframe with only valid laps """
        original_laps_epoch_df = self._obj.epochs.get_valid_df()
        filtered_laps_epoch_df = self.ensure_valid_laps_epochs_df(original_laps_epoch_df)
        return filtered_laps_epoch_df

    def trimmed_to_non_overlapping(self) -> pd.DataFrame:
        """ Returns a dataframe with non-overlapping laps """
        return self.trim_overlapping_laps(self._obj)[0]

    def get_lap_flat_indicies(self, lap_id) -> np.ndarray:
        """ Gets flat indices for a specific lap """
        return self._obj.loc[lap_id, ['start_spike_index', 'end_spike_index']].to_numpy()

    def get_lap_times(self, lap_id) -> np.ndarray:
        """ Gets start/stop times for a specific lap """
        return self._obj.loc[lap_id, ['start', 'stop']].to_numpy()

    def time_slice(self, t_start=None, t_stop=None) -> pd.DataFrame:
        """ Time slices the laps dataframe """
        included_indicies = (((self._obj.start >= t_start) & (self._obj.start <= t_stop)) & 
                            ((self._obj.stop >= t_start) & (self._obj.stop <= t_stop)))
        return self._obj[included_indicies]

    def update_column_datatypes(self) -> pd.DataFrame:
        """ Updates the datatypes of the dataframe to the correct type if they exist
        """
        if 'lap_id' in self._obj.columns:
            self._obj[['lap_id']] = self._obj[['lap_id']].astype('int')
        if 'maze_id' in self._obj.columns:
            self._obj[['maze_id']] = self._obj[['maze_id']].astype('int')
        if set(['start_spike_index','end_spike_index']).issubset(self._obj.columns):
            self._obj[['start_spike_index', 'end_spike_index']] = self._obj[['start_spike_index', 'end_spike_index']].astype('int')
            self._obj['num_spikes'] = self._obj['end_spike_index'] - self._obj['start_spike_index'] # builds 'num_spikes'
        if 'lap_dir' in self._obj.columns:
            # Either way, ensure that the lap_dir is an 'int' column.
            self._obj['lap_dir'] = self._obj['lap_dir'].astype('int')
        if 'label' in self._obj.columns:               
            self._obj['label'] = self._obj['lap_id'].astype('str') # add the string "label" column
        return self._obj

    def to_Laps_obj(self, metadata=None) -> "Laps":
        """ builds a Laps object from the LapsAccessor's dataframe 
        Usage:
            pos_df.position.to_Position_obj()
        """
        return Laps(self._obj, metadata=metadata)


    def as_epoch_obj(self) -> Epoch:
        """ Converts into a core.Epoch object containing the time periods """
        return Epoch(self._obj.copy())


    # @function_attributes(short_name=None, tags=['laps', 'lap_dir', 'modern'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-16 06:10', related_items=[])
    def compute_lap_dir_from_net_displacement(self, global_session: Union[Position, DataSession], replace_existing:bool=True, **kwargs) -> pd.DataFrame:
        """ 2025-07-16 - uses the smoothed velocity to determine the proper lap direction

        Adds/Updates Columns in laps_df: ['is_LR_dir', 'lap_dir']
        
        for LR_dir, values become more positive with time

        Usage:

            global_session = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name])
            active_global_laps_df = active_global_laps_df.laps_accessor.compute_lap_dir_from_net_displacement(global_session=global_session, replace_existing=True)
            active_global_laps_df

        """
        return self._perform_compute_lap_dir_from_net_displacement(laps_df=self._obj, global_session=global_session, replace_existing=replace_existing, **kwargs)
        

    def get_valid_laps_epochs_df(self, rebuild_lap_id_columns=True) -> pd.DataFrame:
        """ De-duplicates, sorts, and filters by duration any potential laps. Pure (does not modify `original_laps_epoch_df`)
        Usage:
            override_laps_df = override_laps_df.laps_accessor.get_valid_laps_epochs_df(rebuild_lap_id_columns=True)
        """
        return self.ensure_valid_laps_epochs_df(original_laps_epoch_df=self._obj, rebuild_lap_id_columns=rebuild_lap_id_columns)
    

    def update_computed_columns(self, t_start: Optional[float] = None, t_delta: Optional[float] = None, t_end: Optional[float] = None,
                            global_session: Optional[Union[Position, DataSession]] = None,
                            replace_existing: bool = True) -> pd.DataFrame:
        """ De-duplicates, sorts, and filters by duration any potential laps. Pure (does not modify `original_laps_epoch_df`)
        Usage:
            override_laps_df = override_laps_df.laps_accessor.update_computed_columns(replace_existing=True)
        """
        return self._perform_update_dataframe_computed_vars(laps_df=self._obj, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session, replace_existing=replace_existing)
        
    # ==================================================================================================================================================================================================================================================================================== #
    # Classmethods                                                                                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #

    
    # @function_attributes(short_name=None, tags=['laps', 'lap_dir', 'modern'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-16 06:10', related_items=[])
    @classmethod
    def _perform_compute_lap_dir_from_net_displacement(cls, laps_df: pd.DataFrame, global_session: Union[Position, DataSession], replace_existing:bool=True) -> pd.DataFrame:
        """ 2025-07-16 - uses the smoothed velocity to determine the proper lap direction

        Adds/Updates Columns in laps_df: ['is_LR_dir', 'lap_dir']
        
        for LR_dir, values become more positive with time

        """
        if (not replace_existing):
            if ('is_LR_dir' in laps_df.columns) and ('lap_dir' in laps_df.columns):
                print(f'WARN: (replace_existing == False) and ["is_LR_dir", "lap_dir"] already exist in laps_df. Skipping recomputation.')
                return laps_df

        
        from neuropy.core.position import Position
        
        n_laps: int = np.shape(laps_df)[0]
        if isinstance(global_session, Position):
            global_pos_obj = global_session # passed variable is already a Position object
        else:
            # passed variable is hopefully a DataSession: Extract the position from the passed in session.
            global_pos_obj = global_session.position
            
        global_pos_obj.compute_higher_order_derivatives()
        global_pos_obj.compute_smoothed_position_info()
        
        pos_df: pd.DataFrame = global_pos_obj.to_dataframe()

        ## adds the 'lap' and 'lap_dir' columns -- always do this so they correctly correspond to any updated laps
        pos_df = pos_df.position.adding_lap_info(laps_df=laps_df, inplace=False)

        # Filter rows based on column: 'lap'
        pos_df = pos_df[pos_df['lap'].notna()]
        
        ## 2025-07-16 05:45 - Compute instead based on total displacement -- lap_start, lap_end
        lap_displacement_df: pd.DataFrame = pos_df.groupby(['lap']).agg(lin_pos_first=('lin_pos', 'first'), lin_pos_last=('lin_pos', 'last')).reset_index() ## find net displacement
        lap_displacement_df['displacement'] = lap_displacement_df['lin_pos_last'] - lap_displacement_df['lin_pos_first']
        lap_displacement_df['is_LR_dir'] = (lap_displacement_df['displacement'] > 0.0) # increasing values => LR_dir
        lap_displacement_df['lap_dir'] = np.logical_not(lap_displacement_df['is_LR_dir']).astype(int) # 1 for RL, 0 for LR
        # lap_displacement_df = lap_displacement_df.set_index('lap')

        # Drop existing columns in 'laps_df' that will be replaced by the `lap_displacement_df` versions. 
        # If we don't do this we end up with duplicate columns like: ['lap_x', 'is_LR_dir_x', 'lap_dir_y', 'lap_dir_x', 'lap_y', 'is_LR_dir_y', 'lap_dir_y', 'lap', 'is_LR_dir', 'lap_dir'
        columns_to_drop = ['lap', 'is_LR_dir', 'lap_dir']
        existing_columns = [col for col in columns_to_drop if col in laps_df.columns]
        if existing_columns:
            laps_df = laps_df.drop(columns=existing_columns)

        ## Add in corrected columns to laps_df:
        laps_df = laps_df.merge(lap_displacement_df[['lap', 'is_LR_dir', 'lap_dir']], left_on='lap_id', right_on='lap', how='left')
        
        ## Update the pos_df now with the new info
        global_pos_obj.adding_lap_info(laps_df=laps_df)

        return laps_df


    # ==================================================================================================================== #
    # Class methods moved from Laps class                                                                                 #
    # ==================================================================================================================== #

    @classmethod
    def trim_overlapping_laps(cls, global_laps_df: pd.DataFrame, debug_print=False) -> Tuple[pd.DataFrame, pd.Index]:
        """ 2023-10-27 9pm - trims overlaps by removing the overlap from the even_global_laps 

        Returns: (modified_dataframe, changed_indices)
        """
        safe_trim_delta: float = 10.0 * (1.0/30.0)  # 10 samples assuming 30Hz sampling

        global_laps_df = global_laps_df.copy()

        even_global_laps_df = global_laps_df[global_laps_df.lap_dir == 0]
        odd_global_laps_df = global_laps_df[global_laps_df.lap_dir == 1]

        even_laps_portion = even_global_laps_df.epochs.to_PortionInterval()
        odd_laps_portion = odd_global_laps_df.epochs.to_PortionInterval()

        # Get whichever starts earlier
        if (even_global_laps_df.epochs.t_start < odd_global_laps_df.epochs.t_start):
            intersecting_portion = even_laps_portion.intersection(odd_laps_portion)
            intersecting_epochs = Epoch.from_PortionInterval(intersecting_portion)

            even_epochs_with_changes = np.where(np.isin(even_global_laps_df.stop, intersecting_epochs.stops))
            global_laps_end_change_indicies = even_global_laps_df.index[even_epochs_with_changes]
            desired_stops = intersecting_epochs.starts - safe_trim_delta
        else:
            intersecting_portion = odd_laps_portion.intersection(even_laps_portion)
            intersecting_epochs = Epoch.from_PortionInterval(intersecting_portion)

            odd_epochs_with_changes = np.where(np.isin(odd_global_laps_df.stop, intersecting_epochs.stops))
            global_laps_end_change_indicies = odd_global_laps_df.index[odd_epochs_with_changes]
            desired_stops = intersecting_epochs.starts - safe_trim_delta

        if debug_print:
            print(f'intersecting_epochs: {intersecting_epochs}')
            backup_values = global_laps_df.loc[global_laps_end_change_indicies, 'stop']
            print(f'backup_values: {backup_values}')
            print(f'desired_stops: {desired_stops}')

        global_laps_df.loc[global_laps_end_change_indicies, 'stop'] = desired_stops
        global_laps_df.loc[global_laps_end_change_indicies, 'duration'] = (global_laps_df.loc[global_laps_end_change_indicies, 'stop'] - 
                                                                            global_laps_df.loc[global_laps_end_change_indicies, 'start'])
        global_laps_df.loc[global_laps_end_change_indicies, 'end_t_rel_seconds'] = global_laps_df.loc[global_laps_end_change_indicies, 'stop']

        if debug_print:
            print("WARN: .trim_overlapping_laps(...): need to recompute ['start_position_index', 'end_position_index', 'start_spike_index', 'end_spike_index', 'num_spikes'] for the laps after calling trim_overlapping_laps()!")

        return global_laps_df, global_laps_end_change_indicies

    # @function_attributes(short_name=None, tags=['PURE'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-16 08:14', related_items=[])
    @classmethod
    def ensure_valid_laps_epochs_df(cls, original_laps_epoch_df: pd.DataFrame, rebuild_lap_id_columns=True) -> pd.DataFrame:
        """ De-duplicates, sorts, and filters by duration any potential laps. Pure (does not modify `original_laps_epoch_df`) """
        laps_epoch_df: pd.DataFrame = deepcopy(original_laps_epoch_df)

        # Filter rows based on column: 'duration'
        if 'duration' not in laps_epoch_df.columns:
            laps_epoch_df['duration'] = laps_epoch_df['stop'] - laps_epoch_df['start']

        laps_epoch_df = laps_epoch_df[laps_epoch_df['duration'] > 1]
        laps_epoch_df = laps_epoch_df[laps_epoch_df['duration'] <= 30]
        # Drop duplicate rows in columns: 'start', 'stop'
        laps_epoch_df = laps_epoch_df.drop_duplicates(subset=['start', 'stop'])
        # Sort by column: 'start' (ascending)
        laps_epoch_df = laps_epoch_df.sort_values(['start']).reset_index(drop=True)

        ## Rebuild lap_id and label column:
        if rebuild_lap_id_columns or ('lap_id' not in laps_epoch_df):
            laps_epoch_df['lap_id'] = (laps_epoch_df.index + 1).astype('int') ## set lap_id from the index
        else:
            laps_epoch_df[['lap_id']] = laps_epoch_df[['lap_id']].astype('int')

        if rebuild_lap_id_columns or ('label' not in laps_epoch_df):
            laps_epoch_df['label'] = laps_epoch_df['lap_id'].astype('str') # add the string "label" column
        else:
            laps_epoch_df['label'] = laps_epoch_df['label'].astype('str')

        ## Ensure correct datatypes in the end                    
        laps_epoch_df = laps_epoch_df.laps_accessor.update_column_datatypes()
        
        return laps_epoch_df


    @classmethod
    def _perform_update_dataframe_computed_vars(cls, laps_df: pd.DataFrame,
                            t_start: Optional[float] = None, t_delta: Optional[float] = None, t_end: Optional[float] = None,
                            global_session: Optional[Union[Position, DataSession]] = None,
                            replace_existing: bool = True) -> pd.DataFrame:
        """ Updates computed variables in the dataframe
        
        Internally calls: .get_valid_laps_epochs_df(rebuild_lap_id_columns=True)
        
        Updates global_session.position if `global_session` is not None
        Updates columns: ['lap_dir', 'is_LR_dir', 'num_spikes', '
        
        Usage:
            laps_df = cls._perform_update_dataframe_computed_vars(laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session, replace_existing=True)
        INSTEAD USE:
            laps_df = laps_df.laps_accessor.update_computed_columns(t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session, replace_existing=True)
        
        """
        if 'lap_id' not in laps_df.columns:
            laps_df = laps_df.laps_accessor.get_valid_laps_epochs_df(rebuild_lap_id_columns=True)
                    
        if ((t_start is not None) and (t_delta is not None) and (t_end is not None)):
            # computes 'track_id' from t_start, t_delta, and t_end where t_delta corresponds to the transition point (track change).
            laps_df = ensure_dataframe(laps_df).epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end, replace_existing=True, labels_column_name='lap_id')

        laps_df = laps_df.laps_accessor.update_column_datatypes()

        if set(['start_spike_index','end_spike_index']).issubset(laps_df.columns):
            laps_df['num_spikes'] = laps_df['end_spike_index'] - laps_df['start_spike_index'] # builds 'num_spikes'

        needs_recompute: bool = False
        ## ['lap_dir', 'is_LR_dir']
        if (('lap_dir' not in laps_df.columns) or ('is_LR_dir' not in laps_df.columns)):
            # compute the lap_dir if that field doesn't exist:
            if global_session is not None:
                ## computes proper 'is_LR_dir' and 'lap_dir' columns:
                needs_recompute = True

            else:
                # No global_session or position passed, using old even/odd 'lap_dir' determination.
                if (('lap_dir' not in laps_df) or replace_existing):
                    if (('lap_dir' in laps_df) and replace_existing):
                        print(f"WARNING: No global_session or position passed but replace_existing == True!\n\tWould replace existing 'lap_dir' column: {laps_df['lap_dir']} using old even/odd 'lap_dir' determination.")
                    else:
                        print(f"WARNING: No global_session or position passed but there is no existing 'lap_dir' column, using old even/odd 'lap_dir' determination.")

                    # raise NotImplementedError(f"THIS SHOULD NEVER HAPPEN ANYMORE, HOW IS IT?")
                    warn(f"THIS SHOULD NEVER HAPPEN ANYMORE, HOW IS IT?")
                    laps_df['lap_dir'] = np.full_like(laps_df['lap_id'].to_numpy(), -1)
                    # laps_df.loc[np.logical_not(np.isnan(laps_df.lap_id.to_numpy())), 'lap_dir'] = np.mod(laps_df.loc[np.logical_not(np.isnan(laps_df.lap_id.to_numpy())), 'lap_id'], 2)

        elif (replace_existing and (global_session is not None)): 
            needs_recompute = True
        else:
            pass
        
        if needs_recompute:
            laps_df = laps_df.laps_accessor.compute_lap_dir_from_net_displacement(global_session=global_session) # adds 'is_LR_dir'


        laps_df = laps_df.laps_accessor.get_valid_laps_epochs_df(rebuild_lap_id_columns=True)
        
        if (global_session is not None):
            ## Update the pos_df now with the new info
            global_session.position.adding_lap_info(laps_df=laps_df)
        
        return laps_df


    @classmethod
    def init_dataframe_from_mat_loaded_dict(cls, mat_file_loaded_dict: dict, time_variable_name='t_rel_seconds', absolute_start_timestamp=None,
                            t_start: Optional[float] = None, t_delta: Optional[float] = None, t_end: Optional[float] = None,
                            global_session: Optional[Union[Position, DataSession]] = None) -> pd.DataFrame:
        """ Builds a laps dataframe from a mat file dictionary """
        laps_df = pd.DataFrame(mat_file_loaded_dict)
        
        print('setting laps object.')
        if time_variable_name == 't_seconds':
            t_variable_column_names = ['start_t_seconds', 'end_t_seconds']
            t_variable = laps_df[t_variable_column_names].to_numpy()
        elif time_variable_name == 't_rel_seconds':
            t_variable_column_names = ['start_t_seconds', 'end_t_seconds']
            t_variable = laps_df[t_variable_column_names].to_numpy()
            # need to subtract off the absolute start timestamp
            t_variable = t_variable - absolute_start_timestamp
            laps_df[['start_t_rel_seconds', 'end_t_rel_seconds']] = t_variable
        else:
            t_variable_column_names = ['start_t', 'end_t']
            t_variable = laps_df[t_variable_column_names].to_numpy()

        # finally assign the 'start' and 'stop' time columns to the appropriate variable
        laps_df[['start','stop']] = t_variable
        
        laps_df = laps_df.laps_accessor.update_computed_columns(t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session, replace_existing=True)
        # laps_df = cls._perform_update_dataframe_computed_vars(laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session, replace_existing=True)


        return laps_df


    @classmethod
    def init_from_estimated_laps(cls, pos_t_rel_seconds: NDArray, desc_crossing_begining_idxs: NDArray, desc_crossing_ending_idxs: NDArray, asc_crossing_begining_idxs: NDArray, asc_crossing_ending_idxs: NDArray, debug_print=True,
                            t_start: Optional[float] = None, t_delta: Optional[float] = None, t_end: Optional[float] = None,
                            global_session: Optional[Union[Position, DataSession]] = None) -> pd.DataFrame:
        """Builds a laps dataframe from the output of the neuropy.analyses.laps.estimate_laps function."""

        custom_test_laps_df = pd.DataFrame({
            'start_position_index': np.concatenate([desc_crossing_begining_idxs, asc_crossing_begining_idxs]),
            'end_position_index': np.concatenate([desc_crossing_ending_idxs, asc_crossing_ending_idxs]),
            'lap_dir': np.concatenate([np.zeros_like(desc_crossing_begining_idxs), np.ones_like(asc_crossing_begining_idxs)])
        })

        # IMPORTANT 2023-10-27 - iterate through the pairs and insure no overlap between indicies
        custom_test_laps_df = custom_test_laps_df.sort_values(by=['start_position_index']).reset_index(drop=True)

        prev_index = None
        prev_end_value = None
        indicies_to_change = {}

        for index, row in custom_test_laps_df.iterrows():
            if prev_end_value is not None:
                if prev_end_value > row['start_position_index']:
                    indicies_to_change[prev_index] = (row['start_position_index'] - 1)
            prev_end_value = row['end_position_index']
            prev_index = index

        ## Make changes:
        if len(indicies_to_change) > 0:
            if debug_print:
                print(f'WARN: from_estimated_laps(...) found {len(indicies_to_change)} indicies to change:\n\tindicies_to_change: {indicies_to_change}')
        for an_index, a_new_end_pos_index in indicies_to_change.items():
            custom_test_laps_df.loc[an_index, 'end_position_index'] = a_new_end_pos_index

        # Get start/end times from the indicies
        custom_test_laps_df['start_t_rel_seconds'] = np.array([pos_t_rel_seconds[an_idx] for an_idx in custom_test_laps_df['start_position_index'].to_numpy()])
        custom_test_laps_df['end_t_rel_seconds'] = np.array([pos_t_rel_seconds[an_idx] for an_idx in custom_test_laps_df['end_position_index'].to_numpy()])
        custom_test_laps_df['start'] = custom_test_laps_df['start_t_rel_seconds']
        custom_test_laps_df['stop'] = custom_test_laps_df['end_t_rel_seconds']
        # Sort the laps based on the start time, reset the index, and finally assign lap_id's from the sorted laps
        custom_test_laps_df = custom_test_laps_df.sort_values(by=['start']).reset_index(drop=True)
        custom_test_laps_df['lap_id'] = (custom_test_laps_df.index + 1)
        custom_test_laps_df['label'] = custom_test_laps_df['lap_id']
        custom_test_laps_df = custom_test_laps_df.laps_accessor.filter_to_valid() ## drop invalid/zero index ones first
        # #TODO 2025-07-16 11:38: - [ ] exception occured: CapturedException(Cannot convert non-finite values (NA or inf) to integer, traceback=/home/halechr/repos/Spike3D/.venv/lib/python3.9/site-packages/pandas/core/dtypes/astype.py:182<fn: _astype_float_to_int_nansafe>: pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer)
        custom_test_laps_df = custom_test_laps_df.laps_accessor.update_computed_columns(t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session, replace_existing=True)
        custom_test_laps_df = custom_test_laps_df.laps_accessor.filter_to_valid()
        return custom_test_laps_df




# TODO: implement: NeuronUnitSlicableObjectProtocol, StartStopTimesMixin, TimeSlicableObjectProtocol
class Laps(Epoch):
    # epoch column labels: ["start", "stop", "label"]
    df_all_fieldnames = ['lap_id','maze_id','start_spike_index', 'end_spike_index', 'start_t', 'end_t', 'start_t_seconds', 'end_t_seconds', 'duration_seconds']
    
    def __init__(self, laps: pd.DataFrame, metadata=None) -> None:
        """[summary]
        Args:
            laps (pd.DataFrame): Each column is a pd.Series(["start", "stop", "label"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        super().__init__(laps, metadata=metadata)
        # self._data = laps # set to the laps dataframe
        self._df = self._df.laps_accessor.update_computed_columns(replace_existing=False) ## DO NOT allow replacement of the good epochs with the bad ones.
        self._df = self._df.laps_accessor.filter_to_valid()
        self._df = self._df.sort_values(by=['start']) # sorts all values in ascending order

    @property
    def _data(self):
        """ 2023-10-27 - a passthrough property for backwards compatibility. After adapting to a subclass of Epoch, the internal property is known as `self._df` not `self._data` """
        return self._df
    @_data.setter
    def _data(self, value):
        self._df = value


    @property
    def lap_id(self) -> NDArray:
        return self._data['lap_id'].to_numpy()
    
    @property
    def maze_id(self) -> NDArray:
        return self._data['maze_id'].to_numpy()
            
    @property
    def n_laps(self) -> NDArray:
        return len(self.lap_id)
        
    @property
    def starts(self) -> NDArray:
        return self._data['start'].to_numpy()

    @property
    def stops(self) -> NDArray:
        return self._data['stop'].to_numpy()
    
    def filtered_by_lap_flat_index(self, lap_indicies: NDArray):
        return self.filtered_by_lap_id(self.lap_id[lap_indicies])
    
    def filtered_by_lap_id(self, lap_ids: NDArray):
        sliced_copy = deepcopy(self) # get copy of the dataframe
        sliced_copy._data = self._df.laps_accessor.filtered_by_lap_id(lap_ids)
        return sliced_copy

    def update_maze_id_if_needed(self, t_start: float, t_delta: float, t_end: float) -> None:
        """ adds the 'maze_id' column to the internal dataframe if needed.
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        laps_obj: Laps = curr_active_pipeline.sess.laps
        laps_obj.update_maze_id_if_needed(t_start, t_delta, t_end)
        laps_df = laps_obj.to_dataframe()
        laps_df

        """
        self._df.laps_accessor.update_maze_id_if_needed(t_start, t_delta, t_end)
        return None


    def update_lap_dir_from_net_displacement(self, pos_input: Union[Position, DataSession], **kwargs) -> None:
        """ 2025-07-16 - uses the smoothed velocity to determine the proper lap direction

        Adds/Updates Columns in laps_df: ['is_LR_dir', 'lap_dir']
        
        for LR_dir, values become more positive with time

        Usage:

            global_session = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name])
            active_global_laps_df = active_global_laps_df.laps_accessor.compute_lap_dir_from_net_displacement(global_session=global_session, replace_existing=True)
            active_global_laps_df

        """
        self._df = self._df.laps_accessor.compute_lap_dir_from_net_displacement(global_session=pos_input, **kwargs)
        assert 'is_LR_dir' in self._df.columns, f"'is_LR_dir' is still missing even after adding it?!?"
        return None
        

    def update_computed_columns(self, t_start: Optional[float] = None, t_delta: Optional[float] = None, t_end: Optional[float] = None,
                            global_session: Optional[Union[Position, DataSession]] = None,
                            replace_existing: bool = True) -> None:
        """ De-duplicates, sorts, and filters by duration any potential laps. Pure (does not modify `original_laps_epoch_df`)
        Usage:
            override_laps_df = override_laps_df.laps_accessor.update_computed_columns(replace_existing=True)
        """
        self._df = self._df.laps_accessor.update_computed_columns(t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session, replace_existing=replace_existing)        
        return None
        



    def adding_true_decoder_identifier(self, t_start: float, t_delta: float, t_end: float, labels_column_name: str='lap_id') -> pd.DataFrame:
        """ adds the 'maze_id' column to the internal dataframe if needed.
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        laps_obj: Laps = curr_active_pipeline.sess.laps
        laps_obj.update_maze_id_if_needed(t_start, t_delta, t_end)
        laps_df = laps_obj.to_dataframe()
        laps_df

        """
        return self._df.laps_accessor.adding_true_decoder_identifier(t_start, t_delta, t_end, labels_column_name)

    def filter_to_valid(self) -> "Laps":
        filtered_laps_df = self._df.laps_accessor.filter_to_valid()
        return Laps(filtered_laps_df, metadata=self.metadata)

    def trimmed_to_non_overlapping(self) -> "Laps":
        trimmed_laps_df, _ = LapsAccessor.trim_overlapping_laps(self._df)
        return Laps(trimmed_laps_df, metadata=self.metadata)

    def get_lap_flat_indicies(self, lap_id):
        return self._df.laps_accessor.get_lap_flat_indicies(lap_id)

    def get_lap_times(self, lap_id):
        return self._df.laps_accessor.get_lap_times(lap_id)

    def as_epoch_obj(self) -> Epoch:
        """ Converts into a core.Epoch object containing the time periods """
        return self._df.laps_accessor.as_epoch_obj()

    def time_slice(self, t_start=None, t_stop=None):
        #TODO: #WM: Test this, it's not done! It should filter out the laps that occur outside of the start/end times that 
        time_sliced_df = self._df.laps_accessor.time_slice(t_start, t_stop)
        return Laps(time_sliced_df, metadata=self.metadata)

    # ==================================================================================================================== #
    # Methods to keep in Laps class (serialization, etc.)                                                                #
    # ==================================================================================================================== #

    def to_dataframe(self) -> pd.DataFrame:
        return self._data

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Laps":
        return cls(df)

    @staticmethod
    def from_dict(d: dict):
        return Laps((d.get('_df', None) or d.get('_data', None)), metadata = d.get('metadata', None))
        # return Laps(d['_data'], metadata = d.get('metadata', None))
        
    def to_dict(self):
        return self.__dict__
    
     ## For serialization/pickling:
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # for backwards compatibility with pre-Epoch baseclass versions of Laps loaded from pickle
        if '_df' not in state:
            assert '_data' in state
            state['_df'] = state.pop('_data', None)
        self.__dict__.update(state)
            
    
    # ==================================================================================================================== #
    # Class Methods delegated to LapsAccessor                                                                             #
    # ==================================================================================================================== #

    @classmethod
    def trim_overlapping_laps(cls, global_laps: "Laps", debug_print=False) -> "Laps":
        """ 2023-10-27 9pm - trims overlaps by removing the overlap from the even_global_laps """
        trimmed_laps_df, global_laps_end_change_indicies = LapsAccessor.trim_overlapping_laps(global_laps._df, debug_print)
        return Laps(trimmed_laps_df, metadata=global_laps.metadata), global_laps_end_change_indicies

    @classmethod
    def ensure_valid_laps_epochs_df(cls, original_laps_epoch_df: pd.DataFrame, rebuild_lap_id_columns=True) -> pd.DataFrame:
        """ De-duplicates, sorts, and filters by duration any potential laps """
        return LapsAccessor.ensure_valid_laps_epochs_df(original_laps_epoch_df, rebuild_lap_id_columns)

    @classmethod
    def _update_dataframe_computed_vars(cls, laps_df: pd.DataFrame,
                        t_start: Optional[float] = None, t_delta: Optional[float] = None, t_end: Optional[float] = None,
                        global_session: Optional[Union[Position, DataSession]] = None,
                        replace_existing: bool = True):
        """ this function is pretty bad. """
        return LapsAccessor._perform_update_dataframe_computed_vars(laps_df, t_start, t_delta, t_end, global_session, replace_existing)

    @classmethod
    def init_from_estimated_laps(cls, pos_t_rel_seconds, desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs, debug_print=True,
                            t_start: Optional[float] = None, t_delta: Optional[float] = None, t_end: Optional[float] = None,
                            global_session: Optional[Union[Position, DataSession]] = None) -> "Laps":
        """Builds a Laps object from the output of the neuropy.analyses.laps.estimate_laps function."""
        filtered_laps_df = LapsAccessor.init_from_estimated_laps(pos_t_rel_seconds, desc_crossing_begining_idxs=desc_crossing_begining_idxs, desc_crossing_ending_idxs=desc_crossing_ending_idxs, asc_crossing_begining_idxs=asc_crossing_begining_idxs, asc_crossing_ending_idxs=asc_crossing_ending_idxs,
                                                                  debug_print=debug_print,
                                                                  t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=global_session)
        return Laps(filtered_laps_df)

