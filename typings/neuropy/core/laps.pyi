"""
This type stub file was generated by pyright.
"""

import pandas as pd
from typing import Optional, TYPE_CHECKING, Union
from pandas.core.frame import DataFrame
from neuropy.core.epoch import Epoch
from neuropy.core import Position
from neuropy.core.session.dataSession import DataSession

if TYPE_CHECKING:
    ...
class Laps(Epoch):
    df_all_fieldnames = ...
    def __init__(self, laps: pd.DataFrame, metadata=...) -> None:
        """[summary]
        Args:
            laps (pd.DataFrame): Each column is a pd.Series(["start", "stop", "label"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        ...
    
    @property
    def lap_id(self): # -> ndarray[Any, Any]:
        ...
    
    @property
    def maze_id(self): # -> ndarray[Any, Any]:
        ...
    
    @property
    def n_laps(self): # -> int:
        ...
    
    @property
    def starts(self): # -> ndarray[Any, Any]:
        ...
    
    @property
    def stops(self): # -> ndarray[Any, Any]:
        ...
    
    def filtered_by_lap_flat_index(self, lap_indicies): # -> Self:
        ...
    
    def filtered_by_lap_id(self, lap_ids): # -> Self:
        ...
    
    def update_maze_id_if_needed(self, t_start: float, t_delta: float, t_end: float) -> None:
        """ adds the 'maze_id' column to the internal dataframe if needed.
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        laps_obj: Laps = curr_active_pipeline.sess.laps
        laps_obj.update_maze_id_if_needed(t_start, t_delta, t_end)
        laps_df = laps_obj.to_dataframe()
        laps_df
        
        
        """
        ...
    
    def update_lap_dir_from_smoothed_velocity(self, pos_input: Union[Position, DataSession]) -> None:
        ...
    
    def filter_to_valid(self) -> Laps:
        ...
    
    def trimmed_to_non_overlapping(self) -> Laps:
        ...
    
    def to_dataframe(self): # -> DataFrame[Any]:
        ...
    
    def get_lap_flat_indicies(self, lap_id): # -> ndarray[Any, Any]:
        ...
    
    def get_lap_times(self, lap_id): # -> ndarray[Any, Any]:
        ...
    
    def as_epoch_obj(self): # -> Epoch:
        """ Converts into a core.Epoch object containing the time periods """
        ...
    
    @staticmethod
    def from_dict(d: dict): # -> Laps:
        ...
    
    def to_dict(self): # -> dict[str, Any]:
        ...
    
    def time_slice(self, t_start=..., t_stop=...): # -> Laps:
        ...
    
    def __getstate__(self): # -> dict[str, Any]:
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    
    @classmethod
    def trim_overlapping_laps(cls, global_laps: Laps, debug_print=...) -> Laps:
        """ 2023-10-27 9pm - trims overlaps by removing the overlap from the even_global_laps (assuming that even first... hmmm that might be a problem? No, because even is always first because it refers to the 0 index lap_id.

        Avoids major issues introduced by Portion library by first splitting into odd/even (adjacent epochs) and only considering the overlap between the adjacent ones.
        Then it gets the indicies of the ones that changed so it can manually update the stop times for those epochs only on the even epochs so the other column's data isn't lost like it is in the portion/Epoch methods.

        ## SHOOT: changing this will change the other computed numbers!! 

        Modifies: ['end_t_rel_seconds', 'stop', 'duration']

        Invalidates: ['start_position_index', 'end_position_index', 'start_spike_index', 'end_spike_index', 'num_spikes']


        TODO 2023-10-27 - could refactor to parent Epochs class?

        """
        ...
    
    @classmethod
    def ensure_valid_laps_epochs_df(cls, original_laps_epoch_df: pd.DataFrame, rebuild_lap_id_columns=...) -> pd.DataFrame:
        """ De-duplicates, sorts, and filters by duration any potential laps
        
        laps_epoch_obj: Epoch = deepcopy(global_laps).as_epoch_obj()
        original_laps_epoch_df = laps_epoch_obj.to_dataframe()        
        filtered_laps_epoch_df = cls.ensure_valid_laps_epochs_df(original_laps_epoch_df)

        """
        ...
    
    @classmethod
    def build_dataframe(cls, mat_file_loaded_dict: dict, time_variable_name=..., absolute_start_timestamp=..., t_start: Optional[float] = ..., t_delta: Optional[float] = ..., t_end: Optional[float] = ..., global_session: Optional[Union[Position, DataSession]] = ...): # -> DataFrame[Any]:
        ...
    
    @classmethod
    def from_estimated_laps(cls, pos_t_rel_seconds, desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs, debug_print=...): # -> Laps:
        """Builds a Laps object from the output of the neuropy.analyses.laps.estimate_laps function.
        Args:
            pos_t_rel_seconds ([type]): [description]
            desc_crossing_beginings ([type]): [description]
            desc_crossing_endings ([type]): [description]
            asc_crossing_beginings ([type]): [description]
            asc_crossing_endings ([type]): [description]

        Usage:
        
            position_obj = sess.position.linear_pos_obj
            position_obj.compute_higher_order_derivatives()
            pos_df = position_obj.compute_smoothed_position_info(N=N) ## Smooth the velocity curve to apply meaningful logic to it
            pos_df: pd.DataFrame = position_obj.to_dataframe()
            # If the index doesn't start at zero, it will need to for compatibility with the lap splitting logic because it uses the labels via "df.loc"
            if 'index_backup' not in pos_df.columns:
                pos_df['index_backup'] = pos_df.index  # Backup the current index to a new column
            # Drop rows with missing data in columns: 't', 'velocity_x_smooth' and 2 other columns. This occurs from smoothing
            pos_df = pos_df.dropna(subset=['t', 'x', 'x_smooth', 'velocity_x_smooth', 'acceleration_x_smooth'])    
            pos_df.reset_index(drop=True, inplace=True) # Either way, reset the index
            lap_change_indicies = _subfn_perform_estimate_lap_splits_1D(pos_df, hardcoded_track_midpoint_x=None, debug_print=debug_print) # allow smart midpoint determiniation
            (desc_crossing_begining_idxs, desc_crossing_midpoint_idxs, desc_crossing_ending_idxs), (asc_crossing_begining_idxs, asc_crossing_midpoint_idxs, asc_crossing_ending_idxs) = lap_change_indicies    
            custom_test_laps_obj = Laps.from_estimated_laps(pos_df['t'].to_numpy(), desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs) ## Get the timestamps corresponding to the indicies
            assert custom_test_laps_obj.n_laps > 0, f"estimation for {sess} produced no laps!"
            
        Returns:
            [type]: [description]
        """
        ...
    
    @classmethod
    def from_dataframe(cls, df): # -> Self:
        ...
    


