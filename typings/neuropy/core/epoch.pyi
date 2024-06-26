"""
This type stub file was generated by pyright.
"""

import pandas as pd
from typing import Optional, Union
from importlib import metadata
from nptyping import NDArray
from neuropy.utils.mixins.dataframe_representable import DataFrameInitializable, DataFrameRepresentable
from .datawriter import DataWriter
from neuropy.utils.mixins.print_helpers import OrderedMeta, SimplePrintable
from neuropy.utils.mixins.time_slicing import StartStopTimesMixin, TimeColumnAliasesProtocol, TimeSlicableObjectProtocol, TimeSlicedMixin
from neuropy.utils.mixins.HDF5_representable import HDFMixin

def find_data_indicies_from_epoch_times(a_df: pd.DataFrame, epoch_times: NDArray, t_column_names=..., atol: float = ..., not_found_action=..., debug_print=...) -> NDArray:
    """ returns the matching data indicies corresponding to the epoch [start, stop] times 
    epoch_times: S x 2 array of epoch start/end times


    skip_index: drops indicies that can't be found, meaning that the number of returned indicies might be less than len(epoch_times)


    Returns: (S, ) array of data indicies corresponding to the times.

    Uses:
        from neuropy.core.epoch import find_data_indicies_from_epoch_times

        selection_start_stop_times = deepcopy(active_epochs_df[['start', 'stop']].to_numpy())
        print(f'np.shape(selection_start_stop_times): {np.shape(selection_start_stop_times)}')

        test_epochs_data_df: pd.DataFrame = deepcopy(ripple_simple_pf_pearson_merged_df)
        print(f'np.shape(test_epochs_data_df): {np.shape(test_epochs_data_df)}')

        # 2D_search (for both start, end times):
        found_data_indicies = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=selection_start_stop_times)
        print(f'np.shape(found_data_indicies): {np.shape(found_data_indicies)}')

        # 1D_search (only for start times):
        found_data_indicies_1D_search = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=np.squeeze(selection_start_stop_times[:, 0]))
        print(f'np.shape(found_data_indicies_1D_search): {np.shape(found_data_indicies_1D_search)}')
        found_data_indicies_1D_search

        assert np.array_equal(found_data_indicies, found_data_indicies_1D_search)
    

    - [ ] TODO FATAL 2024-03-04 19:55 - This function is incorrect as it can return multiple matches for each passed time due to the tolerance. Unfinished.
        
    """
    ...

class NamedTimerange(SimplePrintable, metaclass=OrderedMeta):
    """ A simple named period of time with a known start and end time """
    def __init__(self, name, start_end_times) -> None:
        ...
    
    @property
    def t_start(self):
        ...
    
    @t_start.setter
    def t_start(self, t): # -> None:
        ...
    
    @property
    def duration(self):
        ...
    
    @property
    def t_stop(self):
        ...
    
    @t_stop.setter
    def t_stop(self, t): # -> None:
        ...
    
    def to_Epoch(self): # -> Epoch:
        ...
    


@pd.api.extensions.register_dataframe_accessor("epochs")
class EpochsAccessor(TimeColumnAliasesProtocol, TimeSlicedMixin, StartStopTimesMixin, TimeSlicableObjectProtocol):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals """
    _time_column_name_synonyms = ...
    _required_column_names = ...
    def __init__(self, pandas_obj) -> None:
        ...
    
    @property
    def is_valid(self): # -> Literal[True]:
        """ The dataframe is valid (because it passed _validate(...) in __init__(...) so just return True."""
        ...
    
    @property
    def starts(self):
        ...
    
    @property
    def midtimes(self):
        """ since each epoch is given by a (start, stop) time, the midtimes are the center of this epoch. """
        ...
    
    @property
    def stops(self):
        ...
    
    @property
    def t_start(self):
        ...
    
    @t_start.setter
    def t_start(self, t): # -> None:
        ...
    
    @property
    def duration(self):
        ...
    
    @property
    def t_stop(self):
        ...
    
    @property
    def durations(self):
        ...
    
    @property
    def n_epochs(self): # -> int:
        ...
    
    @property
    def labels(self):
        ...
    
    @property
    def extra_data_column_names(self): # -> list[Any]:
        """Any additional columns in the dataframe beyond those that exist by default. """
        ...
    
    @property
    def extra_data_dataframe(self) -> pd.DataFrame:
        """The subset of the dataframe containing additional information in its columns beyond that what is required. """
        ...
    
    def as_array(self) -> NDArray:
        ...
    
    def get_unique_labels(self):
        ...
    
    def get_start_stop_tuples_list(self): # -> list[tuple[Any, Any]]:
        """ returns a list of (start, stop) tuples. """
        ...
    
    def get_valid_df(self) -> pd.DataFrame:
        """ gets a validated copy of the dataframe. Looks better than doing `epochs_df.epochs._obj` """
        ...
    
    def get_non_overlapping_df(self, debug_print=...) -> pd.DataFrame:
        """ Returns a dataframe with overlapping epochs removed. """
        ...
    
    def get_epochs_longer_than(self, minimum_duration, debug_print=...) -> pd.DataFrame:
        """ returns a copy of the dataframe contining only epochs longer than the specified minimum_duration. """
        ...
    
    def time_slice(self, t_start, t_stop) -> pd.DataFrame:
        """ trim the epochs down to the provided time range
        
        """
        ...
    
    def label_slice(self, label) -> pd.DataFrame:
        ...
    
    def find_data_indicies_from_epoch_times(self, epoch_times: NDArray, atol: float = ..., t_column_names=...) -> NDArray:
        """ returns the matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        Uses:
            self.plots_data.epoch_slices
        
        - [X] FIXED 2024-03-04 19:55 - This function was peviously incorrect and could return multiple matches for each passed time due to the tolerance.

        """
        ...
    
    def matching_epoch_times_slice(self, epoch_times: NDArray, atol: float = ..., t_column_names=...) -> pd.DataFrame:
        """ slices the dataframe to return only the rows that match the epoch_times with some tolerance.
        
        Internally calls self.find_data_indicies_from_epoch_times(...)

        """
        ...
    
    def filtered_by_duration(self, min_duration=..., max_duration=...):
        ...
    
    @classmethod
    def from_PortionInterval(cls, portion_interval): # -> DataFrame[Any]:
        ...
    
    def to_PortionInterval(self): # -> Interval:
        ...
    
    def adding_active_aclus_information(self, spikes_df: pd.DataFrame, epoch_id_key_name: str = ..., add_unique_aclus_list_column: bool = ...) -> pd.DataFrame:
        """ 
        adds the columns: ['unique_active_aclus', 'n_unique_aclus'] 

        Usage:

            active_epochs_df = add_active_aclus_information(active_epochs_df, active_spikes_df, add_unique_aclus_list_column=True)

        """
        ...
    
    def adding_maze_id_if_needed(self, t_start: float, t_delta: float, t_end: float, replace_existing: bool = ..., labels_column_name: str = ...) -> pd.DataFrame:
        """ 2024-01-17 - adds the 'maze_id' column if it doesn't exist

        WARNING: does NOT modify in place!


        Usage:
            from neuropy.core.session.dataSession import Laps

            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            laps_obj: Laps = curr_active_pipeline.sess.laps
            laps_df = laps_obj.to_dataframe()
            laps_df = laps_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            laps_df

        """
        ...
    
    def to_dataframe(self) -> pd.DataFrame:
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        ...
    
    def to_Epoch(self) -> Epoch:
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        ...
    


class Epoch(HDFMixin, StartStopTimesMixin, TimeSlicableObjectProtocol, DataFrameRepresentable, DataFrameInitializable, DataWriter):
    """ An Epoch object holds one ore more periods of time (marked by start/end timestamps) along with their corresponding metadata.

    """
    def __init__(self, epochs: pd.DataFrame, metadata=...) -> None:
        """[summary]
        Args:
            epochs (pd.DataFrame): Each column is a pd.Series(["start", "stop", "label"])
            metadata (dict, optional): [description]. Defaults to None.
        """
        ...
    
    @property
    def starts(self):
        ...
    
    @property
    def stops(self):
        ...
    
    @property
    def t_start(self):
        ...
    
    @t_start.setter
    def t_start(self, t): # -> None:
        ...
    
    @property
    def duration(self):
        ...
    
    @property
    def t_stop(self):
        ...
    
    @property
    def durations(self):
        ...
    
    @property
    def midtimes(self):
        """ since each epoch is given by a (start, stop) time, the midtimes are the center of this epoch. """
        ...
    
    @property
    def n_epochs(self):
        ...
    
    @property
    def labels(self):
        ...
    
    def get_unique_labels(self):
        ...
    
    def get_named_timerange(self, epoch_name): # -> NamedTimerange:
        ...
    
    @property
    def metadata(self): # -> dict[Any, Any]:
        ...
    
    @metadata.setter
    def metadata(self, metadata): # -> None:
        """metadata compatibility"""
        ...
    
    @property
    def epochs(self) -> EpochsAccessor:
        """ a passthrough accessor to the Pandas dataframe `EpochsAccessor` to allow complete pass-thru compatibility with either Epoch or pd.DataFrame versions of epochs.
        Instead of testing whether it's an `Epoch` object or pd.DataFrame and then converting back and forth, should just be able to pretend it's a dataframe for the most part and use the `some_epochs.epochs.*` properties and methods.
        """
        ...
    
    def __repr__(self) -> str:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __len__(self): # -> int:
        """ allows using `len(epochs_obj)` and getting the number of epochs. """
        ...
    
    def str_for_concise_display(self) -> str:
        """ returns a minimally descriptive string like: '60 epochs in (17.9, 524.1)' that doesn't print all the array elements only the number of epochs and the first and last. """
        ...
    
    def str_for_filename(self) -> str:
        ...
    
    @property
    def __array_interface__(self):
        """ wraps the internal dataframe's `__array_interface__` which Pandas uses to provide numpy with information about dataframes such as np.shape(a_df) info.
        Allows np.shape(an_epoch_obj) to work.

        """
        ...
    
    def __getitem__(self, slice_): # -> NDArray[Any]:
        """ Allows pass-thru indexing like it were a numpy array.

        2024-03-07 Potentially more dangerous than helpful.

        having issue whith this being called with pd.Dataframe columns (when assuming a pd.DataFrame epochs format but actually an Epoch object)

        IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
               Occurs because `_slice == ['lap_id']` which doesn't pass the first check because it's a list of strings not a string itself
        Example:
            Error line `laps_df[['lap_id']] = laps_df[['lap_id']].astype('int')`
        """
        ...
    
    def time_slice(self, t_start, t_stop): # -> Epoch:
        ...
    
    def label_slice(self, label): # -> Epoch:
        ...
    
    def boolean_indicies_slice(self, boolean_indicies): # -> Epoch:
        ...
    
    def filtered_by_duration(self, min_duration=..., max_duration=...): # -> Epoch:
        ...
    
    @classmethod
    def filter_epochs(cls, curr_epochs: Union[pd.DataFrame, Epoch], pos_df: Optional[pd.DataFrame] = ..., spikes_df: pd.DataFrame = ..., require_intersecting_epoch: Epoch = ..., min_epoch_included_duration=..., max_epoch_included_duration=..., maximum_speed_thresh=..., min_inclusion_fr_active_thresh=..., min_num_unique_aclu_inclusions=..., debug_print=...) -> Epoch:
        """filters the provided replay epochs by specified constraints.

        Args:
            curr_epochs (Epoch): the epochs to filter on
            min_epoch_included_duration (float, optional): all epochs shorter than min_epoch_included_duration will be excluded from analysis. Defaults to 0.06.
            max_epoch_included_duration (float, optional): all epochs longer than max_epoch_included_duration will be excluded from analysis. Defaults to 0.6.
            maximum_speed_thresh (float, optional): epochs are only included if the animal's interpolated speed (as determined from the session's position dataframe) is below the speed. Defaults to 2.0 [cm/sec].
            min_inclusion_fr_active_thresh: minimum firing rate (in Hz) for a unit to be considered "active" for inclusion.
            min_num_unique_aclu_inclusions: minimum number of unique active cells that must be included in an epoch to have it included.

            save_on_compute (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            Epoch: the filtered epochs as an Epoch object

        NOTE: this really is a general method that works for any Epoch object or Epoch-type dataframe to filter it.

        TODO 2023-04-11 - This really belongs in the Epoch class or the epoch dataframe accessor. 

        """
        ...
    
    def to_dict(self, recurrsively=...): # -> dict[str, Any]:
        ...
    
    @staticmethod
    def from_dict(d: dict): # -> Epoch:
        ...
    
    def fill_blank(self, method=...): # -> None:
        ...
    
    def delete_in_between(self, t1, t2): # -> Epoch:
        ...
    
    def get_proportion_by_label(self, t_start=..., t_stop=...): # -> dict[Any, Any]:
        ...
    
    def count(self, t_start=..., t_stop=..., binsize=...): # -> NDArray[Any]:
        ...
    
    def to_neuroscope(self, ext=...): # -> Path:
        """ exports to a Neuroscope compatable .evt file. """
        ...
    
    @classmethod
    def from_neuroscope(cls, in_filepath): # -> Self:
        """ imports from a Neuroscope compatible .evt file.
        Usage:
            from neuropy.core.epoch import Epoch

            evt_filepath = Path('/Users/pho/Downloads/2006-6-07_16-40-19.bst.evt').resolve()
            # evt_filepath = Path('/Users/pho/Downloads/2006-6-08_14-26-15.bst.evt').resolve()
            evt_epochs: Epoch = Epoch.from_neuroscope(in_filepath=evt_filepath)
            evt_epochs

        """
        ...
    
    def as_array(self): # -> ndarray[Any, Any]:
        ...
    
    @classmethod
    def from_PortionInterval(cls, portion_interval, metadata=...): # -> Epoch:
        ...
    
    def to_PortionInterval(self):
        ...
    
    def get_non_overlapping(self, debug_print=...): # -> Epoch:
        """ Returns a copy with overlapping epochs removed. """
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
    def read_hdf(cls, file_path, key: str, **kwargs) -> Epoch:
        """  Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = Epoch.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        ...
    
    def to_dataframe(self) -> pd.DataFrame:
        ...
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame): # -> Self:
        ...
    
    def to_Epoch(self) -> Epoch:
        """ Ensures code exchangeability of epochs in either `Epoch` object / pd.DataFrame """
        ...
    


def ensure_dataframe(epochs: Union[Epoch, pd.DataFrame]) -> pd.DataFrame:
    """ 
        Usage:

        from neuropy.core.epoch import ensure_dataframe

    """
    ...

def subdivide_epochs(df: pd.DataFrame, subdivide_bin_size: float, start_col=..., stop_col=...) -> pd.DataFrame:
    """ splits each epoch into equally sized chunks determined by subidivide_bin_size.
    
    # Example usage
        from neuropy.core.epoch import subdivide_epochs, ensure_dataframe

        df: pd.DataFrame = ensure_dataframe(deepcopy(long_LR_epochs_obj)) 
        df['epoch_type'] = 'lap'
        df['interval_type_id'] = 666

        subdivide_bin_size = 0.100  # Specify the size of each sub-epoch in seconds
        subdivided_df: pd.DataFrame = subdivide_epochs(df, subdivide_bin_size)
        print(subdivided_df)

    """
    ...

