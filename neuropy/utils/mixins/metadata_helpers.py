from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import pandas as pd

class DataframeMetadataProtocol:
    """ Allows general events (marked by a single point in time) represented as Pandas DataFrames to be easily time-sliced and manipulated along with their accompanying data without making a custom class.
    
    Generalized from SpikesAccessor on 2025-01-15 14:51 - refactored instantaneous-event functionality out into `TimePointEventAccessor` accessible via `a_df.time_point_event.adding_epochs_identity_column(....)`
    
    Examples: spikes_df, pos_df

        from neuropy.utils.mixins.metadata_helpers import DataframeMetadataProtocol, MetadataAccessor
        
    """
    @property
    def metadata(self) -> Dict:
        """ gets the dataframe's `df.attrs` dictionary metadata """
        if self._obj.attrs is None:
            self._obj.attrs = {} ## initialize to empty dict
        return self._obj.attrs
    @metadata.setter
    def metadata(self, value: Dict):
        self._obj.attrs = value



@pd.api.extensions.register_dataframe_accessor("metadata")
class MetadataAccessor(DataframeMetadataProtocol):
    """ Allows general events (marked by a single point in time) represented as Pandas DataFrames to be easily time-sliced and manipulated along with their accompanying data without making a custom class.
    
    Generalized from SpikesAccessor on 2025-01-15 14:51 - refactored instantaneous-event functionality out into `TimePointEventAccessor` accessible via `a_df.time_point_event.adding_epochs_identity_column(....)`
    
    Examples: spikes_df, pos_df

        from neuropy.utils.mixins.metadata_helpers import DataframeMetadataProtocol, MetadataAccessor
        
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj


    # def update_df_metadata(self, **updated_metadata):
    #     """ updates the metadata of the internal df from the explicit metadata"""
    #     if getattr(self._df, 'attrs', None) is None:
    #         self._df.attrs = {} ## setup df metadata
    #     if len(updated_metadata) > 0:
    #         self._df.attrs.update(**updated_metadata)
            