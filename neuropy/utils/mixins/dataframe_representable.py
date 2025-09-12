## Implementors can be faithfully (although maybe not completely) represented by a Pandas DataFrame. 
# dataframe_representable.py
from typing import Union
import pandas as pd

class DataFrameInitializable:
	""" Implementors can be initialized from a Pandas DataFrame. 
	"""
	@classmethod
	def from_dataframe(cls, df):
		raise NotImplementedError

class DataFrameRepresentable(DataFrameInitializable):
	def to_dataframe(self):
		raise NotImplementedError


def ensure_dataframe(epochs: Union[DataFrameRepresentable, pd.DataFrame]) -> pd.DataFrame:
    """ Ensures that the item are returned as an Pandas DataFrame, does nothing if they already are a DataFrame.
    Based off of to `ensure_Epoch(...)`/`ensure_dataframe(...)`
    Usage:
        from neuropy.utils.mixins.dataframe_representable import ensure_dataframe
    """
    if isinstance(epochs, pd.DataFrame):
        return epochs
    else:
        return epochs.to_dataframe()
	