## Implementors can be faithfully (although maybe not completely) represented by a Pandas DataFrame. 
# dataframe_representable.py
from typing import Union, Optional, Dict, Any
import numpy as np
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


	def update_columns(self, columns: Union[pd.DataFrame, Dict[str, pd.Series]], join_on: Optional[str] = None, fill_value: Any = np.nan, inplace: bool = True) -> Optional[pd.DataFrame]:
		"""Add or update columns on the backing dataframe.

		Usage:
			from neuropy.utils.mixins.dataframe_representable import DataFrameRepresentable, ensure_dataframe

			# Equal length — direct positional assignment:
			a_sess.laps.update_columns(new_cols_df, inplace=True)

			# Unequal length — left-join on a key column, fill missing:
			a_sess.laps.update_columns(partial_df, join_on='lap_id', fill_value=np.nan, inplace=True)

			# Non-mutating — return updated copy:
			updated_df = a_sess.laps.update_columns(new_cols_df, inplace=False)
		"""
		current: pd.DataFrame = self.to_dataframe()
		if isinstance(columns, dict):
			columns = pd.DataFrame(columns)

		new_col_names = [c for c in columns.columns if c != join_on]

		if len(current) == len(columns):
			updated = current.copy()
			for col in new_col_names:
				updated[col] = columns[col].values
		else:
			if join_on is None:
				raise ValueError(f"Row counts differ (self has {len(current)}, columns has {len(columns)}). Provide join_on (and optionally fill_value) to merge by key.")
			if join_on not in current.columns:
				raise ValueError(f"join_on column '{join_on}' not found in current dataframe. Available: {list(current.columns)}")
			if join_on not in columns.columns:
				if join_on == columns.index.name:
					columns = columns.reset_index()
				else:
					raise ValueError(f"join_on column '{join_on}' not found in columns dataframe. Available: {list(columns.columns)} (index.name={columns.index.name!r})")
			overlap_cols = [c for c in new_col_names if c in current.columns]
			if overlap_cols:
				current = current.drop(columns=overlap_cols)
			updated = current.merge(columns[[join_on] + new_col_names], on=join_on, how='left')
			if fill_value is not np.nan:
				updated[new_col_names] = updated[new_col_names].fillna(fill_value)

		if inplace:
			self._df = updated
			return None
		return updated


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
	