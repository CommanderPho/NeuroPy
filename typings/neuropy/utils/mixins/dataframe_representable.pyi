"""
This type stub file was generated by pyright.
"""

"""
This type stub file was generated by pyright.
"""
class DataFrameInitializable:
	""" Implementors can be initialized from a Pandas DataFrame. 
	"""
	@classmethod
	def from_dataframe(cls, df):
		...
	


class DataFrameRepresentable(DataFrameInitializable):
	def to_dataframe(self):
		...
	


