# concatenatable.py

from typing import Iterable
from typing import Iterable

class ConcatenationInitializable:
	""" Implementors can be meaningfully concatenated together to produce a combined object of the same type
	"""
	@classmethod
	def concat(cls, objList: Iterable):
		"""Designed after Pandas `new_df = pd.concat([df1, df2, ...])` function.
		Args:
			objList (Iterable): a list of the objects to be concatenated
		Returns:
			new_obj: an object of the same type as the input objects, representing their concatenation
		"""
		raise NotImplementedError


