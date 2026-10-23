import inspect
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import pandas as pd

tests_folder = Path(os.path.dirname(__file__))

try:
    import neuropy
except ModuleNotFoundError:
    root_project_folder = tests_folder.parent
    sys.path.insert(0, str(root_project_folder))
finally:
    from neuropy.utils.indexing_helpers import PandasHelpers


class TestPandasHelpersSafeMerge(unittest.TestCase):
    """ ``PandasHelpers.safe_merge`` is a drop-in for ``pd.merge`` that preserves ``DataFrame.attrs``. """

    def _left_right(self):
        left = pd.DataFrame({'key': [1, 2, 2], 'x': [10, 20, 21]})
        right = pd.DataFrame({'key': [2, 3], 'y': [200, 300]})
        left.attrs['data_context'] = {'grain': 'per_time_bin'}
        left.attrs['session'] = 'left_sess'
        right.attrs['data_context'] = {'grain': 'per_epoch'}
        right.attrs['file_type'] = 'epoch_df'
        return left, right


    def _assert_equals_pandas_merge(self, left, right, *args, **kwargs):
        expected = pd.merge(left, right, *args, **kwargs)
        actual = PandasHelpers.safe_merge(left, right, *args, **kwargs)
        pd.testing.assert_frame_equal(actual, expected)
        return actual, expected


    def test_default_how_is_inner_like_pandas(self):
        left, right = self._left_right()
        actual, expected = self._assert_equals_pandas_merge(left, right, on='key')
        self.assertListEqual(actual['key'].tolist(), expected['key'].tolist())
        self.assertTrue((actual['key'] == 2).all())


    def test_how_options_match_pandas(self):
        left, right = self._left_right()
        for how in ('left', 'right', 'outer', 'inner'):
            self._assert_equals_pandas_merge(left, right, how=how, on='key')
        ## END for how in ('left', 'right', 'outer', 'inner')...


    def test_cross_merge_matches_pandas(self):
        left = pd.DataFrame({'a': [1, 2]})
        right = pd.DataFrame({'b': ['x', 'y']})
        left.attrs['data_context'] = {'grain': 'per_time_bin'}
        actual, _expected = self._assert_equals_pandas_merge(left, right, how='cross')
        self.assertEqual(len(actual), 4)
        self.assertEqual(actual.attrs['data_context'], {'grain': 'per_time_bin'})


    def test_positional_args_match_pd_merge(self):
        left, right = self._left_right()
        self._assert_equals_pandas_merge(left, right, 'left', 'key')


    def test_on_list_of_columns(self):
        left = pd.DataFrame({'k1': [1, 1], 'k2': ['a', 'b'], 'x': [10, 20]})
        right = pd.DataFrame({'k1': [1, 1], 'k2': ['a', 'c'], 'y': [100, 300]})
        self._assert_equals_pandas_merge(left, right, how='left', on=['k1', 'k2'])


    def test_left_on_right_on(self):
        left = pd.DataFrame({'lkey': [1, 2], 'x': [10, 20]})
        right = pd.DataFrame({'rkey': [2, 3], 'y': [200, 300]})
        self._assert_equals_pandas_merge(left, right, how='inner', left_on='lkey', right_on='rkey')


    def test_left_index_right_index(self):
        left = pd.DataFrame({'x': [10, 20, 30]}, index=pd.Index([1, 2, 3], name='key'))
        right = pd.DataFrame({'y': [200, 400]}, index=pd.Index([2, 4], name='key'))
        self._assert_equals_pandas_merge(left, right, how='outer', left_index=True, right_index=True)


    def test_left_index_right_on(self):
        left = pd.DataFrame({'x': [10, 20]}, index=pd.Index([1, 2], name='idx'))
        right = pd.DataFrame({'key': [2, 3], 'y': [200, 300]})
        self._assert_equals_pandas_merge(left, right, how='left', left_index=True, right_on='key')


    def test_suffixes(self):
        left = pd.DataFrame({'key': [1, 2], 'val': [10, 20]})
        right = pd.DataFrame({'key': [1, 2], 'val': [100, 200]})
        self._assert_equals_pandas_merge(left, right, how='inner', on='key', suffixes=('_L', '_R'))
        actual, _expected = self._assert_equals_pandas_merge(left, right, how='inner', on='key')
        self.assertIn('val_x', actual.columns)
        self.assertIn('val_y', actual.columns)


    def test_sort(self):
        left = pd.DataFrame({'key': [3, 1, 2], 'x': [30, 10, 20]})
        right = pd.DataFrame({'key': [2, 3, 1], 'y': [200, 300, 100]})
        self._assert_equals_pandas_merge(left, right, how='inner', on='key', sort=False)
        self._assert_equals_pandas_merge(left, right, how='inner', on='key', sort=True)


    def test_indicator_bool_and_name(self):
        left, right = self._left_right()
        actual_bool, _expected_bool = self._assert_equals_pandas_merge(left, right, how='outer', on='key', indicator=True)
        self.assertIn('_merge', actual_bool.columns)
        actual_named, _expected_named = self._assert_equals_pandas_merge(left, right, how='outer', on='key', indicator='merge_source')
        self.assertIn('merge_source', actual_named.columns)


    def test_validate_one_to_one_success_and_failure(self):
        left = pd.DataFrame({'key': [1, 2], 'x': [10, 20]})
        right = pd.DataFrame({'key': [1, 2], 'y': [100, 200]})
        self._assert_equals_pandas_merge(left, right, how='inner', on='key', validate='one_to_one')
        bad_right = pd.DataFrame({'key': [1, 1], 'y': [100, 101]})
        with self.assertRaises(pd.errors.MergeError):
            pd.merge(left, bad_right, how='inner', on='key', validate='one_to_one')
        with self.assertRaises(pd.errors.MergeError):
            PandasHelpers.safe_merge(left, bad_right, how='inner', on='key', validate='one_to_one')


    def test_copy_kwarg_forwarded_when_supported(self):
        left, right = self._left_right()
        pd_merge_params = inspect.signature(pd.merge).parameters
        if 'copy' not in pd_merge_params:
            self.skipTest('this pandas version does not accept merge(copy=...)')
        self._assert_equals_pandas_merge(left, right, how='left', on='key', copy=True)
        self._assert_equals_pandas_merge(left, right, how='left', on='key', copy=False)


    def test_unknown_kwarg_raises_like_pandas(self):
        left, right = self._left_right()
        with self.assertRaises(TypeError):
            pd.merge(left, right, how='left', on='key', not_a_real_merge_kwarg=True)
        with self.assertRaises(TypeError):
            PandasHelpers.safe_merge(left, right, how='left', on='key', not_a_real_merge_kwarg=True)


    def test_invalid_how_raises_like_pandas(self):
        left, right = self._left_right()
        pandas_exc_type = None
        try:
            pd.merge(left, right, how='diagonal', on='key')
        except Exception as e:
            pandas_exc_type = type(e)
        else:
            self.fail('pd.merge should reject how="diagonal"')
        with self.assertRaises(pandas_exc_type):
            PandasHelpers.safe_merge(left, right, how='diagonal', on='key')


    def test_named_series_right_operand(self):
        left = pd.DataFrame({'x': [10, 20, 30]}, index=pd.Index([0, 1, 2], name='idx'))
        right = pd.Series([100, 200], name='y', index=pd.Index([1, 3], name='idx'))
        self._assert_equals_pandas_merge(left, right, how='outer', left_index=True, right_index=True)


    def test_empty_frames(self):
        left = pd.DataFrame({'key': pd.Series(dtype=int), 'x': pd.Series(dtype=float)})
        right = pd.DataFrame({'key': pd.Series(dtype=int), 'y': pd.Series(dtype=float)})
        left.attrs['data_context'] = {'grain': 'per_time_bin'}
        actual, _expected = self._assert_equals_pandas_merge(left, right, how='left', on='key')
        self.assertEqual(len(actual), 0)
        self.assertEqual(actual.attrs['data_context'], {'grain': 'per_time_bin'})


    def test_preserves_left_attrs_and_fills_right_only_keys(self):
        left, right = self._left_right()
        merged = PandasHelpers.safe_merge(left, right, how='left', on='key')
        self.assertEqual(merged.attrs['data_context'], {'grain': 'per_time_bin'})
        self.assertEqual(merged.attrs['session'], 'left_sess')
        self.assertEqual(merged.attrs['file_type'], 'epoch_df')


    def test_does_not_mutate_source_attrs(self):
        left, right = self._left_right()
        left_attrs_before = deepcopy(dict(left.attrs))
        right_attrs_before = deepcopy(dict(right.attrs))
        merged = PandasHelpers.safe_merge(left, right, how='left', on='key')
        merged.attrs['data_context'] = {'grain': 'mutated'}
        self.assertEqual(dict(left.attrs), left_attrs_before)
        self.assertEqual(dict(right.attrs), right_attrs_before)


    def test_plain_pandas_merge_drops_attrs(self):
        left, right = self._left_right()
        merged = pd.merge(left, right, how='left', on='key')
        self.assertNotIn('data_context', merged.attrs)


    def test_accessor_matches_dataframe_merge_and_preserves_attrs(self):
        left, right = self._left_right()
        expected = left.merge(right, how='left', on='key')
        actual = left.neuropy.safe_merge(right, how='left', on='key')
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual.attrs['data_context'], {'grain': 'per_time_bin'})
        self.assertEqual(actual.attrs['file_type'], 'epoch_df')


    def test_accessor_positional_how_on(self):
        left, right = self._left_right()
        expected = left.merge(right, 'inner', 'key')
        actual = left.neuropy.safe_merge(right, 'inner', 'key')
        pd.testing.assert_frame_equal(actual, expected)


    def test_multiindex_columns_on(self):
        left = pd.DataFrame({('k', 'a'): [1, 2], ('v', 'x'): [10, 20]})
        right = pd.DataFrame({('k', 'a'): [2, 3], ('v', 'y'): [200, 300]})
        self._assert_equals_pandas_merge(left, right, how='inner', on=[('k', 'a')])


if __name__ == '__main__':
    unittest.main()
