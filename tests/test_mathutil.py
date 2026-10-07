import unittest
import numpy as np
from neuropy.utils.mathutil import map_to_fixed_range

class TestMathUtil(unittest.TestCase):

    def test_map_to_fixed_range_default(self):
        lin_pos = np.array([1.0, 2.0, 3.0])
        expected = np.array([0.0, 0.5, 1.0])
        result = map_to_fixed_range(lin_pos)
        np.testing.assert_array_almost_equal(result, expected)

    def test_map_to_fixed_range_custom_range(self):
        lin_pos = np.array([1.0, 2.0, 3.0])
        expected = np.array([10.0, 15.0, 20.0])
        result = map_to_fixed_range(lin_pos, x_min=10.0, x_max=20.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_map_to_fixed_range_with_nan(self):
        lin_pos = np.array([1.0, 2.0, np.nan, 3.0])
        expected = np.array([0.0, 0.5, np.nan, 1.0])
        result = map_to_fixed_range(lin_pos)
        np.testing.assert_array_almost_equal(result, expected)

    def test_map_to_fixed_range_identical_values(self):
        lin_pos = np.array([2.0, 2.0, 2.0])
        # max - min = 0, so division by zero leads to nan
        with np.errstate(invalid='ignore'):
            result = map_to_fixed_range(lin_pos)
        self.assertTrue(np.all(np.isnan(result)))

    def test_map_to_fixed_range_scalar(self):
        # A single scalar array
        lin_pos = np.array([5.0])
        with np.errstate(invalid='ignore'):
            result = map_to_fixed_range(lin_pos)
        self.assertTrue(np.all(np.isnan(result)))

if __name__ == '__main__':
    unittest.main()
