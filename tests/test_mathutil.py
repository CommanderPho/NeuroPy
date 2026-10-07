import unittest
import numpy as np

import sys, os
from pathlib import Path

# Add Neuropy to the path as needed
tests_folder = Path(os.path.dirname(__file__))

try:
    import neuropy
except ModuleNotFoundError as e:
    root_project_folder = tests_folder.parent
    neuropy_folder = root_project_folder.joinpath('neuropy')
    sys.path.insert(0, str(root_project_folder))
finally:
    from neuropy.utils.mathutil import map_value


class TestMathUtil(unittest.TestCase):

    def test_map_value_standard(self):
        # standard midpoint
        self.assertEqual(map_value(5, (0, 10), (0, 100)), 50.0)
        # standard lower bound
        self.assertEqual(map_value(0, (0, 10), (0, 100)), 0.0)
        # standard upper bound
        self.assertEqual(map_value(10, (0, 10), (0, 100)), 100.0)

    def test_map_value_negative_range(self):
        # source negative range to positive target
        self.assertEqual(map_value(0, (-10, 10), (0, 100)), 50.0)
        # source positive range to negative target
        self.assertEqual(map_value(5, (0, 10), (-50, 50)), 0.0)
        # negative to negative
        self.assertEqual(map_value(-5, (-10, 0), (-100, 0)), -50.0)

    def test_map_value_reverse_range(self):
        # source range reversed
        self.assertEqual(map_value(2, (10, 0), (0, 100)), 80.0)
        # target range reversed
        self.assertEqual(map_value(5, (0, 10), (100, 0)), 50.0)
        # both reversed
        self.assertEqual(map_value(5, (10, 0), (100, 0)), 50.0)
        self.assertEqual(map_value(2, (10, 0), (100, 0)), 20.0)

    def test_map_value_out_of_bounds(self):
        # extrapolation below lower bound
        self.assertEqual(map_value(-5, (0, 10), (0, 100)), -50.0)
        # extrapolation above upper bound
        self.assertEqual(map_value(15, (0, 10), (0, 100)), 150.0)

    def test_map_value_numpy_array(self):
        values = np.array([0, 5, 10])
        expected = np.array([0.0, 50.0, 100.0])
        mapped = map_value(values, (0, 10), (0, 100))
        np.testing.assert_array_equal(mapped, expected)

    def test_map_value_zero_division(self):
        # when from_low == from_high, it should raise ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            map_value(5, (10, 10), (0, 100))

if __name__ == '__main__':
    unittest.main()
