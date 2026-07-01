import unittest
import numpy as np
from neuropy.utils.efficient_interval_search import verify_non_overlapping, _compiled_verify_non_overlapping

class TestEfficientIntervalSearch(unittest.TestCase):

    def test_verify_non_overlapping_empty(self):
        """Test with empty array"""
        intervals = np.empty((0, 2))
        self.assertTrue(verify_non_overlapping(intervals))

    def test_verify_non_overlapping_single(self):
        """Test with single interval"""
        intervals = np.array([[1.0, 2.0]])
        self.assertTrue(verify_non_overlapping(intervals))

    def test_verify_non_overlapping_non_overlapping(self):
        """Test with distinct non-overlapping intervals"""
        intervals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertTrue(verify_non_overlapping(intervals))

    def test_verify_non_overlapping_overlapping(self):
        """Test with overlapping intervals"""
        intervals = np.array([[1.0, 3.0], [2.0, 4.0]])
        self.assertFalse(verify_non_overlapping(intervals))

    def test_verify_non_overlapping_adjacent(self):
        """Test with adjacent intervals (start == previous stop)"""
        intervals = np.array([[1.0, 2.0], [2.0, 3.0]])
        # verify_non_overlapping checks strictly greater than, so adjacent intervals are considered overlapping
        self.assertFalse(verify_non_overlapping(intervals))

    def test_verify_non_overlapping_contained(self):
        """Test with contained intervals"""
        intervals = np.array([[1.0, 4.0], [2.0, 3.0]])
        self.assertFalse(verify_non_overlapping(intervals))

    def test_verify_non_overlapping_large_array(self):
        """Test with a large array of non-overlapping intervals"""
        num_intervals = 1000
        starts = np.arange(0, num_intervals * 2, 2)
        stops = starts + 1
        intervals = np.column_stack((starts, stops))
        self.assertTrue(verify_non_overlapping(intervals))

    def test_verify_non_overlapping_large_array_with_overlap(self):
        """Test with a large array where one interval overlaps"""
        num_intervals = 1000
        starts = np.arange(0, num_intervals * 2, 2)
        stops = starts + 1
        intervals = np.column_stack((starts, stops))
        # Introduce an overlap
        intervals[500] = [starts[499] + 0.5, stops[499] + 0.5]
        self.assertFalse(verify_non_overlapping(intervals))

if __name__ == '__main__':
    unittest.main()
