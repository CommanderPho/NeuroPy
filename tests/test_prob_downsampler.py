import unittest
from unittest.mock import MagicMock
import dill
import copy
from neuropy.utils.result_context import IdentifyingContext, CollisionOutcome

import unittest
import sys
import os
from pathlib import Path
import numpy as np

# Add project to path
tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
sys.path.insert(0, str(src_folder))

from neuropy.utils.probability_downsampling import RigorousPDFDownsampler


class TestRigorousPDFDownsampler(unittest.TestCase):
    """Test suite for RigorousPDFDownsampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple normalized 2D Gaussian PDF for testing
        Nx_f, Ny_f = 100, 100
        x_f = np.linspace(0, 10, Nx_f)
        y_f = np.linspace(0, 10, Ny_f)
        X_f, Y_f = np.meshgrid(x_f, y_f)
        mu_x, mu_y, sigma = 5.0, 5.0, 1.0
        self.fine_pdf = np.exp(-((X_f - mu_x)**2 + (Y_f - mu_y)**2) / (2 * sigma**2))
        self.dx_f = x_f[1] - x_f[0]
        self.dy_f = y_f[1] - y_f[0]
        # Normalize
        self.fine_pdf /= np.sum(self.fine_pdf) * self.dx_f * self.dy_f
        
        # Create a uniform PDF for simpler tests
        self.uniform_pdf = np.ones((50, 50))
        self.uniform_pdf /= np.sum(self.uniform_pdf) * self.dx_f * self.dy_f
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    # ========== Initialization Tests ==========
    
    def test_init_valid_2d_array(self):
        """Test initialization with valid 2D array."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        self.assertEqual(downsampler.fine_pdf.shape, self.fine_pdf.shape)
        self.assertEqual(downsampler.Nx_f, self.fine_pdf.shape[1])
        self.assertEqual(downsampler.Ny_f, self.fine_pdf.shape[0])
        self.assertEqual(downsampler.dx_f, self.dx_f)
        self.assertEqual(downsampler.dy_f, self.dy_f)
    
    def test_init_invalid_1d_array(self):
        """Test initialization fails with 1D array."""
        with self.assertRaises(ValueError):
            RigorousPDFDownsampler(np.array([1, 2, 3]), 1.0, 1.0)
    
    def test_init_invalid_3d_array(self):
        """Test initialization fails with 3D array."""
        with self.assertRaises(ValueError):
            RigorousPDFDownsampler(np.ones((10, 10, 10)), 1.0, 1.0)
    
    def test_init_warning_unnormalized(self):
        """Test that unnormalized PDF triggers warning."""
        unnormalized = self.fine_pdf * 2.0  # Double the mass
        # Should not raise, but should print warning
        downsampler = RigorousPDFDownsampler(unnormalized, self.dx_f, self.dy_f)
        self.assertIsNotNone(downsampler)
    
    # ========== Mass Conservation Tests ==========
    
    def test_mass_conservation_integer_factor(self):
        """Test mass conservation with integer downsampling factor."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.0, ry=4.0)
        
        input_mass = np.sum(self.fine_pdf) * self.dx_f * self.dy_f
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        
        self.assertAlmostEqual(input_mass, 1.0, places=6, msg="Input should be normalized")
        self.assertAlmostEqual(output_mass, 1.0, places=5, msg="Output should preserve mass")
        self.assertAlmostEqual(input_mass, output_mass, places=5, msg="Mass should be conserved")
    
    def test_mass_conservation_non_integer_factor(self):
        """Test mass conservation with non-integer downsampling factor."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.2, ry=3.8)
        
        input_mass = np.sum(self.fine_pdf) * self.dx_f * self.dy_f
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        
        self.assertAlmostEqual(output_mass, 1.0, places=4, msg="Output should preserve mass")
        self.assertAlmostEqual(input_mass, output_mass, places=4, msg="Mass should be conserved")
    
    def test_mass_conservation_asymmetric(self):
        """Test mass conservation with asymmetric downsampling (rx != ry)."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=3.0, ry=5.0)
        
        input_mass = np.sum(self.fine_pdf) * self.dx_f * self.dy_f
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        
        self.assertAlmostEqual(output_mass, 1.0, places=4, msg="Output should preserve mass")
        self.assertAlmostEqual(input_mass, output_mass, places=4, msg="Mass should be conserved")
    
    def test_mass_conservation_uniform_pdf(self):
        """Test mass conservation with uniform PDF."""
        downsampler = RigorousPDFDownsampler(self.uniform_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=2.5, ry=2.5)
        
        input_mass = np.sum(self.uniform_pdf) * self.dx_f * self.dy_f
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        
        self.assertAlmostEqual(output_mass, 1.0, places=5, msg="Output should preserve mass")
        self.assertAlmostEqual(input_mass, output_mass, places=5, msg="Mass should be conserved")
    
    # ========== Downsampling Factor Tests ==========
    
    def test_downsample_single_factor(self):
        """Test downsampling with single factor (ry=None defaults to rx)."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.0)
        
        # Should use same factor for both dimensions
        expected_Nx_c = int(np.ceil(self.fine_pdf.shape[1] / 4.0))
        expected_Ny_c = int(np.ceil(self.fine_pdf.shape[0] / 4.0))
        
        self.assertEqual(coarse_pdf.shape[0], expected_Ny_c)
        self.assertEqual(coarse_pdf.shape[1], expected_Nx_c)
        self.assertAlmostEqual(dx_c, dy_c, places=6, msg="dx_c and dy_c should be equal when rx=ry")
    
    def test_downsample_different_factors(self):
        """Test downsampling with different rx and ry."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=3.0, ry=5.0)
        
        expected_Nx_c = int(np.ceil(self.fine_pdf.shape[1] / 3.0))
        expected_Ny_c = int(np.ceil(self.fine_pdf.shape[0] / 5.0))
        
        self.assertEqual(coarse_pdf.shape[0], expected_Ny_c)
        self.assertEqual(coarse_pdf.shape[1], expected_Nx_c)
        self.assertNotAlmostEqual(dx_c, dy_c, places=1, msg="dx_c and dy_c should differ when rx != ry")
    
    def test_downsample_large_factor(self):
        """Test downsampling with large factor."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=10.0, ry=10.0)
        
        # Should still preserve mass
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        self.assertAlmostEqual(output_mass, 1.0, places=4)
        
        # Should have smaller shape
        self.assertLess(coarse_pdf.shape[0], self.fine_pdf.shape[0])
        self.assertLess(coarse_pdf.shape[1], self.fine_pdf.shape[1])
    
    def test_downsample_small_factor(self):
        """Test downsampling with small factor (close to 1.0)."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=1.1, ry=1.1)
        
        # Should still preserve mass
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        self.assertAlmostEqual(output_mass, 1.0, places=4)
        
        # Should have slightly smaller shape
        self.assertLessEqual(coarse_pdf.shape[0], self.fine_pdf.shape[0])
        self.assertLessEqual(coarse_pdf.shape[1], self.fine_pdf.shape[1])
    
    # ========== Edge Cases ==========
    
    def test_downsample_small_array(self):
        """Test downsampling with small input array."""
        small_pdf = np.ones((10, 10))
        small_pdf /= np.sum(small_pdf) * self.dx_f * self.dy_f
        
        downsampler = RigorousPDFDownsampler(small_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=2.0, ry=2.0)
        
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        self.assertAlmostEqual(output_mass, 1.0, places=4)
    
    def test_downsample_rectangular_array(self):
        """Test downsampling with rectangular (non-square) array."""
        rectangular_pdf = np.ones((50, 100))
        rectangular_pdf /= np.sum(rectangular_pdf) * self.dx_f * self.dy_f
        
        downsampler = RigorousPDFDownsampler(rectangular_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=3.0, ry=2.0)
        
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        self.assertAlmostEqual(output_mass, 1.0, places=4)
        
        # Check shape is correct
        expected_Nx_c = int(np.ceil(100 / 3.0))
        expected_Ny_c = int(np.ceil(50 / 2.0))
        self.assertEqual(coarse_pdf.shape, (expected_Ny_c, expected_Nx_c))
    
    def test_downsample_delta_function(self):
        """Test downsampling with delta-like function (single peak)."""
        delta_pdf = np.zeros((100, 100))
        delta_pdf[50, 50] = 1.0
        delta_pdf /= np.sum(delta_pdf) * self.dx_f * self.dy_f
        
        downsampler = RigorousPDFDownsampler(delta_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.0, ry=4.0)
        
        output_mass = np.sum(coarse_pdf) * dx_c * dy_c
        self.assertAlmostEqual(output_mass, 1.0, places=4)
        
        # Peak should be preserved (though spread out)
        self.assertGreater(np.max(coarse_pdf), 0.0)
    
    # ========== Density Preservation Tests ==========
    
    def test_relative_density_preservation(self):
        """Test that relative densities are preserved (ratios of integrated masses)."""
        # Create PDF with two distinct regions
        test_pdf = np.zeros((100, 100))
        test_pdf[20:40, 20:40] = 2.0  # High density region
        test_pdf[60:80, 60:80] = 1.0  # Low density region
        test_pdf /= np.sum(test_pdf) * self.dx_f * self.dy_f
        
        downsampler = RigorousPDFDownsampler(test_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=5.0, ry=5.0)
        
        # Calculate mass in each region (approximate, since boundaries shift)
        # This is a qualitative test - the high density region should remain higher
        coarse_max = np.max(coarse_pdf)
        coarse_min = np.min(coarse_pdf[coarse_pdf > 0])
        
        # High density region should still be higher than low density region
        self.assertGreater(coarse_max, coarse_min * 1.5, 
                          msg="Relative density ratios should be roughly preserved")
    
    # ========== Output Properties Tests ==========
    
    def test_output_non_negative(self):
        """Test that output PDF is non-negative."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.0, ry=4.0)
        
        self.assertTrue(np.all(coarse_pdf >= 0), msg="All output values should be non-negative")


    def test_output_remains_normalized_fast(self):
        """Test that output PDF is non-negative."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.0, ry=4.0, method='fast')
        normalized_sum = np.sum(coarse_pdf) * dx_c * dy_c

        self.assertAlmostEqual(normalized_sum, 1.0, places=4, msg=f"Output should remain normalized but sums to {normalized_sum}")


    def test_output_remains_normalized_slow(self):
        """Test that output PDF is non-negative."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.0, ry=4.0, method='slow')
        normalized_sum = np.sum(coarse_pdf) * dx_c * dy_c

        self.assertAlmostEqual(normalized_sum, 1.0, places=4, msg=f"Output should remain normalized but sums to {normalized_sum}")


    def test_output_pdf_reducible_to_pmf(self):
        """Test that output PDF is non-negative."""
        dx_f = 1.0
        dy_f = 1.0

        downsampler = RigorousPDFDownsampler(self.fine_pdf, dx_f, dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.0, ry=4.0, method='fast')
        print(f'dx_c: {dx_c}, dy_c: {dx_c}')
        # normalized_sum = np.sum(coarse_pdf) * dx_c * dy_c
        normalized_sum = np.sum(coarse_pdf)

        self.assertAlmostEqual(normalized_sum, 1.0, places=4, msg=f"Output should remain normalized but sums to {normalized_sum}")


    
    def test_output_shape_consistency(self):
        """Test that output shape is consistent with input and downsampling factor."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        rx, ry = 4.2, 3.8
        
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=rx, ry=ry)
        
        expected_Nx_c = int(np.ceil(self.fine_pdf.shape[1] / rx))
        expected_Ny_c = int(np.ceil(self.fine_pdf.shape[0] / ry))
        
        self.assertEqual(coarse_pdf.shape, (expected_Ny_c, expected_Nx_c))
    
    def test_output_spacing_consistency(self):
        """Test that output spacings are consistent with grid dimensions."""
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.0, ry=4.0)
        
        # Total domain size should be preserved
        total_x_fine = self.fine_pdf.shape[1] * self.dx_f
        total_y_fine = self.fine_pdf.shape[0] * self.dy_f
        total_x_coarse = coarse_pdf.shape[1] * dx_c
        total_y_coarse = coarse_pdf.shape[0] * dy_c
        
        self.assertAlmostEqual(total_x_fine, total_x_coarse, places=6)
        self.assertAlmostEqual(total_y_fine, total_y_coarse, places=6)
    
    # ========== Method Parameter Tests ==========
    
    def test_method_parameter_ignored(self):
        """Test that method parameter exists (even if not fully implemented)."""
        # The method parameter is in the signature but may not be fully used
        downsampler = RigorousPDFDownsampler(self.fine_pdf, self.dx_f, self.dy_f)
        
        # Should work with default method
        coarse_pdf1, _, _ = downsampler.downsample(rx=4.0, method='trapezoidal')
        
        # Should work with 'sum' method (if implemented)
        # Note: This test may need adjustment based on actual implementation
        coarse_pdf2, _, _ = downsampler.downsample(rx=4.0, method='sum')
        
        # Both should preserve mass
        mass1 = np.sum(coarse_pdf1) * (self.fine_pdf.shape[1] * self.dx_f / coarse_pdf1.shape[1]) * \
                (self.fine_pdf.shape[0] * self.dy_f / coarse_pdf1.shape[0])
        mass2 = np.sum(coarse_pdf2) * (self.fine_pdf.shape[1] * self.dx_f / coarse_pdf2.shape[1]) * \
                (self.fine_pdf.shape[0] * self.dy_f / coarse_pdf2.shape[0])
        
        self.assertAlmostEqual(mass1, 1.0, places=4)
        self.assertAlmostEqual(mass2, 1.0, places=4)


if __name__ == '__main__':
    unittest.main()
