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

from neuropy.utils.probability_downsampling import RigorousPDFDownsampler  # type: ignore[import]


class TestRigorousPDFDownsamplerDiscretePMF(unittest.TestCase):
    """Tests for RigorousPDFDownsampler in discrete PMF mode."""

    def test_init_global_pmf(self):
        """Initialization with a simple 2D PMF should preserve shape metadata."""
        pmf = np.ones((10, 12), dtype=float)
        pmf /= pmf.sum()

        ds = RigorousPDFDownsampler(pmf, spatial_axes=(0, 1))
        self.assertEqual(ds.fine_pdf.shape, pmf.shape)
        self.assertEqual(ds.ndim, 2)
        self.assertEqual(ds.shape_f, pmf.shape)

    def test_per_time_pmf_normalization_preserved(self):
        """
        For a 4D tensor (n_x, n_y, n_cond, n_t) representing per-time PMFs over
        spatial axes (0, 1, 2), each time slice should remain normalized.
        """
        n_x, n_y, n_cond, n_t = 16, 18, 2, 7
        fine = np.random.rand(n_x, n_y, n_cond, n_t)
        mass = fine.sum(axis=(0, 1, 2), keepdims=True)
        fine /= mass

        ds = RigorousPDFDownsampler(fine, spatial_axes=(0, 1, 2))
        factors = (3.0, 4.0)
        axes = (0, 1)
        coarse, _, _ = ds.downsample(factors=factors, axes=axes, method="fast")

        expected_nx = int(np.ceil(n_x / factors[0]))
        expected_ny = int(np.ceil(n_y / factors[1]))
        self.assertEqual(coarse.shape, (expected_nx, expected_ny, n_cond, n_t))

        coarse_mass = coarse.sum(axis=(0, 1, 2))
        self.assertTrue(np.allclose(coarse_mass, 1.0, rtol=1e-6))

    def test_integer_factor_required(self):
        """Non-integer factors should raise in discrete PMF mode."""
        pmf = np.ones((20, 20), dtype=float)
        pmf /= pmf.sum()

        ds = RigorousPDFDownsampler(pmf, spatial_axes=(0, 1))
        with self.assertRaises(ValueError):
            ds.downsample(factors=(2.5, 2.0), axes=(0, 1))

    def test_block_sums_match_direct_grouping(self):
        """Coarse bins should equal sums of corresponding fine bins."""
        n_x, n_y = 12, 10
        fine = np.random.rand(n_x, n_y)
        fine /= fine.sum()

        ds = RigorousPDFDownsampler(fine, spatial_axes=(0, 1))
        rx, ry = 3.0, 2.0
        coarse, _, _ = ds.downsample(factors=(rx, ry), axes=(0, 1))

        expected_nx = int(np.ceil(n_x / rx))
        expected_ny = int(np.ceil(n_y / ry))
        self.assertEqual(coarse.shape, (expected_nx, expected_ny))

        rx_i, ry_i = int(rx), int(ry)
        for ix in range(expected_nx):
            for iy in range(expected_ny):
                x_start = ix * rx_i
                x_end = min((ix + 1) * rx_i, n_x)
                y_start = iy * ry_i
                y_end = min((iy + 1) * ry_i, n_y)
                block = fine[x_start:x_end, y_start:y_end]
                expected_mass = block.sum()
                self.assertAlmostEqual(coarse[ix, iy], expected_mass, places=10)

        self.assertAlmostEqual(coarse.sum(), 1.0, places=10)

    def test_default_axes_match_explicit(self):
        """Omitting axes should default to the first len(factors) axes."""
        pmf = np.random.rand(15, 21)
        pmf /= pmf.sum()

        ds = RigorousPDFDownsampler(pmf, spatial_axes=(0, 1))
        factors = (2.0, 3.0)

        coarse_auto, binsizes_auto, bins_auto = ds.downsample(factors=factors, axes=None)
        coarse_explicit, binsizes_explicit, bins_explicit = ds.downsample(factors=factors, axes=(0, 1))

        self.assertEqual(coarse_auto.shape, coarse_explicit.shape)
        self.assertTrue(np.allclose(coarse_auto, coarse_explicit))
        self.assertTrue(np.allclose(binsizes_auto, binsizes_explicit))
        for b_a, b_e in zip(bins_auto, bins_explicit):
            self.assertTrue(np.allclose(b_a, b_e))

    def test_bins_metadata_shape_and_lengths(self):
        """If bins are provided, returned coarse bins should match coarse shapes."""
        n_x, n_y = 14, 16
        x_bins = np.linspace(0.0, 10.0, n_x)
        y_bins = np.linspace(-5.0, 5.0, n_y)

        pmf = np.random.rand(n_x, n_y)
        pmf /= pmf.sum()

        ds = RigorousPDFDownsampler(pmf, bins=(x_bins, y_bins), spatial_axes=(0, 1))
        factors = (3.0, 4.0)
        coarse, coarse_bin_sizes, coarse_bins = ds.downsample(factors=factors, axes=(0, 1))

        expected_nx = int(np.ceil(n_x / factors[0]))
        expected_ny = int(np.ceil(n_y / factors[1]))
        self.assertEqual(coarse.shape, (expected_nx, expected_ny))
        self.assertEqual(coarse_bins[0].shape[0], expected_nx)
        self.assertEqual(coarse_bins[1].shape[0], expected_ny)
        self.assertEqual(len(coarse_bin_sizes), 2)


if __name__ == "__main__":
    unittest.main()


