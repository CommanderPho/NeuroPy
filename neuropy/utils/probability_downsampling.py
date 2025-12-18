import numpy as np
from typing import Tuple, Optional, Sequence, Union
from attrs import define, field
from neuropy.utils.mixins.AttrsClassHelpers import SimpleFieldSizesReprMixin


ArrayLike = Union[np.ndarray, Sequence[float]]


# TODO 2025-12-18 - TODO 🚧❗ `RigorousPDFDownsampler` is broken, it doesn't produce correctly normalized outputs and is excessively slow
# @metadata_attributes(short_name=None, tags=['UNFINISHED', 'UNEVALUATED', 'AI'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-17 09:17', related_items=[])
@define(slots=False, eq=False, repr=False)
class RigorousPDFDownsampler(SimpleFieldSizesReprMixin):
    """
    High-performance N-D **discrete PMF** downsampler using conservative coarse-graining.

    This class is designed for large arrays that represent **probability mass functions**
    (PMFs), not continuous densities. Typical use case: a decoding posterior with shape
    (n_x, n_y, n_cond, n_t), where one or more axes are *spatial* and the remaining
    axes are *non-spatial* indices (e.g., time, condition, trial).

    Semantics:
    - `fine_pdf` contains **probability masses**, not densities.
    - A set of `spatial_axes` defines which axes are integrated over when checking and
      enforcing normalization.
    - For each fixed index of the non-spatial axes, the sum over all spatial axes
      should be 1.0 (up to numerical tolerance) both before and after downsampling.

    Geometry:
    - `bins` and `bin_sizes` are treated as **optional metadata**.
      They may be used to derive and return coarse bin centres and coarse bin sizes,
      but they do *not* affect probability mass calculations or normalization.

    Downsampling:
    - Only the user-specified axes are downsampled; all other axes are treated as
      independent and left unchanged.
    - Downsampling factors are assumed to be effectively integer (within tolerance)
      and implemented as conservative **block sums** over fine indices.
      Any leftover tail along a spatial axis is grouped into a final coarse bin.
    """

    fine_pdf: np.ndarray = field()
    bin_sizes: Optional[ArrayLike] = field(default=None)
    bins: Optional[Sequence[Optional[ArrayLike]]] = field(default=None)
    spatial_axes: Optional[Sequence[int]] = field(default=None)

    ndim: int = field(init=False)
    shape_f: Tuple[int, ...] = field(init=False)
    _bin_sizes_arr: np.ndarray = field(init=False, repr=False)
    _fine_bins: Tuple[np.ndarray, ...] = field(init=False, repr=False)
    _spatial_axes: Tuple[int, ...] = field(init=False, repr=False)

    def __attrs_post_init__(self):
        """Validates input and computes derived attributes."""
        if self.fine_pdf.ndim < 1:
            raise ValueError("fine_pdf must be at least 1D array")

        self.shape_f = self.fine_pdf.shape
        self.ndim = self.fine_pdf.ndim

        # Infer bin sizes and canonical fine bins from either bin_sizes or bins.
        if self.bin_sizes is None and self.bins is None:
            # raise ValueError("Either bin_sizes or bins must be provided.")
            print(f'WARNING: no bin_sizes or bins were provided, so using constant spacings')
            self.bin_sizes = [1.0 for v in self.shape_f] # all bins are size == 1.0
            # self.bin_sizes = [(1.0/float(v)) for v in self.shape_f] # all bins are size == 1.0/size(dim)
            print(f'\tself.bin_sizes: {self.bin_sizes}')

        fine_bins_list = []

        if (self.bins is not None) and (self.bin_sizes is None):
            # Derive bin_sizes from provided bins (with None -> index-based bins).
            if len(self.bins) != self.ndim:
                raise ValueError(f"bins must have length {self.ndim}, got {len(self.bins)}.")

            bin_sizes_per_axis = []
            for d in range(self.ndim):
                b = self.bins[d]
                if b is None:
                    axis_bins = np.arange(self.shape_f[d], dtype=float)
                else:
                    axis_bins = np.asarray(b, dtype=float)
                    if axis_bins.ndim != 1:
                        raise ValueError(f"bins[{d}] must be 1D, got shape {axis_bins.shape}.")
                    if axis_bins.shape[0] != self.shape_f[d]:
                        raise ValueError(f"bins[{d}] length {axis_bins.shape[0]} does not match "
                                         f"fine_pdf.shape[{d}] = {self.shape_f[d]}.")

                if axis_bins.size > 1:
                    diffs = np.diff(axis_bins)
                    spacing_d = float(np.median(diffs))
                    if not np.isfinite(spacing_d) or spacing_d <= 0:
                        raise ValueError(f"Derived spacing for axis {d} must be positive and finite, got {spacing_d}.")
                else:
                    spacing_d = 1.0

                fine_bins_list.append(axis_bins)
                bin_sizes_per_axis.append(spacing_d)

            bin_sizes_arr = np.asarray(bin_sizes_per_axis, dtype=float)
            self._bin_sizes_arr = bin_sizes_arr
            # Store the normalized versions back for introspection
            self.bin_sizes = bin_sizes_arr
            self.bins = tuple(fine_bins_list)

        else:
            # bin_sizes provided (with or without bins). Validate and optionally
            # check consistency with bins if they are also provided.
            bin_sizes_arr = np.asarray(self.bin_sizes, dtype=float)
            if bin_sizes_arr.ndim != 1 or bin_sizes_arr.shape[0] != self.ndim:
                raise ValueError(f"bin_sizes must be 1D with length {self.ndim}, got shape {bin_sizes_arr.shape}")
            if np.any(bin_sizes_arr <= 0):
                raise ValueError("All bin_sizes must be positive.")
            self._bin_sizes_arr = bin_sizes_arr

            if self.bins is not None:
                if len(self.bins) != self.ndim:
                    raise ValueError(f"bins must have length {self.ndim}, got {len(self.bins)}.")
                for d in range(self.ndim):
                    b = self.bins[d]
                    if b is None:
                        axis_bins = np.arange(self.shape_f[d], dtype=float)
                    else:
                        axis_bins = np.asarray(b, dtype=float)
                        if axis_bins.ndim != 1:
                            raise ValueError(f"bins[{d}] must be 1D, got shape {axis_bins.shape}.")
                        if axis_bins.shape[0] != self.shape_f[d]:
                            raise ValueError(f"bins[{d}] length {axis_bins.shape[0]} does not match "
                                             f"fine_pdf.shape[{d}] = {self.shape_f[d]}.")
                    fine_bins_list.append(axis_bins)

                    if axis_bins.size > 1:
                        diffs = np.diff(axis_bins)
                        spacing_d = float(np.median(diffs))
                        if np.isfinite(spacing_d) and spacing_d > 0:
                            if not np.isclose(spacing_d, self._bin_sizes_arr[d], rtol=1e-3, atol=1e-6):
                                raise ValueError(f"Inconsistent spacing between bin_sizes[{d}]={self._bin_sizes_arr[d]} "
                                                 f"and bins[{d}] derived spacing {spacing_d}.")
                self.bins = tuple(fine_bins_list)
            else:
                # No bins provided; construct a default index-based grid consistent with bin_sizes.
                for d in range(self.ndim):
                    spacing_d = float(self._bin_sizes_arr[d])
                    axis_bins = np.arange(self.shape_f[d], dtype=float) * spacing_d
                    fine_bins_list.append(axis_bins)
                self.bins = tuple(fine_bins_list)

        self._fine_bins = tuple(fine_bins_list)

        # Normalize and store spatial axes (dimensions to integrate over for mass checks/renorm)
        if self.spatial_axes is None:
            spatial_axes_norm = list(range(self.ndim))
        else:
            spatial_axes_norm = []
            for ax in self.spatial_axes:
                ax_i = int(ax)
                if ax_i < 0:
                    ax_i += self.ndim
                if ax_i < 0 or ax_i >= self.ndim:
                    raise ValueError(f"spatial_axes entry {ax} is out of bounds for ndim={self.ndim}.")
                spatial_axes_norm.append(ax_i)
            if len(set(spatial_axes_norm)) != len(spatial_axes_norm):
                raise ValueError(f"spatial_axes must be unique, got {self.spatial_axes}.")
        # Store as a sorted tuple for stable indexing
        self._spatial_axes = tuple(sorted(spatial_axes_norm))

        # Verify input is roughly normalized.
        # If spatial_axes is None, treat the whole array as a single global PMF.
        # If spatial_axes is provided, check normalization per non-spatial index
        # by integrating (summing) over spatial_axes only.
        if self.spatial_axes is None:
            total_mass = np.sum(self.fine_pdf)
            if not np.isclose(total_mass, 1.0, rtol=1e-6):
                print(f"Warning: Input total mass = {total_mass:.6f} (should be ~1)")
        else:
            total_mass = np.sum(self.fine_pdf, axis=tuple(self._spatial_axes))
            if not np.allclose(total_mass, 1.0, rtol=1e-6):
                max_dev = np.max(np.abs(total_mass - 1.0))
                print(f"Warning: Max deviation from unit mass over spatial_axes = {max_dev:.6e}")

    # ---------------------------------------------------------------------- #
    # Core N-D downsampling
    # ---------------------------------------------------------------------- #

    def _downsample_along_axis(self, arr: np.ndarray, axis: int, factor: float, bin_centers: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Downsample an array of **discrete probabilities** along a single axis
        using conservative block summation.

        Args:
            arr: Input probability mass array.
            axis: Axis index to downsample.
            factor: Downsampling factor (>1 for downsampling). Must be effectively
                an integer (within a small tolerance).

        Returns:
            new_arr: Array downsampled along the given axis by conservative
                summation of fine bins.
            new_bin_centers: Coarse bin centers along that axis (or None if
                bin_centers is None).
        """
        if factor <= 1.0:
            # No effective downsampling; return input unchanged
            return arr, bin_centers

        axis = int(axis)
        if axis < 0:
            axis += arr.ndim
        if axis < 0 or axis >= arr.ndim:
            raise ValueError(f"axis={axis} is out of bounds for array with ndim={arr.ndim}")

        # Require effectively integer factors for block summation
        factor_int = int(round(factor))
        if factor_int < 1 or not np.isclose(factor_int, factor, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Downsampling factor must be an integer > 1 for discrete PMF mode, got {factor}.")

        # Move target axis to the last dimension for easier handling
        arr_moved = np.moveaxis(arr, axis, -1)
        *leading_shape, N_f = arr_moved.shape

        # Number of full blocks and remainder along the axis
        blocks = N_f // factor_int
        rem = N_f % factor_int

        # Head: full blocks that can be reshaped
        if blocks > 0:
            head = arr_moved[..., : blocks * factor_int]
            new_shape = tuple(leading_shape) + (blocks, factor_int)
            head = head.reshape(new_shape)
            head_sum = head.sum(axis=-1)
        else:
            head_sum = None

        # Tail: any remaining elements are grouped into one final coarse bin
        if rem > 0:
            tail = arr_moved[..., blocks * factor_int :]
            tail_sum = tail.sum(axis=-1, keepdims=False)  # (...,)
            if head_sum is not None:
                coarse_last = np.concatenate([head_sum, tail_sum[..., np.newaxis]], axis=-1)
            else:
                coarse_last = tail_sum[..., np.newaxis]
        else:
            coarse_last = head_sum if head_sum is not None else arr_moved

        new_arr_moved = coarse_last

        # Restore original axis order
        new_arr = np.moveaxis(new_arr_moved, -1, axis)

        # Coarse bin centers: block-average of fine bin centers, if provided
        new_bin_centers = None
        if bin_centers is not None:
            centers = np.asarray(bin_centers, dtype=float)
            if centers.ndim != 1 or centers.shape[0] != N_f:
                raise ValueError(f"bin_centers must be 1D with length {N_f}, got shape {centers.shape}.")

            if blocks > 0:
                head_c = centers[: blocks * factor_int].reshape(blocks, factor_int).mean(axis=-1)
            else:
                head_c = None

            if rem > 0:
                tail_c = centers[blocks * factor_int :].mean(keepdims=False)
                if head_c is not None:
                    new_bin_centers = np.concatenate([head_c, np.asarray([tail_c])], axis=0)
                else:
                    new_bin_centers = np.asarray([tail_c], dtype=float)
            else:
                new_bin_centers = head_c if head_c is not None else centers

        return new_arr, new_bin_centers


    def downsample(self, factors: Union[float, Sequence[float]], axes: Optional[Sequence[int]] = None, method: str = 'fast') -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...]]:
        """
        N-D downsampling along selected axes.

        Args:
            factors: Scalar or sequence of downsampling factors (>1) corresponding
                to the `axes`. If scalar and axes has length > 1, the same factor
                is used for all specified axes.
            axes: Sequence of axis indices to downsample (supports negative
                indices). If None, defaults to the first `len(factors)` axes.
            method: Currently only 'fast' is supported; kept for API compatibility.

        Returns:
            coarse_pdf: Downsampled density array.
            coarse_bin_sizes: 1D array of per-axis spacings after downsampling.
            coarse_bins: Tuple of 1D arrays of bin centers after downsampling.
        """
        if method != 'fast':
            raise ValueError(f"Only method='fast' is supported for N-D downsampling, got method={method!r}")

        # Normalize factors
        if isinstance(factors, (int, float)):
            factors_seq = [float(factors)]
        else:
            factors_seq = [float(f) for f in factors]

        if axes is None:
            axes_seq = list(range(len(factors_seq)))
        else:
            axes_seq = list(axes)

        if len(factors_seq) != len(axes_seq):
            raise ValueError(f"Length of factors ({len(factors_seq)}) must match length of axes ({len(axes_seq)}).")

        # Normalize axes to non-negative indices relative to original ndim
        norm_axes = []
        for ax in axes_seq:
            ax_i = int(ax)
            if ax_i < 0:
                ax_i += self.ndim
            if ax_i < 0 or ax_i >= self.ndim:
                raise ValueError(f"Axis index {ax} is out of bounds for ndim={self.ndim}.")
            norm_axes.append(ax_i)

        if len(set(norm_axes)) != len(norm_axes):
            raise ValueError(f"Axes must be unique, got {axes_seq}.")

        coarse_pdf = np.asarray(self.fine_pdf, dtype=float)
        coarse_bin_sizes = self._bin_sizes_arr.astype(float).copy()
        coarse_bins = [np.asarray(b, dtype=float) for b in self._fine_bins]

        # Apply downsampling along each requested axis using discrete block sums
        for ax, factor in zip(norm_axes, factors_seq):
            coarse_pdf, new_bin_centers = self._downsample_along_axis(
                coarse_pdf, ax, factor, coarse_bins[ax]
            )
            # Update bin metadata for this axis
            if new_bin_centers is not None:
                coarse_bins[ax] = new_bin_centers
                if new_bin_centers.size > 1:
                    diffs = np.diff(new_bin_centers)
                    coarse_bin_sizes[ax] = float(np.median(diffs))
                else:
                    coarse_bin_sizes[ax] = 1.0
            if new_bin_centers is not None:
                coarse_bins[ax] = new_bin_centers

        # Optional renormalization for numerical robustness.
        # If spatial_axes is None, treat the whole array as a single PMF.
        # If spatial_axes is provided, renormalize per non-spatial index so that
        # each slice integrated (summed) over spatial_axes has unit mass.
        if self.spatial_axes is None:
            total_mass = float(np.sum(coarse_pdf))
            if total_mass > 0.0 and not np.isclose(total_mass, 1.0, rtol=1e-10):
                coarse_pdf = coarse_pdf / total_mass
        else:
            spatial_axes_set = set(self._spatial_axes)
            all_axes_set = set(range(coarse_pdf.ndim))
            non_spatial_axes = sorted(all_axes_set - spatial_axes_set)

            if non_spatial_axes:
                # Mass per non-spatial index
                mass = np.sum(coarse_pdf, axis=tuple(self._spatial_axes))
                # Broadcast mass back to full shape
                reshape = [1] * coarse_pdf.ndim
                for i, ax in enumerate(non_spatial_axes):
                    reshape[ax] = mass.shape[i]
                mass_broadcast = mass.reshape(reshape)

                # Avoid division by zero; only renormalize where mass is positive
                with np.errstate(invalid="ignore", divide="ignore"):
                    mask = mass_broadcast > 0.0
                    coarse_pdf = np.divide(coarse_pdf, mass_broadcast, out=coarse_pdf, where=mask)
            else:
                # All axes are spatial; fall back to global renormalization.
                total_mass = float(np.sum(coarse_pdf))
                if total_mass > 0.0 and not np.isclose(total_mass, 1.0, rtol=1e-10):
                    coarse_pdf = coarse_pdf / total_mass

        coarse_bins_tuple = tuple(np.asarray(b, dtype=float) for b in coarse_bins)

        return coarse_pdf, coarse_bin_sizes, coarse_bins_tuple


    def plot_comparison(self, coarse_pdf: np.ndarray, dx_c: float, dy_c: float, figsize: Tuple[int, int] = (12, 5)):
        """Visualize fine vs coarse PDF."""
        import matplotlib.pyplot as plt  # For optional visualization

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        im1 = ax1.imshow(self.fine_pdf.T, origin='lower', cmap='hot', aspect='equal')
        ax1.set_title('Fine PDF')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(coarse_pdf.T, origin='lower', cmap='hot', aspect='equal')
        ax2.set_title('Coarse PDF')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        return fig, (ax1, ax2)


    @classmethod
    def _TEST(cls):
        """ """
        from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update

        _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

        # Example: 2D Gaussian
        Nx_f, Ny_f = 200, 200
        x_f = np.linspace(0, 10, Nx_f)
        y_f = np.linspace(0, 10, Ny_f)
        X_f, Y_f = np.meshgrid(x_f, y_f, indexing='ij')
        mu_x, mu_y, sigma = 5.0, 5.0, 1.0
        fine_pdf = np.exp(-((X_f - mu_x) ** 2 + (Y_f - mu_y) ** 2) / (2 * sigma ** 2))
        dx_f = x_f[1] - x_f[0]
        dy_f = y_f[1] - y_f[0]
        fine_pdf /= np.sum(fine_pdf) * dx_f * dy_f

        downsampler = RigorousPDFDownsampler(fine_pdf, bin_sizes=(dx_f, dy_f))
        coarse_pdf, coarse_bin_sizes, coarse_bins = downsampler.downsample(factors=(4.2, 3.8), axes=(0, 1))
        dx_c, dy_c = coarse_bin_sizes


        fig, (ax1, ax2) = downsampler.plot_comparison(coarse_pdf, dx_c, dy_c)
        return downsampler, fig, (ax1, ax2)




# ==================================================================================================================================================================================================================================================================================== #
# Testing/Evaluation                                                                                                                                                                                                                                                                   #
# ==================================================================================================================================================================================================================================================================================== #
def _test1():
    import matplotlib.pyplot as plt

    # Example: 2D Gaussian
    Nx_f, Ny_f = 200, 200
    x_f = np.linspace(0, 10, Nx_f)
    y_f = np.linspace(0, 10, Ny_f)
    X_f, Y_f = np.meshgrid(x_f, y_f, indexing='ij')
    mu_x, mu_y, sigma = 5.0, 5.0, 1.0
    fine_pdf = np.exp(-((X_f - mu_x) ** 2 + (Y_f - mu_y) ** 2) / (2 * sigma ** 2))
    dx_f = x_f[1] - x_f[0]
    dy_f = y_f[1] - y_f[0]
    fine_pdf /= np.sum(fine_pdf) * dx_f * dy_f

    downsampler = RigorousPDFDownsampler(fine_pdf, bin_sizes=(dx_f, dy_f))

    downsample_factors = [
        (2, 2),
        (4, 4),
        (10, 5),
        (20, 15),
    ]

    # Create a single figure with 4 rows (one for each downsample factor) and 2 columns (fine vs coarse)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))

    for idx, (rx, ry) in enumerate(downsample_factors):
        print(f'(rx: {rx}, ry: {ry})')
        coarse_pdf, coarse_bin_sizes, coarse_bins = downsampler.downsample(factors=(rx, ry), axes=(0, 1))
        dx_c, dy_c = coarse_bin_sizes

        # Left column: Fine PDF (same for all rows)
        ax_fine = axes[idx, 0]
        im1 = ax_fine.imshow(downsampler.fine_pdf.T, origin='lower', cmap='hot', aspect='equal')
        ax_fine.set_title(f'Fine PDF (rx={rx}, ry={ry})')
        plt.colorbar(im1, ax=ax_fine)

        # Right column: Coarse PDF
        ax_coarse = axes[idx, 1]
        im2 = ax_coarse.imshow(coarse_pdf.T, origin='lower', cmap='hot', aspect='equal')
        ax_coarse.set_title(f'Coarse PDF (shape: {coarse_pdf.shape})')
        plt.colorbar(im2, ax=ax_coarse)

    plt.tight_layout()
    plt.show()  # Wait until the plot windows are closed before exiting


# def _test2():

#     from neuropy.utils.probability_downsampling import RigorousPDFDownsampler

#     p_x_given_n = decoding_locality_measures.p_x_given_n ## np.shape(p_x_given_n) # (62, 62, 2, 151732)
#     fine_pdf = p_x_given_n

#     ## before downsampling
#     fine_norm_sum = np.nansum(fine_pdf, axis=(0, 1, 2))
#     fine_norm_sum # array([1, 1, 1, ..., 1, 1, 1]) -- summing each t_step over all position bins yields 1 because that's how a probability density function over space should be normalized.

#     ## Perform downsampling:
#     downsampler.bins
#         # (array([4.27832, 6.50776, 8.73721, 10.9667, 13.1961, 15.4255, 17.655, 19.8844, 22.1139, 24.3433, 26.5728, 28.8022, 31.0316, 33.2611, 35.4905, 37.72, 39.9494, 42.1789, 44.4083, 46.6377, 48.8672, 51.0966, 53.3261, 55.5555, 57.785, 60.0144, 62.2438, 64.4733, 66.7027, 68.9322, 71.1616, 73.3911, 75.6205, 77.8499, 80.0794, 82.3088, 84.5383, 86.7677, 88.9972, 91.2266, 93.456, 95.6855, 97.9149, 100.144, 102.374, 104.603, 106.833, 109.062, 111.292, 113.521, 115.75, 117.98, 120.209, 122.439, 124.668, 126.898, 129.127, 131.357, 133.586, 135.815, 138.045, 140.274]),
#         #  array([1.99629, 3.45248, 4.90867, 6.36486, 7.82104, 9.27723, 10.7334, 12.1896, 13.6458, 15.102, 16.5582, 18.0144, 19.4705, 20.9267, 22.3829, 23.8391, 25.2953, 26.7515, 28.2077, 29.6639, 31.12, 32.5762, 34.0324, 35.4886, 36.9448, 38.401, 39.8572, 41.3134, 42.7695, 44.2257, 45.6819, 47.1381, 48.5943, 50.0505, 51.5067, 52.9628, 54.419, 55.8752, 57.3314, 58.7876, 60.2438, 61.7, 63.1562, 64.6123, 66.0685, 67.5247, 68.9809, 70.4371, 71.8933, 73.3495, 74.8057, 76.2618, 77.718, 79.1742, 80.6304, 82.0866, 83.5428, 84.999, 86.4552, 87.9113, 89.3675, 90.8237]),
#         #  array([0, 1]),
#         #  array([0.125, 0.375, 0.625, ..., 37932.4, 37932.6, 37932.9]))
#     coarse_bins
#         # (array([8.47996, 19.1127, 29.7454, 40.3781, 51.0109, 61.6436, 72.2763, 82.9091, 93.5418, 104.175, 114.807, 125.44, 136.073]),
#         #  array([4.74065, 11.6855, 18.6304, 25.5753, 32.5202, 39.4651, 46.41, 53.3549, 60.2998, 67.2447, 74.1896, 81.1345, 88.0794]),
#         #  array([0, 1]),
#         #  array([0.125, 0.375, 0.625, ..., 37932.4, 37932.6, 37932.9]))
#     # downsampler = RigorousPDFDownsampler(fine_pdf, bins=(decoding_locality_measures.xbin_centers, decoding_locality_measures.ybin_centers, None, decoding_locality_measures.time_window_centers)) ## Warning: Input total mass = 123148.967784 (should be ~1)
#     downsampler = RigorousPDFDownsampler(fine_pdf, spatial_axes=(0, 1, 2)) ## Warning: Input total mass = 123148.967784 (should be ~1)
#     assert np.allclose(fine_norm_sum, 1)
#     rx = 5
#     ry = 5
#     print(f'(rx: {rx}, ry: {ry})')
#     coarse_pdf, coarse_bin_sizes, coarse_bins = downsampler.downsample(factors=(rx, ry), axes=(0, 1)) ## 1m for ## np.shape(p_x_given_n) # (62, 62, 2, 151732)
#     np.shape(coarse_pdf) # (13, 13, 2, 151732)

#     ## after downsampling
#     coarse_norm_sum = np.nansum(coarse_pdf, axis=(0, 1, 2))
#     # np.shape(norm_sum)
#     coarse_norm_sum # array([3.57004e-07, 3.57004e-07, 3.57004e-07, ..., 3.57004e-07, 3.57004e-07, 3.57004e-07]) -- !! unfortunately after downsampling, the sum over all position bins for each time bin do NOT sum to 1.0, meaning they aren't valid PDF functions. Why is this? Is there a normalization error in `RigorousPDFDownsampler` or am I missing something conceptually?
#     np.shape(coarse_pdf)
#     assert np.allclose(coarse_norm_sum, 1)



# Example Usage: 2D Gaussian PDF
if __name__ == "__main__":
    _test1()


