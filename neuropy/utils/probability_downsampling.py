import numpy as np
from typing import Tuple, Optional, Sequence, Union
from attrs import define, field
from neuropy.utils.mixins.AttrsClassHelpers import SimpleFieldSizesReprMixin


ArrayLike = Union[np.ndarray, Sequence[float]]


# @metadata_attributes(short_name=None, tags=['UNFINISHED', 'UNEVALUATED', 'AI'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-17 09:17', related_items=[])
@define(slots=False, eq=False, repr=False)
class RigorousPDFDownsampler(SimpleFieldSizesReprMixin):
    """
    Mathematically rigorous N-D PDF downsampler using conservative coarse-graining.

    Preserves:
    - Total integrated probability (≈ 1)
    - Relative densities (ratios of integrated masses)

    Handles:
    - Arbitrary (non-integer) downsampling factors per axis
    - Partial cell overlaps via cumulative integration and interpolation
    - Uniform grids only (extension to non-uniform possible)

    Input:
        fine_pdf: N-D array of densities with shape (n0, n1, ..., n_{D-1})
        bin_sizes: Optional 1D sequence of length D giving grid spacing along each axis
        bins: Optional sequence of 1D arrays giving bin centers per axis. If provided
            (with or without bin_sizes), per-axis spacings are inferred from the median
            of np.diff(bins[d]). If an entry is None, a default index-based grid
            np.arange(shape_f[d]) is used and spacing 1.0 is assumed.

    Only the user-specified axes are downsampled; all other axes are treated as
    independent and left unchanged. For example, a posterior with shape
    (n_x, n_y, n_t) and axes=(0, 1), factors=(a, b) yields an output of shape
    (ceil(n_x/a), ceil(n_y/b), n_t).
    """

    fine_pdf: np.ndarray = field()
    bin_sizes: Optional[ArrayLike] = field(default=None)
    bins: Optional[Sequence[Optional[ArrayLike]]] = field(default=None)

    ndim: int = field(init=False)
    shape_f: Tuple[int, ...] = field(init=False)
    _bin_sizes_arr: np.ndarray = field(init=False, repr=False)
    _fine_bins: Tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __attrs_post_init__(self):
        """Validates input and computes derived attributes."""
        if self.fine_pdf.ndim < 1:
            raise ValueError("fine_pdf must be at least 1D array")

        self.shape_f = self.fine_pdf.shape
        self.ndim = self.fine_pdf.ndim

        # Infer bin sizes and canonical fine bins from either bin_sizes or bins.
        if self.bin_sizes is None and self.bins is None:
            raise ValueError("Either bin_sizes or bins must be provided.")

        fine_bins_list = []

        if self.bins is not None and self.bin_sizes is None:
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

        # Verify input is roughly normalized (tolerance for numerics)
        total_mass = np.sum(self.fine_pdf) * float(np.prod(self._bin_sizes_arr))
        if not np.isclose(total_mass, 1.0, rtol=1e-6):
            print(f"Warning: Input total mass = {total_mass:.6f} (should be ~1)")

    # ---------------------------------------------------------------------- #
    # Core N-D downsampling
    # ---------------------------------------------------------------------- #

    def _downsample_along_axis(self, arr: np.ndarray, axis: int, factor: float, spacing: float, bin_centers: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """
        Downsample an array of densities along a single axis using rigorous
        integration via cumulative sums and interpolation.

        Args:
            arr: Input density array.
            axis: Axis index to downsample.
            factor: Downsampling factor (>1 for downsampling).
            spacing: Bin spacing along the given axis.

        Returns:
            new_arr: Density array downsampled along the given axis.
            new_spacing: New spacing along that axis.
            new_bin_centers: Coarse bin centers along that axis (or None if bin_centers is None).
        """
        if factor <= 1.0:
            # No effective downsampling; return input unchanged
            return arr, spacing, bin_centers

        axis = int(axis)
        if axis < 0:
            axis += arr.ndim
        if axis < 0 or axis >= arr.ndim:
            raise ValueError(f"axis={axis} is out of bounds for array with ndim={arr.ndim}")

        # Move target axis to the last dimension for easier handling
        arr_moved = np.moveaxis(arr, axis, -1)
        *leading_shape, N_f = arr_moved.shape
        leading_size = int(np.prod(leading_shape)) if leading_shape else 1

        # Compute coarse grid parameters
        N_c = int(np.ceil(N_f / factor))
        total_length = N_f * spacing
        dx_c = total_length / N_c

        # Determine physical edges of the fine and coarse grids. If bin_centers
        # are provided, align edges with them; otherwise assume origin 0.
        if bin_centers is not None and bin_centers.size > 0:
            diffs_centers = np.diff(bin_centers) if bin_centers.size > 1 else np.array([spacing])
            spacing_from_centers = float(np.median(diffs_centers))
            if np.isfinite(spacing_from_centers) and spacing_from_centers > 0:
                spacing = spacing_from_centers
            start_edge = float(bin_centers[0] - 0.5 * spacing)
        else:
            start_edge = 0.0

        fine_edges = start_edge + np.arange(N_f + 1, dtype=float) * spacing
        coarse_edges = start_edge + np.arange(N_c + 1, dtype=float) * dx_c

        # Flatten all non-axis dimensions into one for efficient looping
        arr_flat = arr_moved.reshape(leading_size, N_f)

        # Mass along the axis: density * spacing
        fine_mass = arr_flat * spacing  # shape (leading_size, N_f)
        cum_mass = np.insert(np.cumsum(fine_mass, axis=1), 0, 0.0, axis=1)  # (leading_size, N_f+1)

        interp = np.empty((leading_size, N_c + 1), dtype=float)
        for i in range(leading_size):
            interp[i, :] = np.interp(coarse_edges, fine_edges, cum_mass[i, :])

        masses = np.diff(interp, axis=1)  # (leading_size, N_c)

        # Convert back to densities along the axis
        densities_flat = masses / dx_c
        new_shape = tuple(leading_shape) + (N_c,)
        new_arr_moved = densities_flat.reshape(new_shape)

        # Restore original axis order
        new_arr = np.moveaxis(new_arr_moved, -1, axis)

        # Coarse bin centers are midpoints of the coarse edges.
        new_bin_centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])

        return new_arr, dx_c, new_bin_centers

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

        # Apply downsampling along each requested axis
        for ax, factor in zip(norm_axes, factors_seq):
            coarse_pdf, new_spacing, new_bin_centers = self._downsample_along_axis(
                coarse_pdf, ax, factor, coarse_bin_sizes[ax], coarse_bins[ax]
            )
            coarse_bin_sizes[ax] = new_spacing
            if new_bin_centers is not None:
                coarse_bins[ax] = new_bin_centers

        # Optional small renormalization for numerical robustness
        total_mass = float(np.sum(coarse_pdf) * np.prod(coarse_bin_sizes))
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


# Example Usage: 2D Gaussian PDF
if __name__ == "__main__":

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
