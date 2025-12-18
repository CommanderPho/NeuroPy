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
        bin_sizes: 1D sequence of length D giving grid spacing along each axis

    Only the user-specified axes are downsampled; all other axes are treated as
    independent and left unchanged. For example, a posterior with shape
    (n_x, n_y, n_t) and axes=(0, 1), factors=(a, b) yields an output of shape
    (ceil(n_x/a), ceil(n_y/b), n_t).
    """

    fine_pdf: np.ndarray = field()
    bin_sizes: ArrayLike = field()

    ndim: int = field(init=False)
    shape_f: Tuple[int, ...] = field(init=False)
    _bin_sizes_arr: np.ndarray = field(init=False, repr=False)

    def __attrs_post_init__(self):
        """Validates input and computes derived attributes."""
        if self.fine_pdf.ndim < 1:
            raise ValueError("fine_pdf must be at least 1D array")

        self.shape_f = self.fine_pdf.shape
        self.ndim = self.fine_pdf.ndim

        bin_sizes_arr = np.asarray(self.bin_sizes, dtype=float)
        if bin_sizes_arr.ndim != 1 or bin_sizes_arr.shape[0] != self.ndim:
            raise ValueError(f"bin_sizes must be 1D with length {self.ndim}, got shape {bin_sizes_arr.shape}")
        if np.any(bin_sizes_arr <= 0):
            raise ValueError("All bin_sizes must be positive.")
        self._bin_sizes_arr = bin_sizes_arr

        # Verify input is roughly normalized (tolerance for numerics)
        total_mass = np.sum(self.fine_pdf) * float(np.prod(self._bin_sizes_arr))
        if not np.isclose(total_mass, 1.0, rtol=1e-6):
            print(f"Warning: Input total mass = {total_mass:.6f} (should be ~1)")

    # ---------------------------------------------------------------------- #
    # Core N-D downsampling
    # ---------------------------------------------------------------------- #

    def _downsample_along_axis(self, arr: np.ndarray, axis: int, factor: float, spacing: float) -> Tuple[np.ndarray, float]:
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
        """
        if factor <= 1.0:
            # No effective downsampling; return input unchanged
            return arr, spacing

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

        fine_edges = np.arange(N_f + 1, dtype=float) * spacing
        coarse_edges = np.arange(N_c + 1, dtype=float) * dx_c

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
        return new_arr, dx_c

    def downsample(self, factors: Union[float, Sequence[float]], axes: Optional[Sequence[int]] = None, method: str = 'fast') -> Tuple[np.ndarray, np.ndarray]:
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

        # Apply downsampling along each requested axis
        for ax, factor in zip(norm_axes, factors_seq):
            coarse_pdf, new_spacing = self._downsample_along_axis(coarse_pdf, ax, factor, coarse_bin_sizes[ax])
            coarse_bin_sizes[ax] = new_spacing

        # Optional small renormalization for numerical robustness
        total_mass = float(np.sum(coarse_pdf) * np.prod(coarse_bin_sizes))
        if total_mass > 0.0 and not np.isclose(total_mass, 1.0, rtol=1e-10):
            coarse_pdf = coarse_pdf / total_mass

        return coarse_pdf, coarse_bin_sizes


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
        coarse_pdf, coarse_bin_sizes = downsampler.downsample(factors=(4.2, 3.8), axes=(0, 1))
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
        coarse_pdf, coarse_bin_sizes = downsampler.downsample(factors=(rx, ry), axes=(0, 1))
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
