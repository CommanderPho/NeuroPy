import numpy as np
from typing import Tuple, Optional
from attrs import define, field
from neuropy.utils.mixins.AttrsClassHelpers import SimpleFieldSizesReprMixin

# @metadata_attributes(short_name=None, tags=['UNFINISHED', 'UNEVALUATED', 'AI'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-17 09:17', related_items=[])
@define(slots=False, eq=False, repr=False)
class RigorousPDFDownsampler(SimpleFieldSizesReprMixin):
    """
    Mathematically rigorous 2D PDF downsampler using conservative coarse-graining.
    
    Preserves:
    - Total integrated probability (=1)
    - Relative densities (ratios of integrated masses)
    
    Handles:
    - Arbitrary (non-integer) downsampling factors
    - Partial cell overlaps with exact fractional weighting
    - Uniform grids only (extension to non-uniform possible)
    
    Input: fine_pdf (Nx x Ny array of densities p[k,l])
    Coarse cells: Integrate via overlap-weighted sum of fine masses.


    Usage:

        from neuropy.utils.probability_downsampling import RigorousPDFDownsampler

        # app = pg.mkApp('RigorousPDFDownsampler test')
        _out = RigorousPDFDownsampler._TEST()
        _out

    """
    
    fine_pdf: np.ndarray = field()
    dx_f: float = field()
    dy_f: float = field()
    Ny_f: int = field(init=False)
    Nx_f: int = field(init=False)
    
    def __attrs_post_init__(self):
        """
        Validates input and computes derived attributes.
        """
        if self.fine_pdf.ndim != 2:
            raise ValueError("fine_pdf must be 2D array (Ny, Nx)")
        self.Ny_f, self.Nx_f = self.fine_pdf.shape
        
        # Verify input is roughly normalized (tolerance for numerics)
        total_mass = np.sum(self.fine_pdf) * self.dx_f * self.dy_f
        if not np.isclose(total_mass, 1.0, rtol=1e-6):
            print(f"Warning: Input total mass = {total_mass:.6f} (should be ~1)")
    

    def downsample(self, rx: float, ry: Optional[float]=None, method: str = 'trapezoidal') -> Tuple[np.ndarray, float, float]:
        """
        Downsample with factors rx, ry (>1 for downsampling).
        
        Args:
            rx, ry: Downsampling ratios (Nx_c = Nx_f / rx, etc.)
            method: 'sum' (discrete mass sum, exact for aligned), 'trapezoidal' (continuous approx)
        
        Returns:
            coarse_pdf: (Ny_c, Nx_c) density array
            dx_c, dy_c: Coarse spacings
        """
        if ry is None:
            ry = rx ## same scale factor
        Nx_c_float = self.Nx_f / rx
        Ny_c_float = self.Ny_f / ry
        Nx_c = int(np.ceil(Nx_c_float))
        Ny_c = int(np.ceil(Ny_c_float))
        actual_rx = self.Nx_f / Nx_c
        actual_ry = self.Ny_f / Ny_c
        if not np.isclose(rx, actual_rx, rtol=1e-12):
            print(f"Warning: Downsampling factor rx={rx} was rounded to an effective factor of {actual_rx:.6f} in x (Nx_c={Nx_c})")
        if not np.isclose(ry, actual_ry, rtol=1e-12):
            print(f"Warning: Downsampling factor ry={ry} was rounded to an effective factor of {actual_ry:.6f} in y (Ny_c={Ny_c})")

        dx_c = self.Nx_f * self.dx_f / Nx_c  # Adjusted for integer grid
        dy_c = self.Ny_f * self.dy_f / Ny_c
        
        coarse_pdf = np.zeros((Ny_c, Nx_c))
        total_coarse_mass = 0.0
        
        for i in range(Nx_c):
            x_left = i * dx_c
            x_right = min((i + 1) * dx_c, self.Nx_f * self.dx_f)
            
            for j in range(Ny_c):
                y_bottom = j * dy_c
                y_top = min((j + 1) * dy_c, self.Ny_f * self.dy_f)
                
                # Fine indices overlapping this coarse cell
                k_left = max(0, int(np.floor(x_left / self.dx_f)))
                k_right = min(self.Nx_f, int(np.ceil(x_right / self.dx_f)))
                l_bottom = max(0, int(np.floor(y_bottom / self.dy_f)))
                l_top = min(self.Ny_f, int(np.ceil(y_top / self.dy_f)))
                
                P_ij = 0.0
                for k in range(k_left, k_right):
                    for l in range(l_bottom, l_top):
                        # Exact overlap fraction for fine cell [xk, xk+dx_f] x [yl, yl+dy_f]
                        xk_left = k * self.dx_f
                        xk_right = min((k + 1) * self.dx_f, self.Nx_f * self.dx_f)
                        yl_bottom = l * self.dy_f
                        yl_top = min((l + 1) * self.dy_f, self.Ny_f * self.dy_f)
                        
                        overlap_x = min(x_right, xk_right) - max(x_left, xk_left)
                        overlap_y = min(y_top, yl_top) - max(y_bottom, yl_bottom)
                        if overlap_x > 0 and overlap_y > 0:
                            w_kl = (overlap_x * overlap_y) / (self.dx_f * self.dy_f)
                            P_ij += self.fine_pdf[l, k] * w_kl
                
                coarse_pdf[j, i] = P_ij / (dx_c * dy_c)  # Density = mass / area
                total_coarse_mass += P_ij
        
        print(f"Mass conservation: input={np.sum(self.fine_pdf)*self.dx_f*self.dy_f:.8f}, "
              f"output={total_coarse_mass:.8f}, error={abs(1-total_coarse_mass):.2e}")
        
        return coarse_pdf, dx_c, dy_c
    

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

        # Example: 2D Gaussian (PASTE THIS INTO JUPYTER)
        Nx_f, Ny_f = 200, 200
        x_f = np.linspace(0, 10, Nx_f)
        y_f = np.linspace(0, 10, Ny_f)
        X_f, Y_f = np.meshgrid(x_f, y_f)
        mu_x, mu_y, sigma = 5.0, 5.0, 1.0
        fine_pdf = np.exp(-((X_f - mu_x)**2 + (Y_f - mu_y)**2) / (2 * sigma**2))
        dx_f = np.median(X_f)
        dy_f = np.median(Y_f)
        fine_pdf /= np.sum(fine_pdf) * dx_f * dy_f  # <-- THE FIXED LINE

        downsampler = RigorousPDFDownsampler(fine_pdf, dx_f, dy_f)
        coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=4.2, ry=3.8)
        fig, (ax1, ax2) = downsampler.plot_comparison(coarse_pdf, dx_c, dy_c)
        return downsampler, fig, (ax1, ax2)








# @function_attributes(short_name=None, tags=['downsample', 'efficiency', 'speed-up', 'approximate', 'pdf', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-12 14:14', related_items=[])
def approx_downsample_pdf(a_p_x_given_n, downsample_factor=4, xbin_centers=None, ybin_centers=None):
    """
    Approximates a downsampled version of a 2D or 3D positional probability
    density function (PDF) while preserving total probability mass.

    Supports:
        - 3D PDFs: shape (H, W, T)
        - 2D PDFs: shape (H, W)

    Also optionally downsamples xbin_centers and ybin_centers consistently
    with the spatial downsampling.

    Args:
        a_p_x_given_n : np.ndarray, (H, W) or (H, W, T)
        downsample_factor : int
        xbin_centers : optional 1D array of length H
        ybin_centers : optional 1D array of length W

    Returns:
        a_p_x_given_n_small
        xbin_centers_small (or None)
        ybin_centers_small (or None)

    USage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import approx_downsample_pdf
        a_p_x_given_n_small, xbin_centers_small, ybin_centers_small = approx_downsample_pdf(a_p_x_given_n=a_p_x_given_n, xbin_centers = _obj.xbin_centers, ybin_centers = _obj.ybin_centers, downsample_factor=4)
        np.shape(a_p_x_given_n_small)
        xbin_centers_small

    """

    a = np.asarray(a_p_x_given_n)
    k = downsample_factor

    # --- handle 2D → 3D ---
    if a.ndim == 2:
        a = a[:, :, None]
        squeeze_back = True
    else:
        squeeze_back = False

    H, W, T = a.shape

    # compute padding
    pad_h = (k - (H % k)) % k
    pad_w = (k - (W % k)) % k

    # pad
    a_padded = np.pad(
        a,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode='edge'
    ).astype(float)

    Hc, Wc = a_padded.shape[0], a_padded.shape[1]

    # block-sum PDF values
    a_blocksum = a_padded.reshape(Hc//k, k, Wc//k, k, T).sum(axis=(1, 3))

    # renormalize per timestamp
    mass = a_blocksum.sum(axis=(0, 1), keepdims=True)
    mass_safe = np.where(mass == 0, 1.0, mass)
    a_p_x_given_n_small = a_blocksum / mass_safe

    if squeeze_back:
        a_p_x_given_n_small = a_p_x_given_n_small[:, :, 0]

    # ---------------------------
    # Downsample xbin_centers, ybin_centers
    # ---------------------------

    # xbin_centers (H dimension)
    if xbin_centers is not None:
        xbin_centers = np.asarray(xbin_centers)

        # pad if needed
        if pad_h > 0:
            xbin_centers = np.pad(
                xbin_centers,
                pad_width=(0, pad_h),
                mode='edge'
            )

        # reshape into blocks and average
        xbin_centers_small = xbin_centers.reshape(-1, k).mean(axis=1)
    else:
        xbin_centers_small = None

    # ybin_centers (W dimension)
    if ybin_centers is not None:
        ybin_centers = np.asarray(ybin_centers)

        if pad_w > 0:
            ybin_centers = np.pad(
                ybin_centers,
                pad_width=(0, pad_w),
                mode='edge'
            )

        ybin_centers_small = ybin_centers.reshape(-1, k).mean(axis=1)
    else:
        ybin_centers_small = None

    return a_p_x_given_n_small, xbin_centers_small, ybin_centers_small



# Example Usage: 2D Gaussian PDF
if __name__ == "__main__":

    # Example: 2D Gaussian (PASTE THIS INTO JUPYTER)
    Nx_f, Ny_f = 200, 200
    x_f = np.linspace(0, 10, Nx_f)
    y_f = np.linspace(0, 10, Ny_f)
    X_f, Y_f = np.meshgrid(x_f, y_f)
    mu_x, mu_y, sigma = 5.0, 5.0, 1.0
    fine_pdf = np.exp(-((X_f - mu_x)**2 + (Y_f - mu_y)**2) / (2 * sigma**2))
    dx_f = np.median(X_f)
    dy_f = np.median(Y_f)
    fine_pdf /= np.sum(fine_pdf) * dx_f * dy_f  # <-- THE FIXED LINE

    downsampler = RigorousPDFDownsampler(fine_pdf, dx_f, dy_f)
    coarse_pdf, dx_c, dy_c = downsampler.downsample(rx=10, ry=5)
    fig, (ax1, ax2) = downsampler.plot_comparison(coarse_pdf, dx_c, dy_c)
    fig.show()