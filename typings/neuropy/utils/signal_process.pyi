"""
This type stub file was generated by pyright.
"""

import numpy as np
from dataclasses import dataclass
from .. import core

class filter_sig:
    @staticmethod
    def bandpass(signal, hf, lf, fs=..., order=..., ax=...): # -> Signal | NDArray[Any]:
        ...
    
    @staticmethod
    def highpass(signal, cutoff, fs=..., order=..., ax=...): # -> NDArray[Any]:
        ...
    
    @staticmethod
    def lowpass(signal, cutoff, fs=..., order=..., ax=...): # -> NDArray[Any]:
        ...
    
    @staticmethod
    def delta(signal, fs=..., order=..., ax=...): # -> Signal | NDArray[Any]:
        ...
    
    @staticmethod
    def theta(signal, fs=..., order=..., ax=...): # -> Signal | NDArray[Any]:
        ...
    
    @staticmethod
    def spindle(signal, fs=..., order=..., ax=...): # -> Signal | NDArray[Any]:
        ...
    
    @staticmethod
    def slowgamma(signal, fs=..., order=..., ax=...): # -> Signal | NDArray[Any]:
        ...
    
    @staticmethod
    def mediumgamma(signal, fs=..., order=..., ax=...): # -> Signal | NDArray[Any]:
        ...
    
    @staticmethod
    def fastgamma(signal, fs=..., order=..., ax=...): # -> Signal | NDArray[Any]:
        ...
    
    @staticmethod
    def ripple(signal, fs=..., order=..., ax=...): # -> Signal | NDArray[Any]:
        ...
    


def whiten(strain, interp_psd, dt): # -> NDArray[float64]:
    ...

class SpectrogramBands:
    def __init__(self, signal: core.Signal, window: float = ..., overlap=..., smooth=..., multitaper=..., norm_sig=...) -> None:
        ...
    
    def get_band_power(self, f1=..., f2=...):
        ...
    
    @property
    def delta(self):
        ...
    
    @property
    def deltaplus(self):
        ...
    
    @property
    def theta(self):
        ...
    
    @property
    def spindle(self):
        ...
    
    @property
    def gamma(self):
        ...
    
    @property
    def ripple(self):
        ...
    
    @property
    def theta_delta_ratio(self):
        ...
    
    @property
    def theta_deltaplus_ratio(self):
        ...
    
    def plotSpect(self, ax=..., freqRange=...): # -> None:
        ...
    


@dataclass
class wavelet_decomp:
    lfp: np.array
    freqs: np.array = ...
    sampfreq: int = ...
    def colgin2009(self): # -> NDArray[signedinteger[Any]]:
        """colgin


        Returns:
            [type]: [description]

        References
        ------------
        1) Colgin, L. L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., ... & Moser, E. I. (2009). Frequency of gamma oscillations routes flow of information in the hippocampus. Nature, 462(7271), 353-357.
        2) Tallon-Baudry, C., Bertrand, O., Delpuech, C., & Pernier, J. (1997). Oscillatory γ-band (30–70 Hz) activity induced by a visual search task in humans. Journal of Neuroscience, 17(2), 722-734.
        """
        ...
    
    def quyen2008(self): # -> NDArray[signedinteger[Any]]:
        """colgin


        Returns:
            [type]: [description]

        References
        ------------
        1) Le Van Quyen, M., Bragin, A., Staba, R., Crépon, B., Wilson, C. L., & Engel, J. (2008). Cell type-specific firing during ripple oscillations in the hippocampal formation of humans. Journal of Neuroscience, 28(24), 6104-6110.
        """
        ...
    
    def bergel2018(self): # -> NDArray[floating[Any]]:
        """colgin


        Returns:
            [type]: [description]

        References:
        ---------------
        1) Bergel, A., Deffieux, T., Demené, C., Tanter, M., & Cohen, I. (2018). Local hippocampal fast gamma rhythms precede brain-wide hyperemic patterns during spontaneous rodent REM sleep. Nature communications, 9(1), 1-12.

        """
        ...
    
    def torrenceCompo(self): # -> None:
        ...
    
    def cohen(self, ncycles=...): # -> NDArray[signedinteger[Any]]:
        """Implementation of ref. 1 chapter 13


        Returns:
            [type]: [description]

        References:
        ---------------
        1) Cohen, M. X. (2014). Analyzing neural time series data: theory and practice. MIT press.

        """
        ...
    


def hilbertfast(signal, ax=...): # -> tuple[Dispatchable, ...]:
    """inputs a signal does padding to next power of 2 for faster computation of hilbert transform

    Arguments:
        signal {array} -- [n, dimensional array]

    Returns:
        [type] -- [description]
    """
    ...

def fftnormalized(signal, fs=...): # -> tuple[Any, NDArray[floating[Any]]]:
    ...

@dataclass
class bicoherence:
    """Generate bicoherence matrix for signal

    Attributes:
    ---------------
        flow: int, low frequency
        fhigh: int, highest frequency
        window: int, segment size
        noverlap:

        bicoher (freq_req x freq_req, array): bicoherence matrix
        freq {array}: frequencies at which bicoherence was calculated
        bispec:
        significance:

    Methods:
    ---------------
    compute
        calculates the bicoherence
    plot
        plots bicoherence matrix in the provided axis

    References:
    -----------------------
    1) Sheremet, A., Burke, S. N., & Maurer, A. P. (2016). Movement enhances the nonlinearity of hippocampal theta. Journal of Neuroscience, 36(15), 4218-4230.
    """
    flow: int = ...
    fhigh: int = ...
    fs: int = ...
    window: int = ...
    overlap: int = ...
    def compute(self, signal: np.array): # -> Any:
        """Computes bicoherence

        Parameters
        -----------
            signal: array,
                lfp signal on which bicoherence will be calculated
        """
        ...
    
    def plot(self, index=..., ax=..., smooth=..., **kwargs): # -> None:
        ...
    


def power_correlation(signal, fs=..., window=..., overlap=..., fband=...): # -> tuple[Any, Any]:
    """Power power correlation between frequencies

    Parameters
    ----------
    signal : [type]
        timeseries for which to calculate
    fs : int, optional
        sampling frequency of the signal, by default 1250
    window : int, optional
        window size for calculating spectrogram, by default 2
    overlap : int, optional
        overlap between adjacent windows, by default 1
    fband : [type], optional
        return correlations between these frequencies only, by default None

    Returns
    -------
    [type]
        [description]
    """
    ...

@dataclass
class Csd:
    lfp: np.array
    coords: np.array
    chan_label: np.array = ...
    fs: int = ...
    def classic(self): # -> None:
        ...
    
    def icsd(self, lfp, coords): # -> None:
        ...
    
    def plot(self, ax=..., smooth=..., plotLFP=..., **kwargs): # -> None:
        ...
    


def mtspect(signal, nperseg, noverlap, fs=...): # -> tuple[Any, Any]:
    ...

@dataclass
class PAC:
    """Phase amplitude coupling

    Attributes
    ----------

    Methods
    -------
    compute(lfp)
        calculates phase amplitude coupling
    """
    fphase: tuple = ...
    famp: tuple = ...
    binsz: int = ...
    def compute(self, lfp): # -> None:
        ...
    
    def comodulo(self, lfp, method=..., njobs=...): # -> None:
        """comodulogram for frequencies of interest"""
        ...
    
    def plot(self, ax=..., **kwargs): # -> Any:
        """Bar plot for phase amplitude coupling

        Parameters
        ----------
        ax : axis object, optional
            axis to plot into, by default None
        kwargs : other keyword arguments
            arguments are to plt.bar()

        Returns
        -------
        ax : matplotlib axes
            Axes object with the heatmap
        """
        ...
    


@dataclass
class ThetaParams:
    """Estimating various theta oscillation features like phase, asymmetry etc.

    References
    -------
    1) hilbert --> Cole, Scott, and Bradley Voytek. "Cycle-by-cycle analysis of neural oscillations." Journal of neurophysiology (2019)
    2) waveshape --> Belluscio, Mariano A., et al. "Cross-frequency phase–phase coupling between theta and gamma oscillations in the hippocampus." Journal of Neuroscience(2012)
    """
    lfp: np.array
    fs: int = ...
    method: str = ...
    def __post_init__(self): # -> None:
        ...
    
    @property
    def rise_mid(self): # -> NDArray[Any]:
        ...
    
    @property
    def fall_mid(self): # -> NDArray[Any]:
        ...
    
    @property
    def peak_width(self): # -> NDArray[floating[Any]]:
        ...
    
    @property
    def trough_width(self): # -> NDArray[floating[Any]]:
        ...
    
    @property
    def asymmetry(self):
        ...
    
    @property
    def peaktrough(self): # -> NDArray[floating[Any]]:
        ...
    
    def break_by_phase(self, y, binsize=..., slideby=...): # -> tuple[list[Any], NDArray[signedinteger[Any]] | NDArray[Any], NDArray[floating[Any]]]:
        """Breaks y into theta phase specific components

        Parameters
        ----------
        lfp : array like
            reference lfp from which theta phases are estimated
        y : array like
            timeseries which is broken into components
        binsize : int, optional
            width of each bin in degrees, by default 20
        slideby : int, optional
            slide each bin by this amount in degrees, by default None

        Returns
        -------
        [list]
            list of broken signal into phase components
        """
        ...
    
    def sanityCheck(self): # -> Any:
        """Plots raw signal with filtered signal and peak, trough locations with phase

        Returns
        -------
        ax : obj
        """
        ...
    


def psd_auc(signal: core.Signal, freq_band: tuple, window=..., overlap=...): # -> list[Any]:
    """Calculates area under the power spectrum for a given frequency band

    Parameters
    ----------
    eeg : [array]
        channels x time, has to be two dimensional

    Returns
    -------
    [type]
        [description]
    """
    ...

def hilbert_ampltiude_stat(signals, freq_band, fs, statistic=...): # -> NDArray[float64]:
    """Calculates hilbert amplitude statistic over the entire signal

    Parameters
    ----------
    signals : list of signals or np.array
        [description]
    statistic : str, optional
        [description], by default "mean"

    Returns
    -------
    [type]
        [description]
    """
    ...

def theta_phase_specfic_extraction(signal, y, fs, binsize=..., slideby=...): # -> tuple[list[Any], NDArray[signedinteger[Any]], Any]:
    """Breaks y into theta phase specific components

    Parameters
    ----------
    signal : array like
        reference lfp from which theta phases are estimated
    y : array like
        timeseries which is broken into components
    binsize : int, optional
        width of each bin in degrees, by default 20
    slideby : int, optional
        slide each bin by this amount in degrees, by default None

    Returns
    -------
    [list]
        list of broken signal into phase components
    """
    ...

