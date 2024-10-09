"""
This type stub file was generated by pyright.
"""

from .. import core

def hmmfit1d(Data): # -> NDArray[floating[Any]] | NDArray[Any]:
    ...

def correlation_emg(signal: core.Signal, probe: core.ProbeGroup, window, overlap, n_jobs=...): # -> NDArray[Any]:
    """Calculating emg

    Parameters
    ----------
    window : int
        window size in seconds
    overlap : float
        overlap between windows in seconds
    n_jobs: int,
        number of cpu/processes to use

    Returns
    -------
    array
        emg calculated at each time window
    """
    ...

def detect_brainstates_epochs(signal: core.Signal, probe: core.ProbeGroup, window=..., overlap=..., emg: core.Signal = ..., sigma=..., ignore_epochs=...): # -> Epoch:
    """detects sleep states for the recording

    Parameters
    ----------
    chans : int, optional
        channel you want to use for sleep detection, by default None
    window : int, optional
        bin size, by default 1
    overlap : float, optional
        seconds of overlap between adjacent window , by default 0.2
    emgfile : bool, optional
        if True load the emg file in the basepath, by default False

    """
    ...

