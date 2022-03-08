import numpy as np
import pandas as pd
from ..utils import mathutil, signal_process
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from ..core import Signal, ProbeGroup, Epoch


def _detect_freq_band_epochs(
    signals, freq_band, thresh, mindur, maxdur, mergedist, fs, sigma, ignore_times=None
):
    """Detects epochs of high power in a given frequency band

    Parameters
    ----------
    thresh : tuple, optional
        low and high threshold for detection
    mindur : float, optional
        minimum duration of epoch
    maxdur : float, optiona
    chans : list
        channels used for epoch detection, if None then chooses best chans
    """

    zscsignal = np.zeros(signals.shape)
    lf, hf = freq_band
    # dt = 1 / fs
    # sigma = sigma / dt
    lowthresh, highthresh = thresh
    for sig_i, sig in enumerate(signals):
        yf = signal_process.filter_sig.bandpass(sig, lf=lf, hf=hf, fs=fs)
        zsc_chan = stats.zscore(np.abs(signal_process.hilbertfast(yf)))
        zscsignal[sig_i] = zsc_chan

    # zscsignal = np.asarray(zscsignal)

    # ---------setting noisy periods zero --------
    if ignore_times is not None:
        assert ignore_times.ndim == 2, "ignore_times should be 2 dimensional array"
        noisy_frames = np.concatenate(
            [
                (np.arange(start * fs, stop * fs)).astype(int)
                for (start, stop) in ignore_times
            ]
        )

        zscsignal[:, noisy_frames] = 0

    # ------hilbert transform --> binarize by > then lowthreshold
    maxPower = np.max(zscsignal, axis=0)

    ThreshSignal = np.where(zscsignal > lowthresh, 1, 0).sum(axis=0)
    ThreshSignal = np.diff(np.where(ThreshSignal > 0, 1, 0))
    start = np.where(ThreshSignal == 1)[0]
    stop = np.where(ThreshSignal == -1)[0]

    # --- getting rid of incomplete epochs at begining or end ---------
    if start[0] > stop[0]:
        stop = stop[1:]
    if start[-1] > stop[-1]:
        start = start[:-1]

    firstPass = np.vstack((start, stop)).T
    print(f"{len(firstPass)} epochs detected initially")

    # --------merging close epochs------------
    min_inter_epoch_samples = mergedist * fs
    secondPass = []
    epoch = firstPass[0]
    for i in range(1, len(firstPass)):
        if firstPass[i, 0] - epoch[1] < min_inter_epoch_samples:
            epoch = [epoch[0], firstPass[i, 1]]
        else:
            secondPass.append(epoch)
            epoch = firstPass[i]
    secondPass.append(epoch)
    secondPass = np.asarray(secondPass)
    print(f"{len(secondPass)} epochs reamining after merging close ones")

    # ------delete epochs with less than threshold power--------
    thirdPass = []
    peakpower, peaktime = [], []

    for i in range(0, len(secondPass)):
        maxValue = max(maxPower[secondPass[i, 0] : secondPass[i, 1]])
        if maxValue > highthresh:
            thirdPass.append(secondPass[i])
            peakpower.append(maxValue)
            peaktime.append(
                secondPass[i, 0]
                + np.argmax(maxPower[secondPass[i, 0] : secondPass[i, 1]])
            )
    thirdPass = np.asarray(thirdPass)
    print(f"{len(thirdPass)} epochs reamining after deleting epochs with weaker power")

    ripple_duration = np.diff(thirdPass, axis=1) / fs
    epochs = pd.DataFrame(
        {
            "start": thirdPass[:, 0],
            "stop": thirdPass[:, 1],
            "peakpower": peakpower,
            "peaktime": np.asarray(peaktime),
            "duration": ripple_duration.squeeze(),
        }
    )

    # ---------delete very short epochs--------
    epochs = epochs[epochs.duration >= mindur]
    print(f"{len(epochs)} epochs reamining after deleting short epochs")

    # ----- delete epochs with unrealistic high power
    # artifactRipples = np.where(peakpower > maxPeakPower)[0]
    # fourthPass = np.delete(thirdPass, artifactRipples, 0)
    # peakpower = np.delete(peakpower, artifactRipples)

    # ---------delete very long epochs---------
    epochs = epochs[epochs.duration <= maxdur]
    print(f"{len(epochs)} epochs reamining after deleting very long epochs")

    # ----- converting to all time stamps to seconds --------
    epochs[["start", "stop", "peaktime"]] /= fs  # seconds

    epochs = epochs.reset_index(drop=True)
    epochs["label"] = ""
    metadata = {
        "params": {
            "lowThres": lowthresh,
            "highThresh": highthresh,
            "freq_band": freq_band,
            "mindur": mindur,
            "maxdur": maxdur,
            "mergedist": mergedist,
        },
    }

    return epochs, metadata


def detect_hpc_slow_wave_epochs(
    signal: Signal, freq_band=(0.5, 4), nrem_epochs: Epoch = None
):
    """Caculate delta events only

    chan --> filter delta --> identify peaks and troughs within sws epochs only --> identifies a slow wave as trough to peak --> thresholds for 100ms minimum duration

    Parameters
    ----------
    signal : Signal object
        signal trace to be used for detection
    freq_band : tuple, optional
        frequency band in Hz, by default (0.5, 4)
    """

    assert signal.n_channels == 1, "Signal should have only 1 channel"

    def _get_sw(sig_):
        # ---- filtering in delta band -----
        trace = sig_.traces[0]
        t = sig_.time
        lf, hf = freq_band
        delta = signal_process.filter_sig.bandpass(trace, lf=lf, hf=hf)

        # ---- normalize and flip the sign to be consistent with cortical lfp ----
        delta = -1 * stats.zscore(delta)

        # ---- finding peaks and trough for delta oscillations

        up = sg.find_peaks(delta)[0]
        down = sg.find_peaks(-delta)[0]

        if up[0] < down[0]:
            up = up[1:]
        if up[-1] > down[-1]:
            up = up[:-1]

        sigdelta = []
        for i in range(len(down) - 1):
            tbeg = t[down[i]]
            tpeak = t[up[i]]
            tend = t[down[i + 1]]
            peakamp = delta[up[i]]
            endamp = delta[down[i + 1]]
            # ------ thresholds for selecting delta --------
            # if (peakamp > 2 and endamp < 0) or (peakamp > 1 and endamp < -1.5):
            sigdelta.append([peakamp, endamp, tpeak, tbeg, tend])

        return np.asarray(sigdelta)

    if nrem_epochs is not None:
        sw = []
        for e in nrem_epochs.as_array():
            sw.append(_get_sw(signal.time_slice(t_start=e[0], t_stop=e[1])))
        sw = np.vstack(sw)
    else:
        sw = _get_sw(signal)

    print(f"{len(sw)} delta waves detected")

    epochs = pd.DataFrame(
        {
            "start": sw[:, 3],
            "stop": sw[:, 4],
            "peaktime": sw[:, 2],
            "peakamp": sw[:, 0],
            "endamp": sw[:, 1],
            "label": "sw",
        }
    )
    params = {"freq_band": freq_band, "channel": signal.channel_id}

    return Epoch(epochs=epochs, metadata=params)


def detect_ripple_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(150, 250),
    thresh=(1, 5),
    mindur=0.05,
    maxdur=0.450,
    mergedist=0.05,
    sigma=None,
    ignore_epochs: Epoch = None,
):
    # TODO chewing artifact frequency (>300 Hz) or emg based rejection of ripple epochs

    if probegroup is None:
        selected_chans = signal.channel_id
        traces = signal.traces

    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
            # if changrp:
        selected_chans = []
        for changrp in changrps:
            signal_slice = signal.time_slice(
                channel_id=changrp.astype("int"), t_start=0, t_stop=3600
            )
            hil_stat = signal_process.hilbert_ampltiude_stat(
                signal_slice.traces,
                freq_band=freq_band,
                fs=signal.sampling_rate,
                statistic="mean",
            )
            selected_chans.append(changrp[np.argmax(hil_stat)])

        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for ripples: {selected_chans}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs, metadata = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
    )
    epochs["start"] = epochs["start"] + signal.t_start
    epochs["stop"] = epochs["stop"] + signal.t_start

    metadata["channels"] = selected_chans
    return Epoch(epochs=epochs, metadata=metadata)


def detect_theta_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(5, 12),
    thresh=(0, 0.5),
    mindur=0.25,
    maxdur=5,
    mergedist=0.5,
    ignore_epochs: Epoch = None,
):

    if probegroup is None:
        selected_chan = signal.channel_id
        traces = signal.traces
    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
        channel_ids = np.concatenate(changrps).astype("int")

        duration = signal.duration
        t1, t2 = signal.t_start, signal.t_start + np.min([duration, 3600])
        signal_slice = signal.time_slice(channel_id=channel_ids, t_start=t1, t_stop=t2)
        hil_stat = signal_process.hilbert_ampltiude_stat(
            signal_slice.traces,
            freq_band=freq_band,
            fs=signal.sampling_rate,
            statistic="mean",
        )
        selected_chan = channel_ids[np.argmax(hil_stat)]

        traces = signal.time_slice(channel_id=selected_chan).traces.reshape(1, -1)

    print(f"Best channel for theta: {selected_chan}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs, metadata = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
    )
    epochs["start"] = epochs["start"] + signal.t_start
    epochs["stop"] = epochs["stop"] + signal.t_start

    metadata["channels"] = selected_chan
    return Epoch(epochs=epochs, metadata=metadata)


def detect_spindle_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(8, 16),
    thresh=(1, 5),
    mindur=0.35,
    maxdur=4,
    mergedist=0.05,
    ignore_epochs: Epoch = None,
    method="hilbert",
):

    if probegroup is None:
        selected_chans = signal.channel_id
        traces = signal.traces

    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
            # if changrp:
        selected_chans = []
        for changrp in changrps:
            signal_slice = signal.time_slice(
                channel_id=changrp.astype("int"), t_start=0, t_stop=3600
            )
            hil_stat = signal_process.hilbert_ampltiude_stat(
                signal_slice.traces,
                freq_band=freq_band,
                fs=signal.sampling_rate,
                statistic="mean",
            )
            selected_chans.append(changrp[np.argmax(hil_stat)])

        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for spindles: {selected_chans}")

    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs, metadata = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
    )
    epochs["start"] = epochs["start"] + signal.t_start
    epochs["stop"] = epochs["stop"] + signal.t_start

    metadata["channels"] = selected_chans
    return Epoch(epochs=epochs, metadata=metadata)


def detect_gamma_epochs():
    pass


class Gamma:
    """Events and analysis related to gamma oscillations"""

    def get_peak_intervals(
        self,
        lfp,
        band=(25, 50),
        lowthresh=0,
        highthresh=1,
        minDistance=300,
        minDuration=125,
    ):
        """Returns strong theta lfp. If it has multiple channels, then strong theta periods are calculated from that channel which has highest area under the curve in the theta frequency band. Parameters are applied on z-scored lfp.

        Parameters
        ----------
        lfp : array like, channels x time
            from which strong periods are concatenated and returned
        lowthresh : float, optional
            threshold above which it is considered strong, by default 0 which is mean of the selected channel
        highthresh : float, optional
            [description], by default 0.5
        minDistance : int, optional
            minimum gap between periods before they are merged, by default 300 samples
        minDuration : int, optional
            [description], by default 1250, which means theta period should atleast last for 1 second

        Returns
        -------
        2D array
            start and end frames where events exceeded the set thresholds
        """

        # ---- filtering --> zscore --> threshold --> strong gamma periods ----
        gammalfp = signal_process.filter_sig.bandpass(lfp, lf=band[0], hf=band[1])
        hil_gamma = signal_process.hilbertfast(gammalfp)
        gamma_amp = np.abs(hil_gamma)

        zsc_gamma = stats.zscore(gamma_amp)
        peakevents = mathutil.threshPeriods(
            zsc_gamma,
            lowthresh=lowthresh,
            highthresh=highthresh,
            minDistance=minDistance,
            minDuration=minDuration,
        )

        return peakevents

    def csd(self, period, refchan, chans, band=(25, 50), window=1250):
        """Calculating current source density using laplacian method

        Parameters
        ----------
        period : array
            period over which theta cycles are averaged
        refchan : int or array
            channel whose theta peak will be considered. If array then median of lfp across all channels will be chosen for peak detection
        chans : array
            channels for lfp data
        window : int, optional
            time window around theta peak in number of samples, by default 1250

        Returns:
        ----------
        csd : dataclass,
            a dataclass return from signal_process module
        """
        lfp_period = self._obj.geteeg(chans=chans, timeRange=period)
        lfp_period = signal_process.filter_sig.bandpass(
            lfp_period, lf=band[0], hf=band[1]
        )

        gamma_lfp = self._obj.geteeg(chans=refchan, timeRange=period)
        nChans = lfp_period.shape[0]
        # lfp_period, _, _ = self.getstrongTheta(lfp_period)

        # --- Selecting channel with strongest theta for calculating theta peak-----
        # chan_order = self._getAUC(lfp_period)
        # gamma_lfp = signal_process.filter_sig.bandpass(
        #     lfp_period[chan_order[0], :], lf=5, hf=12, ax=-1)
        gamma_lfp = signal_process.filter_sig.bandpass(
            gamma_lfp, lf=band[0], hf=band[1]
        )
        peak = sg.find_peaks(gamma_lfp)[0]
        # Ignoring first and last second of data
        peak = peak[np.where((peak > 1250) & (peak < len(gamma_lfp) - 1250))[0]]

        # ---- averaging around theta cycle ---------------
        avg_theta = np.zeros((nChans, window))
        for ind in peak:
            avg_theta = avg_theta + lfp_period[:, ind - window // 2 : ind + window // 2]
        avg_theta = avg_theta / len(peak)

        _, ycoord = self._obj.probemap.get(chans=chans)

        csd = signal_process.Csd(lfp=avg_theta, coords=ycoord, chan_label=chans)
        csd.classic()

        return csd
