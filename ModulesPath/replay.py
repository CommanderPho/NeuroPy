import numpy as np

from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import pandas as pd
from mathutil import parcorr_mult, getICA_Assembly
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from parsePath import Recinfo
from getSpikes import Spikes
from scipy.ndimage import gaussian_filter


class Replay:
    def __init__(self, basepath):
        self.expvar = ExplainedVariance(basepath)
        self.assemblyICA = CellAssemblyICA(basepath)
        self.corr = Correlation(basepath)


class ExplainedVariance:
    colors = {"ev": "#4a4a4a", "rev": "#05d69e"}  # colors of each curve

    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def compute(
        self, template, match, control, binSize=0.250, window=900, slideby=None
    ):
        """Calucate explained variance (EV) and reverse EV

        Parameters
        ----------
        template : list
            template period
        match : list
            match period whose similarity is calculated to template
        control : list
            control period, correlations in this period will be accounted for
        binSize : float,
            bin size within each window, defaults 0.250 seconds
        window : int,
            size of window in which ev is calculated, defaults 900 seconds
        slideby : int,
            calculate EV by sliding window, seconds

        References:
        1) Kudrimoti 1999
        2) Tastsuno et al. 2007
        """

        spikes = Spikes(self._obj)
        if slideby is None:
            slideby = window

        # ----- choosing cells ----------------
        spks = spikes.times
        stability = spikes.stability.info
        stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]
        print(f"Calculating EV for {len(stable_pyr)} stable cells")
        spks = [spks[_] for _ in stable_pyr]

        # ------- windowing the time periods ----------
        nbins_window = int(window / binSize)
        nbins_slide = int(slideby / binSize)

        # ---- function to calculate correlation in each window ---------
        def cal_corr(period, windowing=True):
            bin_period = np.arange(period[0], period[1], binSize)
            spkcnt = np.array([np.histogram(x, bins=bin_period)[0] for x in spks])

            if windowing:
                t = np.arange(period[0], period[1] - window, slideby) + window / 2
                nwindow = len(t)

                window_spkcnt = [
                    spkcnt[:, i : i + nbins_window]
                    for i in range(0, int(nwindow) * nbins_slide, nbins_slide)
                ]

                # if nwindow % 1 > 0.3:
                #     window_spkcnt.append(spkcnt[:, int(nwindow) * nbins_window :])
                #     t = np.append(t, t[-1] + round(nwindow % 1, 3) / 2)

                corr = [
                    np.corrcoef(window_spkcnt[x]) for x in range(len(window_spkcnt))
                ]

            else:
                corr = np.corrcoef(spkcnt)
                t = None

            return corr, t

        # ---- correlation for each time period -----------
        control_corr, self.t_control = cal_corr(period=control)
        template_corr, _ = cal_corr(period=template, windowing=False)
        match_corr, self.t_match = cal_corr(period=match)

        # ----- indices for cross shanks correlation -------
        shnkId = np.asarray(spikes.info.shank)
        shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)
        cross_shnks = np.nonzero(np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1)))

        # --- selecting only pairwise correlations from different shanks -------
        control_corr = [control_corr[x][cross_shnks] for x in range(len(control_corr))]
        template_corr = template_corr[cross_shnks]
        match_corr = [match_corr[x][cross_shnks] for x in range(len(match_corr))]

        parcorr_template_vs_match, rev_corr = parcorr_mult(
            [template_corr], match_corr, control_corr
        )

        ev_template_vs_match = parcorr_template_vs_match ** 2
        rev_corr = rev_corr ** 2

        self.ev = ev_template_vs_match
        self.rev = rev_corr
        self.npairs = template_corr.shape[0]

    def plot(self, ax=None, tstart=0, legend=True):

        ev_mean = np.mean(self.ev.squeeze(), axis=0)
        ev_std = np.std(self.ev.squeeze(), axis=0)
        rev_mean = np.mean(self.rev.squeeze(), axis=0)
        rev_std = np.std(self.rev.squeeze(), axis=0)

        if ax is None:
            plt.clf()
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        t = (self.t_match - tstart) / 3600  # converting to hour

        # ---- plot rev first ---------
        ax.fill_between(
            t,
            rev_mean - rev_std,
            rev_mean + rev_std,
            color=self.colors["rev"],
            zorder=1,
            alpha=0.5,
            label="REV",
        )
        ax.plot(t, rev_mean, color=self.colors["rev"], zorder=2)

        # ------- plot ev -------
        ax.fill_between(
            t,
            ev_mean - ev_std,
            ev_mean + ev_std,
            color=self.colors["ev"],
            zorder=3,
            alpha=0.5,
            label="EV",
        )
        ax.plot(t, ev_mean, self.colors["ev"], zorder=4)

        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Explained variance")
        if legend:
            ax.legend()

        return ax


class CellAssemblyICA:
    def __init__(self, basepath):

        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def getAssemblies(self, x):
        """extracting statisticaly independent components from significant eigenvectors as detected using Marcenko-Pasteur distributionvinput = Matrix  (m x n) where 'm' are the number of cells and 'n' time bins ICA weights thus extracted have highiest weight positive (as done in Gido M. van de Ven et al. 2016) V = ICA weights for each neuron in the coactivation (weight having the highiest value is kept positive) M1 =  originally extracted neuron weights

        Arguments:
            x {[ndarray]} -- [an array of size n * m]

        Returns:
            [type] -- [Independent assemblies]
        """

        zsc_x = stats.zscore(x, axis=1)

        # corrmat = (zsc_x @ zsc_x.T) / x.shape[1]
        corrmat = np.corrcoef(zsc_x)

        lambda_max = (1 + np.sqrt(1 / (x.shape[1] / x.shape[0]))) ** 2
        eig_val, eig_mat = np.linalg.eigh(corrmat)
        get_sigeigval = np.where(eig_val > lambda_max)[0]
        n_sigComp = len(get_sigeigval)
        pca_fit = PCA(n_components=n_sigComp, whiten=False).fit_transform(zsc_x)

        ica_decomp = FastICA(n_components=None, whiten=False).fit(pca_fit)
        W = ica_decomp.components_
        V = eig_mat[:, get_sigeigval] @ W.T

        # --- making highest absolute weight positive and then normalizing ----------
        max_weight = V[np.argmax(np.abs(V), axis=0), range(V.shape[1])]
        V[:, np.where(max_weight < 0)[0]] = (-1) * V[:, np.where(max_weight < 0)[0]]
        V /= np.sqrt(np.sum(V ** 2, axis=0))  # making sum of squares=1

        self.vectors = V
        return self.vectors

    def getActivation(self, template, match, spks=None, binsize=0.250):

        if spks is None:
            spks = self._obj.spikes.pyr

        template_bin = np.arange(template[0], template[1], binsize)
        template = np.asarray(
            [np.histogram(cell, bins=template_bin)[0] for cell in spks]
        )

        V = self.getAssemblies(template)

        match_bin = np.arange(match[0], match[1], binsize)
        match = np.asarray([np.histogram(cell, bins=match_bin)[0] for cell in spks])

        activation = []
        for i in range(V.shape[1]):
            projMat = np.outer(V[:, i], V[:, i])
            np.fill_diagonal(projMat, 0)
            activation.append(
                np.asarray(
                    [match[:, t] @ projMat @ match[:, t] for t in range(match.shape[1])]
                )
            )

        self.activation = np.asarray(activation)
        self.match_bin = match_bin

        return self.activation, self.match_bin

    def plotActivation(self, ax=None):
        activation = self.activation
        vectors = self.vectors
        nvec = activation.shape[0]
        t = self.match_bin[1:]

        if ax is None:
            plt.clf()
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(nvec, 6, figure=fig)
            fig.subplots_adjust(hspace=0.3)

        else:
            gs = gridspec.GridSpecFromSubplotSpec(7, 6, ax, wspace=0.1)

        for vec in range(nvec):
            axact = plt.subplot(gs[vec, 3:])
            axact.plot(t / 3600, activation[vec, :])

            axvec = plt.subplot(gs[vec, :2])
            axvec.stem(vectors[:, vec], markerfmt="C2o")
            if vec == nvec - 1:
                axact.set_xlabel("Time")
                axact.set_ylabel("Activation \n strength")

                axvec.set_xlabel("Neurons")
                axvec.set_ylabel("Weight")

            else:
                axact.set_xticks([])
                axact.set_xticklabels([])

                axvec.set_xticks([])
                axvec.set_xticklabels([])
                axvec.spines["bottom"].set_visible(False)


class Correlation:
    """Class for calculating pairwise correlations

    Attributes
    ----------
    corr : matrix
        correlation between time windows across a period of time
    time : array
        time points
    """

    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)

    def across_time_window(self, period, window=300, binsize=0.25, **kwargs):
        """Correlation of pairwise correlation across a period by dividing into window size epochs

        Parameters
        ----------
        period: array like
            time period where the pairwise correlations are calculated, in seconds
        window : int, optional
            dividing the period into this size window, by default 900
        binsize : float, optional
            [description], by default 0.25
        """

        spikes = Spikes(self._obj)

        # ----- choosing cells ----------------
        spks = spikes.times
        stability = spikes.stability.info
        stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]
        print(f"Calculating EV for {len(stable_pyr)} stable cells")
        spks = [spks[_] for _ in stable_pyr]

        epochs = np.arange(period[0], period[1], window)

        pair_corr_epoch = []
        for i in range(len(epochs) - 1):
            epoch_bins = np.arange(epochs[i], epochs[i + 1], binsize)
            spkcnt = np.asarray([np.histogram(x, bins=epoch_bins)[0] for x in spks])
            epoch_corr = np.corrcoef(spkcnt)
            pair_corr_epoch.append(epoch_corr[np.tril_indices_from(epoch_corr, k=-1)])
        pair_corr_epoch = np.asarray(pair_corr_epoch)

        # masking nan values in the array
        pair_corr_epoch = np.ma.array(pair_corr_epoch, mask=np.isnan(pair_corr_epoch))
        self.corr = np.ma.corrcoef(pair_corr_epoch)  # correlation across windows
        self.time = epochs[:-1] + window / 2

    def pairwise(self, spikes, period, binsize=0.25):
        """Calculates pairwise correlation between given spikes within given period

        Parameters
        ----------
        spikes : list
            list of spike times
        period : list
            time period within which it is calculated , in seconds
        binsize : float, optional
            binning of the time period, by default 0.25 seconds

        Returns
        -------
        N-pairs
            pairwise correlations
        """
        bins = np.arange(period[0], period[1], binsize)
        spk_cnts = np.asarray([np.histogram(cell, bins=bins)[0] for cell in spikes])
        corr = np.corrcoef(spk_cnts)
        return corr[np.tril_indices_from(corr, k=-1)]

    def plot_across_time(self, ax=None, tstart=0, smooth=None, cmap="Spectral_r"):
        """Plots heatmap of correlation matrix calculated in self.across_time_window()

        Parameters
        ----------
        ax : [type], optional
            axis to plot into, by default None
        tstart : int, optional
            if you want to start the time axis at some other time points, by default 0
        cmap : str, optional
            colormap used for heatmap, by default "Spectral_r"
        """
        
        corr_mat = self.corr.copy()
        np.fill_diagonal(corr_mat, 0)

        if smooth is not None:
            corr_mat = gaussian_filter(corr_mat, sigma=smooth)

        if ax is None:
            _, ax = plt.subplots()

        ax.pcolormesh(self.time - tstart, self.time - tstart, corr_mat, cmap=cmap)
        ax.set_xlabel("Time")
        ax.set_ylabel("Time")
