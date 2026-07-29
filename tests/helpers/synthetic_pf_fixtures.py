"""Shared synthetic fixtures for Ratemap / PfND / PfND_TimeDependent unit tests.

These avoid the DVC-backed `neuropy_pf_testing.h5` so the pre-attrs-refactor
regression suite can run without external data.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def ensure_legacy_numpy_aliases() -> None:
    """Restore removed NumPy scalar aliases used by some Neuropy modules."""
    if not hasattr(np, 'float'):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, 'int'):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, 'complex'):
        np.complex = complex  # type: ignore[attr-defined]
    if not hasattr(np, 'bool'):
        np.bool = bool  # type: ignore[attr-defined]


ensure_legacy_numpy_aliases()

from neuropy.core.epoch import Epoch
from neuropy.core.neuron_identities import NeuronExtendedIdentity, NeuronType
from neuropy.core.position import Position
from neuropy.core.ratemap import Ratemap
from neuropy.analyses.placefields import PfND, PlacefieldComputationParameters
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent


def build_synthetic_trajectory(duration_s: float = 30.0, fs: float = 30.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, x, y, speed) for a back-and-forth 2D trajectory."""
    _ = seed  # reserved for future stochastic trajectories
    t = np.arange(0.0, duration_s, 1.0 / fs)
    x = 50.0 + 40.0 * np.sin(2.0 * np.pi * t / 10.0)
    y = 100.0 + 5.0 * np.cos(2.0 * np.pi * t / 8.0)
    dx = np.diff(x, prepend=x[0])
    speed = np.abs(dx) * fs + 5.0
    return t, x, y, speed


def build_synthetic_spikes_df(t: np.ndarray, x: np.ndarray, n_neurons: int = 4, seed: int = 0) -> pd.DataFrame:
    """Build a minimal spikes dataframe compatible with SpikesAccessor / PfND."""
    rng = np.random.default_rng(seed)
    aclus = np.array([2, 5, 7, 9][:n_neurons], dtype=int)
    rows = []
    for i, aclu in enumerate(aclus):
        preferred_x = 30.0 + i * 15.0
        for tt, xx in zip(t, x):
            p = np.exp(-0.5 * ((xx - preferred_x) / 10.0) ** 2)
            if rng.random() < 0.25 * p:
                rows.append(dict(
                    t_rel_seconds=float(tt),
                    aclu=int(aclu),
                    shank=1,
                    cluster=int(i + 1),
                    qclu=1,
                    neuron_type=NeuronType.PYRAMIDAL,
                    flat_spike_idx=len(rows),
                ))
        ## END for tt, xx in zip(t, x)...
    ## END for i, aclu in enumerate(aclus)...

    spikes_df = pd.DataFrame(rows).sort_values('t_rel_seconds').reset_index(drop=True)
    spikes_df['flat_spike_idx'] = np.arange(len(spikes_df), dtype=int)
    return spikes_df


def build_synthetic_position_2d(t: np.ndarray, x: np.ndarray, y: np.ndarray, speed: np.ndarray) -> Position:
    return Position(pd.DataFrame({'t': t, 'x': x, 'y': y, 'speed': speed}))


def build_synthetic_position_1d(t: np.ndarray, x: np.ndarray, speed: np.ndarray) -> Position:
    return Position(pd.DataFrame({'t': t, 'x': x, 'speed': speed}))


def build_session_epoch(t: np.ndarray) -> Epoch:
    return Epoch(pd.DataFrame({'start': [float(t[0])], 'stop': [float(t[-1]) + 1e-3], 'label': ['sess']}))


def build_pf_config_2d(**overrides) -> PlacefieldComputationParameters:
    """Default 2D config. speed_thresh=None skips modern speed filtering (keeps occupancy intact)."""
    defaults = dict(speed_thresh=None, grid_bin=(5, 5), grid_bin_bounds=((0.0, 100.0), (90.0, 110.0)), smooth=(1.0, 1.0), frate_thresh=0.0)
    defaults.update(overrides)
    return PlacefieldComputationParameters(**defaults)


def build_pf_config_1d(**overrides) -> PlacefieldComputationParameters:
    defaults = dict(speed_thresh=None, grid_bin=(5,), grid_bin_bounds=((0.0, 100.0),), smooth=(1.0,), frate_thresh=0.0)
    defaults.update(overrides)
    return PlacefieldComputationParameters(**defaults)


def build_synthetic_pfnd_2d(seed: int = 0, **config_overrides) -> PfND:
    t, x, y, speed = build_synthetic_trajectory(seed=seed)
    spikes_df = build_synthetic_spikes_df(t, x, seed=seed)
    pos = build_synthetic_position_2d(t, x, y, speed)
    epochs = build_session_epoch(t)
    config = build_pf_config_2d(**config_overrides)
    return PfND(deepcopy(spikes_df), deepcopy(pos), epochs, config=config, position_srate=pos.sampling_rate)


def build_synthetic_pfnd_1d(seed: int = 0, **config_overrides) -> PfND:
    t, x, y, speed = build_synthetic_trajectory(seed=seed)
    spikes_df = build_synthetic_spikes_df(t, x, seed=seed)
    pos = build_synthetic_position_1d(t, x, speed)
    epochs = build_session_epoch(t)
    config = build_pf_config_1d(**config_overrides)
    return PfND(deepcopy(spikes_df), deepcopy(pos), epochs, config=config, position_srate=pos.sampling_rate)


def build_synthetic_pfnd_time_dependent_2d(seed: int = 0, **config_overrides) -> PfND_TimeDependent:
    t, x, y, speed = build_synthetic_trajectory(seed=seed)
    spikes_df = build_synthetic_spikes_df(t, x, seed=seed)
    pos = build_synthetic_position_2d(t, x, y, speed)
    epochs = build_session_epoch(t)
    cfg = build_pf_config_2d(**config_overrides)
    return PfND_TimeDependent.from_config_values(
        deepcopy(spikes_df), deepcopy(pos), epochs=epochs,
        frate_thresh=cfg.frate_thresh, speed_thresh=cfg.speed_thresh,
        grid_bin=cfg.grid_bin, grid_bin_bounds=cfg.grid_bin_bounds, smooth=cfg.smooth,
    )


def build_synthetic_neuron_extended_ids(neuron_ids) -> list:
    return [NeuronExtendedIdentity(shank=1, cluster=int(i + 1), aclu=int(aclu), qclu=1) for i, aclu in enumerate(neuron_ids)]


def build_synthetic_ratemap_1d(n_neurons: int = 4, n_xbins: int = 20, seed: int = 0, include_unsmoothed: bool = True, include_extended_ids: bool = True, metadata: Optional[dict] = None) -> Ratemap:
    """Construct a fully-populated 1D Ratemap with gaussian-like tuning curves."""
    rng = np.random.default_rng(seed)
    xbin = np.linspace(0.0, 100.0, n_xbins + 1)
    xcenters = 0.5 * (xbin[:-1] + xbin[1:])
    neuron_ids = np.array([2, 5, 7, 9][:n_neurons], dtype=int)
    tuning_curves = np.zeros((n_neurons, n_xbins), dtype=float)
    spikes_maps = np.zeros((n_neurons, n_xbins), dtype=float)
    for i in range(n_neurons):
        mu = xcenters[int((i + 0.5) * n_xbins / n_neurons)]
        curve = np.exp(-0.5 * ((xcenters - mu) / 8.0) ** 2) * (2.0 + i)
        tuning_curves[i] = curve
        spikes_maps[i] = np.maximum(rng.poisson(curve * 3.0), 0)
    ## END for i in range(n_neurons)...

    unsmoothed = tuning_curves * 1.1 if include_unsmoothed else None
    occupancy = np.maximum(rng.uniform(0.5, 2.0, size=n_xbins), 0.1)
    occupancy[0] = 0.0  # leave one never-visited bin for occupancy-mask tests
    extended_ids = build_synthetic_neuron_extended_ids(neuron_ids) if include_extended_ids else None
    return Ratemap(
        tuning_curves,
        unsmoothed_tuning_maps=unsmoothed,
        spikes_maps=spikes_maps,
        xbin=xbin,
        ybin=None,
        occupancy=occupancy,
        neuron_ids=neuron_ids,
        neuron_extended_ids=extended_ids,
        metadata=metadata,
    )


def build_synthetic_ratemap_2d(n_neurons: int = 3, n_xbins: int = 12, n_ybins: int = 8, seed: int = 1, include_unsmoothed: bool = True) -> Ratemap:
    """Construct a fully-populated 2D Ratemap."""
    rng = np.random.default_rng(seed)
    xbin = np.linspace(0.0, 100.0, n_xbins + 1)
    ybin = np.linspace(90.0, 110.0, n_ybins + 1)
    xcenters = 0.5 * (xbin[:-1] + xbin[1:])
    ycenters = 0.5 * (ybin[:-1] + ybin[1:])
    xx, yy = np.meshgrid(xcenters, ycenters, indexing='ij')
    neuron_ids = np.array([2, 5, 7][:n_neurons], dtype=int)
    tuning_curves = np.zeros((n_neurons, n_xbins, n_ybins), dtype=float)
    spikes_maps = np.zeros((n_neurons, n_xbins, n_ybins), dtype=float)
    for i in range(n_neurons):
        mu_x = xcenters[int((i + 0.5) * n_xbins / n_neurons)]
        mu_y = ycenters[n_ybins // 2]
        curve = np.exp(-0.5 * (((xx - mu_x) / 10.0) ** 2 + ((yy - mu_y) / 4.0) ** 2)) * (1.5 + i)
        tuning_curves[i] = curve
        spikes_maps[i] = np.maximum(rng.poisson(curve * 2.0), 0)
    ## END for i in range(n_neurons)...

    unsmoothed = tuning_curves * 1.05 if include_unsmoothed else None
    occupancy = np.maximum(rng.uniform(0.2, 1.5, size=(n_xbins, n_ybins)), 0.05)
    occupancy[0, 0] = 0.0
    return Ratemap(
        tuning_curves,
        unsmoothed_tuning_maps=unsmoothed,
        spikes_maps=spikes_maps,
        xbin=xbin,
        ybin=ybin,
        occupancy=occupancy,
        neuron_ids=neuron_ids,
        neuron_extended_ids=build_synthetic_neuron_extended_ids(neuron_ids),
        metadata={'source': 'synthetic_2d'},
    )
