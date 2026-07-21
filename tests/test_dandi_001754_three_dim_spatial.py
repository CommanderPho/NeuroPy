from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_dandi_001754_load_position_from_nwb_rejects_nx2(monkeypatch):
    from neuropy.core.session.Formats.Specific.DANDI001754NWBDataSessionFormat import DANDI001754NWBDataSessionFormatRegisteredClass

    timestamps = np.linspace(0.0, 10.0, 100)
    spatial_series = SimpleNamespace(data=np.column_stack([np.arange(100.0), np.arange(100.0) * 0.5]))
    nwbf = SimpleNamespace(identifier='test_nwb.nwb')
    monkeypatch.setattr(DANDI001754NWBDataSessionFormatRegisteredClass, '_get_position_spatial_series', classmethod(lambda cls, _nwbf: spatial_series))

    with pytest.raises(ValueError, match="require N×3 spatial_series"):
        DANDI001754NWBDataSessionFormatRegisteredClass._load_position_from_nwb(nwbf, timestamps=timestamps, t0=0.0)


def test_dandi_001754_load_position_from_nwb_accepts_nx3(monkeypatch):
    from neuropy.core.session.Formats.Specific.DANDI001754NWBDataSessionFormat import DANDI001754NWBDataSessionFormatRegisteredClass

    timestamps = np.linspace(0.0, 10.0, 100)
    spatial_series = SimpleNamespace(data=np.column_stack([np.arange(100.0), np.arange(100.0) * 0.5, np.arange(100.0) * 0.25]))
    nwbf = SimpleNamespace(identifier='test_nwb.nwb')
    monkeypatch.setattr(DANDI001754NWBDataSessionFormatRegisteredClass, '_get_position_spatial_series', classmethod(lambda cls, _nwbf: spatial_series))

    position = DANDI001754NWBDataSessionFormatRegisteredClass._load_position_from_nwb(nwbf, timestamps=timestamps, t0=0.0)

    assert position.ndim == 3
    assert len(position.z) == 100
    np.testing.assert_allclose(position.z, np.arange(100.0) * 0.25)


def test_pfnd_3d_smoke_compute():
    from neuropy.analyses.placefields import PfND
    from neuropy.core import Epoch, Position
    from neuropy.core.flattened_spiketrains import FlattenedSpiketrains

    rng = np.random.default_rng(0)
    n_pos = 500
    t = np.linspace(0.0, 50.0, n_pos)
    x = rng.uniform(0.0, 100.0, n_pos)
    y = rng.uniform(0.0, 100.0, n_pos)
    z = rng.uniform(0.0, 100.0, n_pos)
    position = Position.from_separate_arrays(t, x, y, z=z)
    epochs = Epoch(pd.DataFrame({'start': [0.0], 'stop': [50.0], 'label': ['all']}))

    spike_times = rng.uniform(0.0, 50.0, 200)
    spikes_df = pd.DataFrame({'t_rel_seconds': spike_times, 'aclu': np.ones(len(spike_times), dtype=int), 'neuron_type': ['pyr'] * len(spike_times)})
    spikes_df = FlattenedSpiketrains.interpolate_spike_positions(spikes_df, position.time, position.x, position.y, position_speeds=position.speed, spike_timestamp_column_name='t_rel_seconds', z=position.z)

    grid_bin_bounds = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0))
    pf3d = PfND.from_config_values(spikes_df=spikes_df, position=position, epochs=epochs, speed_thresh=1000.0, frate_thresh=0.0, grid_bin=(10.0, 10.0, 10.0), grid_bin_bounds=grid_bin_bounds, smooth=(0.0, 0.0, 0.0))

    assert pf3d.ndim == 3
    assert pf3d.ratemap.occupancy.ndim == 3
    assert pf3d.ratemap.tuning_curves.ndim == 4
    assert pf3d.ratemap.tuning_curves.shape[1:] == pf3d.ratemap.occupancy.shape
