from types import SimpleNamespace

import numpy as np
import pandas as pd

from neuropy.core.epoch import Epoch
from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
from neuropy.core.clusterless_spike_events import ClusterlessSpikeEvents
from neuropy.core.neuron_identities import NeuronType
from neuropy.core.neurons import Neurons
from neuropy.core.session.SessionSelectionAndFiltering import batch_filter_session


def _build_source_neurons() -> Neurons:
    spiketrains = np.array([np.array([0.1, 0.4, 1.2]), np.array([0.2, 1.1]), np.array([0.3, 0.9])], dtype=object)
    waveforms = np.array([[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3], [3.0, 3.1, 3.2, 3.3]])
    peak_channels = np.array([10, 20, 30])
    shank_ids = np.array([0, 1, 1])
    extended_neuron_properties_df = pd.DataFrame({"aclu": [1, 2, 3], "si_unit_id": [101, 102, 103], "prediction": ["sua", "mua", "sua"]})
    return Neurons(spiketrains, t_stop=2.0, sampling_rate=30000, neuron_ids=np.array([1, 2, 3]),
        neuron_type=np.array([NeuronType.PYRAMIDAL, NeuronType.INTERNEURONS, NeuronType.PYRAMIDAL]),
        waveforms=waveforms, peak_channels=peak_channels, shank_ids=shank_ids, extended_neuron_properties_df=extended_neuron_properties_df)


def _build_spikes_df() -> pd.DataFrame:
    return pd.DataFrame({"t_seconds": [0.1, 0.4, 1.2, 0.2, 1.1, 0.3, 0.9], "t": [0.1, 0.4, 1.2, 0.2, 1.1, 0.3, 0.9], "aclu": [1, 1, 1, 2, 2, 3, 3],
        "neuron_type": [NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL, NeuronType.INTERNEURONS, NeuronType.INTERNEURONS, NeuronType.PYRAMIDAL, NeuronType.PYRAMIDAL],
        "shank": [0, 0, 0, 1, 1, 1, 1], "qclu": [1, 1, 1, 5, 5, 1, 1], "cluster": [1, 1, 1, 2, 2, 3, 3]})


def test_from_dataframe_with_source_neurons_preserves_waveform_metadata():
    source_neurons = _build_source_neurons()
    spikes_df = _build_spikes_df()
    filtered_spikes_df = spikes_df[spikes_df["aclu"].isin([1, 3])].copy()

    out_neurons = Neurons.from_dataframe(filtered_spikes_df, dat_sampling_rate=30000, time_variable_name="t_seconds", source_neurons=source_neurons)

    assert out_neurons.n_neurons == 2
    assert out_neurons.neuron_ids is not None
    assert list(out_neurons.neuron_ids) == [1, 3]
    assert out_neurons.waveforms is not None
    assert source_neurons.waveforms is not None
    np.testing.assert_array_equal(out_neurons.waveforms, source_neurons.waveforms[[0, 2]])
    np.testing.assert_array_equal(out_neurons.peak_channels, np.array([10, 30]))
    assert out_neurons._extended_neuron_properties_df is not None
    assert out_neurons._extended_neuron_properties_df["aclu"].to_list() == [1, 3]


def test_from_dataframe_without_source_neurons_keeps_existing_behavior():
    spikes_df = _build_spikes_df()
    filtered_spikes_df = spikes_df[spikes_df["aclu"].isin([1, 3])].copy()

    out_neurons = Neurons.from_dataframe(filtered_spikes_df, dat_sampling_rate=30000, time_variable_name="t_seconds")

    assert out_neurons.n_neurons == 2
    assert out_neurons.waveforms is None
    assert out_neurons.peak_channels is None
    assert out_neurons._extended_neuron_properties_df is None


def test_getitem_preserves_extended_neuron_properties_df():
    source_neurons = _build_source_neurons()

    out_neurons = source_neurons.get_by_id([1, 3])

    assert out_neurons.n_neurons == 2
    assert out_neurons.waveforms is not None
    assert source_neurons.waveforms is not None
    np.testing.assert_array_equal(out_neurons.waveforms, source_neurons.waveforms[[0, 2]])
    np.testing.assert_array_equal(out_neurons.peak_channels, np.array([10, 30]))
    assert out_neurons._extended_neuron_properties_df is not None
    assert out_neurons._extended_neuron_properties_df["aclu"].to_list() == [1, 3]


class _MinimalPosition:
    metadata = {}

    def __init__(self):
        self._df = pd.DataFrame({"t": [0.0, 0.5, 1.0, 1.5], "x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0, 0.0]})

    def compute_higher_order_derivatives(self):
        return None

    def compute_smoothed_position_info(self):
        return self._df

    def to_dataframe(self):
        return self._df.copy()


def test_batch_filter_session_preserves_source_neuron_waveforms():
    source_neurons = _build_source_neurons()
    spikes_df = _build_spikes_df()
    epochs = Epoch.from_starts_stops_arrays(starts=np.array([0.0]), stops=np.array([1.0]), labels=np.array(["maze"]))
    sess = SimpleNamespace(config=None, filePrefix=None, recinfo=SimpleNamespace(dat_sampling_rate=30000), eegfile=None, datfile=None,
        neurons=source_neurons, probegroup=None, position=_MinimalPosition(), ripple=None, mua=None, laps=None,
        flattened_spiketrains=FlattenedSpiketrains(spikes_df, time_variable_name="t_seconds"), pbe=None)

    filtered_sess = batch_filter_session(sess, sess.position, spikes_df, epochs)

    assert filtered_sess.neurons.neuron_ids is not None
    assert list(filtered_sess.neurons.neuron_ids) == [1, 3]
    assert filtered_sess.neurons.waveforms is not None
    assert filtered_sess.neurons.waveforms.shape[0] == filtered_sess.neurons.n_neurons
    assert source_neurons.waveforms is not None
    np.testing.assert_array_equal(filtered_sess.neurons.waveforms, source_neurons.waveforms[[0, 2]])


def test_batch_filter_session_filters_clusterless_spike_events():
    source_neurons = _build_source_neurons()
    spikes_df = _build_spikes_df()
    epochs = Epoch.from_starts_stops_arrays(starts=np.array([0.0]), stops=np.array([1.0]), labels=np.array(["maze"]))
    clusterless_spike_events = ClusterlessSpikeEvents(spike_times_sec=np.array([0.3, 0.9, 1.2, 1.5], dtype=np.float32), electrode_indices=np.array([0, 1, 1, 2], dtype=np.int16),
        marks=np.ones((4, 4), dtype=np.float32), sampling_frequency_hz=1000.0, electrode_mode="channel", t_start=0.0, t_stop=2.0)
    sess = SimpleNamespace(config=None, filePrefix=None, recinfo=SimpleNamespace(dat_sampling_rate=30000), eegfile=None, datfile=None,
        neurons=source_neurons, probegroup=None, position=_MinimalPosition(), ripple=None, mua=None, laps=None,
        flattened_spiketrains=FlattenedSpiketrains(spikes_df, time_variable_name="t_seconds"), pbe=None, clusterless_spike_events=clusterless_spike_events)

    filtered_sess = batch_filter_session(sess, sess.position, spikes_df, epochs)

    assert filtered_sess.clusterless_spike_events is not None
    np.testing.assert_allclose(filtered_sess.clusterless_spike_events.spike_times_sec, np.array([0.3, 0.9], dtype=np.float32))
    assert filtered_sess.clusterless_spike_events.t_start == 0.0
    assert filtered_sess.clusterless_spike_events.t_stop == 1.0
