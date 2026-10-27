import unittest
import numpy as np
import pandas as pd

from neuropy.core.epoch import Epoch
from neuropy.core.flattened_spiketrains import SpikesAccessor  # registers the `spikes` dataframe accessor
from neuropy.utils.efficient_interval_search import filter_epochs_by_num_active_units, trim_epochs_to_first_last_spikes


def _make_spikes_df(times, aclus) -> pd.DataFrame:
    times = np.asarray(times, dtype=float)
    aclus = np.asarray(aclus)
    return pd.DataFrame({
        't_rel_seconds': times,
        't': times,
        'aclu': aclus,
        'neuron_type': 'pyr',
        'flat_spike_idx': np.arange(len(times)),
    })


def _make_epochs(starts, stops, labels=None) -> Epoch:
    if labels is None:
        labels = [str(i) for i in range(len(starts))]
    return Epoch(pd.DataFrame({'start': starts, 'stop': stops, 'label': labels}))


class TestTrimEpochsToFirstLastSpikes(unittest.TestCase):
    def test_trim_keeps_valid_epochs_aligned(self):
        spikes_df = _make_spikes_df(times=[1.1, 1.5, 3.2, 3.8], aclus=[1, 2, 1, 2])
        epochs = _make_epochs([1.0, 3.0], [2.0, 4.0])

        trimmed, spike_dfs = trim_epochs_to_first_last_spikes(spikes_df, epochs)

        self.assertEqual(trimmed.n_epochs, 2)
        self.assertEqual(len(spike_dfs), 2)
        np.testing.assert_allclose(trimmed.starts, [1.1, 3.2])
        np.testing.assert_allclose(trimmed.stops, [1.5, 3.8])

    def test_trim_drops_zero_duration_coincident_spikes_without_assert(self):
        """Regression: two spikes at the same timestamp trim to duration 0.

        `filtered_by_duration` used to drop that epoch while leaving `epoch_split_spike_dfs`
        unchanged, which failed the count assert and broke `long_short_fr_indicies_analyses`.
        """
        spikes_df = _make_spikes_df(times=[1.1, 1.5, 3.5, 3.5], aclus=[1, 2, 1, 2])
        epochs = _make_epochs([1.0, 3.0], [2.0, 4.0])

        trimmed, spike_dfs = trim_epochs_to_first_last_spikes(spikes_df, epochs)

        self.assertEqual(trimmed.n_epochs, 1)
        self.assertEqual(len(spike_dfs), 1)
        self.assertEqual(trimmed.n_epochs, len(spike_dfs))
        np.testing.assert_allclose(trimmed.starts, [1.1])
        np.testing.assert_allclose(trimmed.stops, [1.5])

    def test_trim_all_zero_duration_returns_empty_aligned_outputs(self):
        spikes_df = _make_spikes_df(times=[1.5, 1.5, 3.5, 3.5], aclus=[1, 2, 1, 2])
        epochs = _make_epochs([1.0, 3.0], [2.0, 4.0])

        trimmed, spike_dfs = trim_epochs_to_first_last_spikes(spikes_df, epochs)

        self.assertEqual(trimmed.n_epochs, 0)
        self.assertEqual(len(spike_dfs), 0)

    def test_filter_epochs_by_num_active_units_survives_zero_duration_trim(self):
        spikes_df = _make_spikes_df(times=[1.1, 1.5, 3.5, 3.5], aclus=[1, 2, 1, 2])
        epochs = _make_epochs([1.0, 3.0], [2.0, 4.0])

        filtered_epochs, extra_outputs = filter_epochs_by_num_active_units(spikes_df, epochs, min_inclusion_fr_active_thresh=0.0, min_num_unique_aclu_inclusions=1, include_intermediate_computations=False)

        self.assertGreaterEqual(filtered_epochs.n_epochs, 1)
        self.assertIsNone(extra_outputs)


if __name__ == '__main__':
    unittest.main()
