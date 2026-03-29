"""Synthetic tests for epochs_spkcount sliding windows (no neuropy_pf_testing.h5)."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from neuropy.analyses.decoders import epochs_spkcount


def _minimal_spikes_df() -> pd.DataFrame:
    return pd.DataFrame({
        'aclu': [1, 1, 2, 2, 3, 3, 3],
        't_rel_seconds': [0.02, 0.11, 0.03, 0.14, 0.07, 0.19, 0.31],
        'neuron_type': ['pyramidal'] * 7,
    })


def _brute_sliding_counts(spikes_df: pd.DataFrame, epoch_start: float, epoch_stop: float, W: float, H: float, included: np.ndarray) -> np.ndarray:
    tcol = spikes_df.spikes.time_variable_name
    dur = float(epoch_stop) - float(epoch_start)
    t0, t1 = float(epoch_start), float(epoch_stop)
    if dur <= 0:
        starts = np.array([t0], dtype=np.float64)
        ends = np.array([t0 + W], dtype=np.float64)
    elif dur < W:
        starts = np.array([t0], dtype=np.float64)
        ends = np.array([t1], dtype=np.float64)
    else:
        n_windows = int(np.floor((dur - W) / H)) + 1
        starts = t0 + np.arange(n_windows, dtype=np.float64) * H
        ends = starts + W
    out = np.zeros((len(included), len(starts)), dtype=np.int32)
    for ui, uid in enumerate(included):
        t = spikes_df.loc[spikes_df['aclu'] == uid, tcol].to_numpy(dtype=np.float64, copy=False)
        t.sort()
        out[ui] = np.searchsorted(t, ends, side='right') - np.searchsorted(t, starts, side='left')
    return out


class TestEpochsSpkcountSynthetic(unittest.TestCase):


    def test_slideby_none_matches_explicit_equal_hop(self):
        spikes_df = _minimal_spikes_df()
        epochs_df = pd.DataFrame({'start': [0.0], 'stop': [0.6]})
        included = np.array([1, 2, 3])
        a, ids_a, nb_a, _ = epochs_spkcount(spikes_df, epochs_df, bin_size=0.2, slideby=None, included_neuron_ids=included)
        b, ids_b, nb_b, _ = epochs_spkcount(spikes_df, epochs_df, bin_size=0.2, slideby=0.2, included_neuron_ids=included)
        self.assertTrue(np.array_equal(ids_a, ids_b))
        self.assertEqual(int(nb_a[0]), int(nb_b[0]))
        self.assertTrue(np.array_equal(a[0], b[0]))


    def test_sliding_matches_brute_force_multi_epoch(self):
        spikes_df = _minimal_spikes_df()
        epochs_df = pd.DataFrame({'start': [0.0, 0.5], 'stop': [0.45, 0.95]})
        W, H = 0.2, 0.05
        included = np.array([1, 2, 3])
        spk, _, nb, tbc = epochs_spkcount(spikes_df, epochs_df, bin_size=W, slideby=H, export_time_bins=True, included_neuron_ids=included, debug_careful_validate_shapes=True)
        self.assertEqual(len(spk), 2)
        for ei, row in enumerate(epochs_df.itertuples()):
            brute = _brute_sliding_counts(spikes_df, row.start, row.stop, W, H, included)
            self.assertTrue(np.array_equal(np.asarray(spk[ei]), brute), msg=f'epoch {ei}')
            self.assertEqual(int(nb[ei]), brute.shape[1])
            self.assertEqual(tbc[ei].num_bins, brute.shape[1])


    def test_non_sliding_equals_contiguous_bins(self):
        spikes_df = _minimal_spikes_df()
        epochs_df = pd.DataFrame({'start': [0.0], 'stop': [0.55]})
        W = 0.1
        spk_s, _, nb_s, _ = epochs_spkcount(spikes_df, epochs_df, bin_size=W, slideby=W, included_neuron_ids=np.array([1, 2, 3]))
        spk_n, _, nb_n, _ = epochs_spkcount(spikes_df, epochs_df, bin_size=W, slideby=None, included_neuron_ids=np.array([1, 2, 3]))
        self.assertEqual(int(nb_s[0]), int(nb_n[0]))
        self.assertTrue(np.array_equal(spk_s[0], spk_n[0]))


if __name__ == '__main__':
    unittest.main()
