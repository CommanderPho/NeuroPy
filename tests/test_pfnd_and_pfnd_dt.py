"""PfND / PfND_TimeDependent integration tests focused on Ratemap behavior.

These exercise the live construction / mutation / slicing / HDF call sites that
must remain compatible across a Ratemap attrs conversion.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
if str(root_project_folder) not in sys.path:
    sys.path.insert(0, str(root_project_folder))
if str(tests_folder) not in sys.path:
    sys.path.insert(0, str(tests_folder))

from helpers.synthetic_pf_fixtures import (  # noqa: E402
    build_synthetic_pfnd_1d,
    build_synthetic_pfnd_2d,
    build_synthetic_pfnd_time_dependent_2d,
    build_synthetic_position_1d,
    build_synthetic_spikes_df,
    build_session_epoch,
    build_synthetic_trajectory,
    ensure_legacy_numpy_aliases,
)

ensure_legacy_numpy_aliases()

from neuropy.core.ratemap import Ratemap  # noqa: E402
from neuropy.analyses.placefields import PfND  # noqa: E402
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent  # noqa: E402


class TestPfNDRatemapIntegration(unittest.TestCase):
    def setUp(self):
        self.pf2d = build_synthetic_pfnd_2d(seed=0)
        self.pf1d = build_synthetic_pfnd_1d(seed=0)

    def test_ratemap_created_with_expected_api(self):
        rm = self.pf2d.ratemap
        self.assertIsInstance(rm, Ratemap)
        self.assertGreater(rm.n_neurons, 0)
        self.assertEqual(rm.ndim, 2)
        self.assertEqual(rm.tuning_curves.shape[0], rm.n_neurons)
        self.assertIsNotNone(rm.unsmoothed_tuning_maps)
        self.assertIsNotNone(rm.spikes_maps)
        self.assertIsNotNone(rm.occupancy)
        self.assertTrue(np.all(np.isin(rm.neuron_ids, self.pf2d.included_neuron_IDs)) or np.all(np.isin(self.pf2d.included_neuron_IDs, rm.neuron_ids)))

    def test_1d_and_2d_ndim(self):
        self.assertEqual(self.pf1d.ratemap.ndim, 1)
        self.assertEqual(self.pf2d.ratemap.ndim, 2)
        self.assertEqual(self.pf1d.ratemap.tuning_curves.ndim, 2)
        self.assertEqual(self.pf2d.ratemap.tuning_curves.ndim, 3)

    def test_occupancy_setter_mutates_ratemap(self):
        original = deepcopy(self.pf2d.ratemap.occupancy)
        new_occ = original * 1.5
        self.pf2d.occupancy = new_occ
        np.testing.assert_allclose(self.pf2d.ratemap.occupancy, new_occ)
        self.pf2d.occupancy = original

    def test_neuron_extended_ids_setter_mutates_ratemap(self):
        original = list(self.pf2d.neuron_extended_ids)
        reversed_ids = original[::-1]
        self.pf2d.neuron_extended_ids = reversed_ids
        self.assertEqual([e.aclu for e in self.pf2d.ratemap.neuron_extended_ids], [e.aclu for e in reversed_ids])
        self.pf2d.neuron_extended_ids = original

    def test_get_by_id_slices_ratemap(self):
        neuron_ids = np.asarray(self.pf2d.included_neuron_IDs)
        subset = neuron_ids[: max(1, len(neuron_ids) // 2)]
        sliced = deepcopy(self.pf2d).get_by_id(subset)
        np.testing.assert_array_equal(np.asarray(sliced.included_neuron_IDs), subset)
        np.testing.assert_array_equal(np.asarray(sliced.ratemap.neuron_ids), subset)
        self.assertEqual(sliced.ratemap.n_neurons, len(subset))
        self.assertEqual(sliced.ratemap.tuning_curves.shape[0], len(subset))

    def test_ratemap_indexing_matches_get_by_id(self):
        rm = self.pf1d.ratemap
        ids = np.asarray(rm.neuron_ids)[:2]
        by_id = rm.get_by_id(ids)
        idxs = [list(rm.neuron_ids).index(i) for i in ids]
        by_idx = rm[idxs]
        np.testing.assert_array_equal(by_id.neuron_ids, by_idx.neuron_ids)
        np.testing.assert_allclose(by_id.tuning_curves, by_idx.tuning_curves)

    def test_normalized_and_peak_accessors_via_pf(self):
        rm = self.pf1d.ratemap
        pdf = rm.pdf_normalized_tuning_curves
        np.testing.assert_allclose(np.nansum(pdf, axis=1), np.ones(rm.n_neurons), rtol=1e-5)
        peaks = rm.tuning_curve_unsmoothed_peak_firing_rates
        self.assertEqual(peaks.shape, (rm.n_neurons,))

    def test_to_1d_maximum_projection_on_pf_ratemap(self):
        rm1d = self.pf2d.ratemap.to_1D_maximum_projection()
        self.assertEqual(rm1d.ndim, 1)
        self.assertEqual(rm1d.n_neurons, self.pf2d.ratemap.n_neurons)

    def test_ybin_mutation_call_site_pattern(self):
        # PendingNotebookCode mutates .ratemap.ybin in place
        rm = self.pf2d.ratemap
        original = deepcopy(rm.ybin)
        shifted = original + 1.0
        rm.ybin = shifted
        np.testing.assert_allclose(self.pf2d.ratemap.ybin, shifted)
        rm.ybin = original

    def test_pickle_pfnd_preserves_ratemap(self):
        blob = pickle.dumps(self.pf1d)
        restored = pickle.loads(blob)
        self.assertIsInstance(restored.ratemap, Ratemap)
        np.testing.assert_allclose(restored.ratemap.tuning_curves, self.pf1d.ratemap.tuning_curves)
        np.testing.assert_array_equal(restored.ratemap.neuron_ids, self.pf1d.ratemap.neuron_ids)

    def test_pfnd_to_hdf_includes_nested_ratemap(self):
        # PfND.to_hdf cannot serialize config.speed_thresh=None as an HDF attribute.
        # Use a numeric sentinel after construction so nested ratemap writing is still covered.
        self.pf1d.config.speed_thresh = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'pfnd_test.h5')
            self.pf1d.to_hdf(path, 'test_pfnd')
            import h5py
            with h5py.File(path, 'r') as f:
                self.assertIn('test_pfnd/ratemap', f)
                group = f['test_pfnd/ratemap']
                self.assertIn('tuning_curves', group)
                self.assertIn('neuron_ids', group)
                np.testing.assert_allclose(group['tuning_curves'][...], self.pf1d.ratemap.tuning_curves)


class TestPfNDFromConfigValues(unittest.TestCase):
    def test_from_config_values_builds_ratemap(self):
        t, x, y, speed = build_synthetic_trajectory(seed=1)
        spikes_df = build_synthetic_spikes_df(t, x, seed=1)
        pos = build_synthetic_position_1d(t, x, speed)
        epochs = build_session_epoch(t)
        pf = PfND.from_config_values(deepcopy(spikes_df), deepcopy(pos), epochs=epochs, frate_thresh=0.0, speed_thresh=None, grid_bin=(5,), grid_bin_bounds=((0.0, 100.0),), smooth=(1.0,))
        self.assertIsInstance(pf.ratemap, Ratemap)
        self.assertEqual(pf.ratemap.ndim, 1)
        self.assertGreater(pf.ratemap.n_neurons, 0)


class TestPfNDTimeDependentRatemap(unittest.TestCase):
    def setUp(self):
        self.pf_dt = build_synthetic_pfnd_time_dependent_2d(seed=0)

    def test_construction_and_earliest_times(self):
        self.assertIsInstance(self.pf_dt, PfND_TimeDependent)
        self.assertTrue(np.isfinite(self.pf_dt.earliest_valid_time))

    def test_ratemap_property_builds_ratemap_instance(self):
        self.pf_dt.reset()
        # Advance far enough that occupancy / spikes accumulate
        end_t = float(self.pf_dt.all_time_filtered_pos_df['t'].to_numpy()[-1])
        self.pf_dt.update(t=end_t, should_snapshot=False)
        rm = self.pf_dt.ratemap
        self.assertIsInstance(rm, Ratemap)
        self.assertEqual(rm.ndim, 2)
        self.assertGreater(rm.n_neurons, 0)
        self.assertEqual(rm.tuning_curves.shape[0], rm.n_neurons)
        np.testing.assert_array_equal(rm.xbin, self.pf_dt.xbin)
        np.testing.assert_array_equal(rm.ybin, self.pf_dt.ybin)

    def test_step_and_snapshot_preserve_ratemap_access(self):
        self.pf_dt.reset()
        self.pf_dt.step(num_seconds_to_advance=5.0, should_snapshot=True)
        self.assertGreaterEqual(len(self.pf_dt.historical_snapshots), 1)
        rm = self.pf_dt.ratemap
        self.assertIsInstance(rm, Ratemap)
        self.assertEqual(rm.n_neurons, len(rm.neuron_ids))

    def test_get_by_id_on_time_dependent(self):
        self.pf_dt.reset()
        end_t = float(self.pf_dt.all_time_filtered_pos_df['t'].to_numpy()[-1])
        self.pf_dt.update(t=end_t, should_snapshot=False)
        ids = np.asarray(self.pf_dt.included_neuron_IDs)[:2]
        sliced = deepcopy(self.pf_dt).get_by_id(ids)
        np.testing.assert_array_equal(np.asarray(sliced.included_neuron_IDs), ids)
        # ratemap property should still construct after slicing
        rm = sliced.ratemap
        self.assertIsInstance(rm, Ratemap)
        self.assertEqual(rm.n_neurons, len(ids))

    def test_pickle_time_dependent_after_update(self):
        self.pf_dt.reset()
        self.pf_dt.update(t=10.0, should_snapshot=True)
        blob = pickle.dumps(self.pf_dt)
        restored = pickle.loads(blob)
        self.assertIsInstance(restored, PfND_TimeDependent)
        self.assertGreaterEqual(len(restored.historical_snapshots), 1)
        rm = restored.ratemap
        self.assertIsInstance(rm, Ratemap)


class TestRatemapConstructionParityWithPfND(unittest.TestCase):
    """Ensure PfND's Ratemap(...) call shape remains the supported constructor form."""

    def test_manual_reconstruction_from_pf_arrays(self):
        pf = build_synthetic_pfnd_1d(seed=2)
        rm = pf.ratemap
        rebuilt = Ratemap(
            rm.tuning_curves,
            unsmoothed_tuning_maps=rm.unsmoothed_tuning_maps,
            spikes_maps=rm.spikes_maps,
            xbin=rm.xbin,
            ybin=rm.ybin,
            occupancy=rm.occupancy,
            neuron_ids=rm.neuron_ids,
            neuron_extended_ids=rm.neuron_extended_ids,
            metadata=rm.metadata,
        )
        np.testing.assert_allclose(rebuilt.tuning_curves, rm.tuning_curves)
        np.testing.assert_array_equal(rebuilt.neuron_ids, rm.neuron_ids)
        self.assertEqual(rebuilt.n_neurons, rm.n_neurons)
        self.assertEqual(rebuilt.ndim, rm.ndim)


if __name__ == '__main__':
    unittest.main()
