"""Comprehensive Ratemap unit tests (pre-attrs-refactor baseline).

These lock constructor signatures, public attribute/property APIs, slicing,
normalization helpers, HDF layout, pickle round-trips, and classmethods used
by PfND / call sites. Run before and after converting Ratemap to attrs.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
if str(root_project_folder) not in sys.path:
    sys.path.insert(0, str(root_project_folder))

# Ensure helpers package importable when running as `python -m unittest tests.test_ratemap`
if str(tests_folder) not in sys.path:
    sys.path.insert(0, str(tests_folder))

from helpers.synthetic_pf_fixtures import (  # noqa: E402
    build_synthetic_ratemap_1d,
    build_synthetic_ratemap_2d,
    ensure_legacy_numpy_aliases,
)

ensure_legacy_numpy_aliases()

from neuropy.core.ratemap import Ratemap  # noqa: E402


class TestRatemapConstruction(unittest.TestCase):
    def test_positional_tuning_curves_constructor(self):
        tuning = np.ones((3, 10), dtype=float)
        rm = Ratemap(tuning, spikes_maps=np.ones((3, 10)), xbin=np.linspace(0, 10, 11), occupancy=np.ones(10), neuron_ids=np.array([1, 2, 3]))
        self.assertEqual(rm.n_neurons, 3)
        self.assertEqual(rm.ndim, 1)
        np.testing.assert_array_equal(rm.neuron_ids, [1, 2, 3])

    def test_keyword_constructor_matches_call_sites(self):
        tuning = np.ones((2, 8), dtype=float)
        spikes = np.ones((2, 8), dtype=float)
        xbin = np.linspace(0, 8, 9)
        rm = Ratemap(tuning_curves=tuning, unsmoothed_tuning_maps=tuning * 1.1, spikes_maps=spikes, xbin=xbin, ybin=None, occupancy=np.ones(8), neuron_ids=np.array([10, 20]), neuron_extended_ids=None, metadata={'k': 1})
        self.assertEqual(rm.metadata.get('k'), 1)
        self.assertIsNotNone(rm.unsmoothed_tuning_maps)
        self.assertIsNone(rm.ybin)

    def test_neuron_ids_length_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            Ratemap(np.ones((2, 5)), neuron_ids=[1, 2, 3])

    def test_neuron_extended_ids_requires_matching_neuron_ids(self):
        from neuropy.core.neuron_identities import NeuronExtendedIdentity
        ext = [NeuronExtendedIdentity(1, 1, 1, 1), NeuronExtendedIdentity(1, 2, 2, 1)]
        with self.assertRaises(AssertionError):
            Ratemap(np.ones((2, 5)), neuron_ids=[1], neuron_extended_ids=ext)

    def test_asarray_conversion_on_inputs(self):
        rm = Ratemap([[1.0, 2.0], [3.0, 4.0]], spikes_maps=[[1, 0], [0, 1]], occupancy=[1.0, 1.0], neuron_ids=[1, 2], xbin=[0, 1, 2])
        self.assertIsInstance(rm.tuning_curves, np.ndarray)
        self.assertIsInstance(rm.spikes_maps, np.ndarray)

    def test_metadata_merge_semantics_from_datawriter(self):
        rm = build_synthetic_ratemap_1d(metadata={'a': 1})
        rm.metadata = {'b': 2}
        self.assertEqual(rm.metadata.get('a'), 1)
        self.assertEqual(rm.metadata.get('b'), 2)


class TestRatemapProperties1D(unittest.TestCase):
    def setUp(self):
        self.rm = build_synthetic_ratemap_1d()

    def test_shape_and_ndim(self):
        self.assertEqual(self.rm.ndim, 1)
        self.assertEqual(self.rm.n_neurons, self.rm.tuning_curves.shape[0])
        self.assertEqual(self.rm.tuning_curves.shape[1], len(self.rm.xbin) - 1)

    def test_bin_centers_from_mixin(self):
        self.assertEqual(len(self.rm.xbin_centers), len(self.rm.xbin) - 1)
        self.assertIsNone(getattr(self.rm, 'ybin', None) or None)
        # ybin_centers may raise or return None depending on mixin; accept either
        ybin = self.rm.ybin
        self.assertTrue(ybin is None)

    def test_neuron_id_property_roundtrip(self):
        original = np.asarray(self.rm.neuron_ids).copy()
        self.rm.neuron_ids = original[::-1]
        np.testing.assert_array_equal(self.rm.neuron_ids, original[::-1])
        self.rm.neuron_ids = original

    def test_neuron_extended_ids_property_roundtrip(self):
        original = list(self.rm.neuron_extended_ids)
        self.rm.neuron_extended_ids = original[::-1]
        self.assertEqual([e.aclu for e in self.rm.neuron_extended_ids], [e.aclu for e in original[::-1]])
        self.rm.neuron_extended_ids = original

    def test_tuning_curves_dict_keys(self):
        d = self.rm.tuning_curves_dict
        self.assertEqual(set(d.keys()), set(np.asarray(self.rm.neuron_ids).tolist()))
        for aclu, curve in d.items():
            idx = list(self.rm.neuron_ids).index(aclu)
            np.testing.assert_allclose(curve, self.rm.tuning_curves[idx])

    def test_pdf_normalization_sums_to_one(self):
        pdf = self.rm.pdf_normalized_tuning_curves
        np.testing.assert_allclose(np.nansum(pdf, axis=1), np.ones(self.rm.n_neurons), rtol=1e-6)
        np.testing.assert_allclose(self.rm.normalized_tuning_curves, pdf)

    def test_unit_max_tuning_curves(self):
        um = self.rm.unit_max_tuning_curves
        np.testing.assert_allclose(np.nanmax(um, axis=1), np.ones(self.rm.n_neurons), rtol=1e-6)

    def test_minmax_normalized_tuning_curves_range(self):
        mm = self.rm.minmax_normalized_tuning_curves
        self.assertTrue(np.nanmin(mm) >= -1e-9)
        self.assertTrue(np.nanmax(mm) <= 1.0 + 1e-9)

    def test_peak_firing_rates(self):
        with self.assertWarns(UserWarning):
            peaks = self.rm.tuning_curve_peak_firing_rates
        np.testing.assert_allclose(peaks, np.nanmax(self.rm.tuning_curves, axis=1))
        unsmoothed_peaks = self.rm.tuning_curve_unsmoothed_peak_firing_rates
        np.testing.assert_allclose(unsmoothed_peaks, np.nanmax(self.rm.unsmoothed_tuning_maps, axis=1))

    def test_spatial_sparcity_shape(self):
        sparcity = self.rm.spatial_sparcity
        self.assertEqual(sparcity.shape, (self.rm.n_neurons,))
        self.assertTrue(np.all(np.isfinite(sparcity)))

    def test_occupancy_masks(self):
        never = self.rm.never_visited_occupancy_mask
        self.assertTrue(never[0])
        nan_occ = self.rm.nan_never_visited_occupancy
        self.assertTrue(np.isnan(nan_occ[0]))
        visited = self.rm.visited_occupancy_mask
        self.assertEqual(visited[0], 0.0)
        self.assertTrue(np.all(visited[1:] == 1.0))
        p_occ = self.rm.probability_normalized_occupancy
        np.testing.assert_allclose(np.nansum(p_occ), 1.0, rtol=1e-6)

    def test_field_mutation_occupancy_and_ybin(self):
        # Call-site compatibility: PfND setters and PendingNotebookCode mutate these
        new_occ = self.rm.occupancy * 2.0
        self.rm.occupancy = new_occ
        np.testing.assert_array_equal(self.rm.occupancy, new_occ)
        self.rm.ybin = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_equal(self.rm.ybin, [0.0, 1.0, 2.0])
        self.rm.ybin = None


class TestRatemapProperties2D(unittest.TestCase):
    def setUp(self):
        self.rm = build_synthetic_ratemap_2d()

    def test_ndim_and_shapes(self):
        self.assertEqual(self.rm.ndim, 2)
        n, nx, ny = self.rm.tuning_curves.shape
        self.assertEqual(n, self.rm.n_neurons)
        self.assertEqual(nx, len(self.rm.xbin) - 1)
        self.assertEqual(ny, len(self.rm.ybin) - 1)
        self.assertEqual(len(self.rm.xbin_centers), nx)
        self.assertEqual(len(self.rm.ybin_centers), ny)

    def test_pdf_normalization_2d(self):
        pdf = self.rm.pdf_normalized_tuning_curves
        sums = np.nansum(pdf, axis=(1, 2))
        np.testing.assert_allclose(sums, np.ones(self.rm.n_neurons), rtol=1e-6)


class TestRatemapSlicing(unittest.TestCase):
    def setUp(self):
        self.rm = build_synthetic_ratemap_1d()

    def test_getitem_integer(self):
        # Scalar integer indexing currently reduces the neuron axis entirely
        # (numpy advanced-index quirk on the stored arrays). Prefer list/bool masks
        # for single-neuron slices in call sites; document both behaviors here.
        sliced_list = self.rm[[1]]
        self.assertEqual(sliced_list.n_neurons, 1)
        self.assertEqual(int(np.asarray(sliced_list.neuron_ids).item()), int(self.rm.neuron_ids[1]))
        np.testing.assert_allclose(sliced_list.tuning_curves, self.rm.tuning_curves[1:2])

        sliced_scalar = self.rm[1]
        # Current behavior: scalar index collapses neuron axis → ndim of tuning_curves drops
        self.assertEqual(sliced_scalar.tuning_curves.ndim, self.rm.tuning_curves.ndim - 1)
        np.testing.assert_allclose(sliced_scalar.tuning_curves, self.rm.tuning_curves[1])

    def test_getitem_list_indices(self):
        idxs = [0, 2]
        sliced = self.rm[idxs]
        self.assertEqual(sliced.n_neurons, 2)
        np.testing.assert_array_equal(sliced.neuron_ids, np.asarray(self.rm.neuron_ids)[idxs])
        np.testing.assert_allclose(sliced.tuning_curves, self.rm.tuning_curves[idxs])
        self.assertEqual(len(sliced.neuron_extended_ids), 2)
        # occupancy / bins are shared (not neuron-sliced)
        np.testing.assert_array_equal(sliced.occupancy, self.rm.occupancy)
        np.testing.assert_array_equal(sliced.xbin, self.rm.xbin)

    def test_getitem_does_not_mutate_original(self):
        original_ids = np.asarray(self.rm.neuron_ids).copy()
        _ = self.rm[[0]]
        np.testing.assert_array_equal(self.rm.neuron_ids, original_ids)

    def test_get_by_id(self):
        ids = np.asarray(self.rm.neuron_ids)[[0, 2]]
        sliced = self.rm.get_by_id(ids)
        np.testing.assert_array_equal(sliced.neuron_ids, ids)
        self.assertEqual(sliced.n_neurons, 2)

    def test_get_by_id_missing_raises(self):
        with self.assertRaises(AssertionError):
            self.rm.get_by_id([9999])

    def test_get_sort_indicies_default_and_custom(self):
        sort_ind = self.rm.get_sort_indicies()
        self.assertEqual(len(sort_ind), self.rm.n_neurons)
        custom = np.array([2, 0, 1, 3][:self.rm.n_neurons])
        np.testing.assert_array_equal(self.rm.get_sort_indicies(sortby=custom), custom)


class TestRatemapStaticHelpers(unittest.TestCase):
    def test_perform_aoc_normalization_1d_and_2d(self):
        tc1 = build_synthetic_ratemap_1d().tuning_curves
        out1 = Ratemap.perform_AOC_normalization(tc1)
        np.testing.assert_allclose(np.nansum(out1, axis=1), 1.0, rtol=1e-6)
        tc2 = build_synthetic_ratemap_2d().tuning_curves
        out2 = Ratemap.perform_AOC_normalization(tc2)
        np.testing.assert_allclose(np.nansum(out2, axis=(1, 2)), 1.0, rtol=1e-6)

    def test_occupancy_static_helpers(self):
        occ = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_equal(Ratemap.build_never_visited_mask(occ), [True, False, False])
        np.testing.assert_array_equal(Ratemap.build_visited_mask(occ), [False, True, True])
        nan_occ = Ratemap.nan_never_visited_locations(occ)
        self.assertTrue(np.isnan(nan_occ[0]))
        visited = Ratemap.visited_locations_mask(occ)
        np.testing.assert_array_equal(visited, [0.0, 1.0, 1.0])

    def test_nanmin_nanmax_scaler_empty(self):
        empty = np.array([]).reshape(0, 3)
        out = Ratemap.nanmin_nanmax_scaler(empty)
        np.testing.assert_array_equal(out, empty)

    def test_normalize_data(self):
        data = np.array([np.nan, 0.0, 5.0, 10.0])
        out = Ratemap.NormalizeData(data.copy())
        self.assertAlmostEqual(float(np.nanmin(out)), 0.0)
        self.assertAlmostEqual(float(np.nanmax(out)), 1.0)


class TestRatemapProjectionsAndMerge(unittest.TestCase):
    def test_build_1d_maximum_projection(self):
        rm2d = build_synthetic_ratemap_2d()
        rm1d = rm2d.to_1D_maximum_projection()
        self.assertEqual(rm1d.ndim, 1)
        self.assertEqual(rm1d.n_neurons, rm2d.n_neurons)
        np.testing.assert_array_equal(rm1d.neuron_ids, rm2d.neuron_ids)
        self.assertIsNone(rm1d.ybin)
        np.testing.assert_allclose(rm1d.tuning_curves, np.nanmax(rm2d.tuning_curves, axis=-1))
        np.testing.assert_allclose(rm1d.occupancy, np.sum(rm2d.occupancy, axis=-1))

    def test_build_1d_projection_rejects_1d_input(self):
        with self.assertRaises(AssertionError):
            build_synthetic_ratemap_1d().to_1D_maximum_projection()

    def test_build_merged_ratemap(self):
        lhs = build_synthetic_ratemap_1d(seed=0)
        rhs = deepcopy(lhs)
        # NeuronExtendedIdentity is defined with eq=False, so merge requires shared identity objects
        rhs.neuron_extended_ids = lhs.neuron_extended_ids
        merged = Ratemap.build_merged_ratemap(lhs, rhs, debug_print=False)
        self.assertEqual(merged.tuning_curves.shape[-1], 2)
        self.assertEqual(merged.spikes_maps.shape[-1], 2)
        self.assertEqual(merged.occupancy.shape[-1], 2)
        np.testing.assert_array_equal(merged.neuron_ids, lhs.neuron_ids)


class TestRatemapPeaks(unittest.TestCase):
    def test_compute_tuning_curve_modes_and_peak_df(self):
        rm = build_synthetic_ratemap_1d()
        peaks_dict, aclu_n_peaks_dict, peaks_df = rm.compute_tuning_curve_modes(height=0.1, width=1)
        self.assertIsInstance(peaks_dict, dict)
        self.assertIsInstance(aclu_n_peaks_dict, dict)
        self.assertIsInstance(peaks_df, pd.DataFrame)
        self.assertIn('aclu', peaks_df.columns)
        for aclu in rm.neuron_ids:
            self.assertIn(int(aclu), {int(k) for k in peaks_dict.keys()} | set(peaks_dict.keys()))


class TestRatemapSerialization(unittest.TestCase):
    def test_pickle_roundtrip(self):
        rm = build_synthetic_ratemap_1d(metadata={'pickle': True})
        blob = pickle.dumps(rm)
        restored = pickle.loads(blob)
        np.testing.assert_allclose(restored.tuning_curves, rm.tuning_curves)
        np.testing.assert_array_equal(restored.neuron_ids, rm.neuron_ids)
        np.testing.assert_allclose(restored.occupancy, rm.occupancy)
        self.assertEqual(restored.metadata.get('pickle'), True)
        self.assertEqual(restored.n_neurons, rm.n_neurons)

    def test_deepcopy_independence(self):
        rm = build_synthetic_ratemap_1d()
        copy_rm = deepcopy(rm)
        copy_rm.tuning_curves = copy_rm.tuning_curves + 1.0
        self.assertFalse(np.allclose(copy_rm.tuning_curves, rm.tuning_curves))

    def test_to_hdf_writes_expected_datasets(self):
        rm = build_synthetic_ratemap_2d()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'ratemap_test.h5')
            rm.to_hdf(path, key='rm')
            with h5py.File(path, 'r') as f:
                group = f['rm']
                self.assertEqual(int(group.attrs['n_neurons']), rm.n_neurons)
                self.assertEqual(int(group.attrs['ndim']), rm.ndim)
                for name in ['occupancy', 'tuning_curves', 'spikes_maps', 'unsmoothed_tuning_maps', 'neuron_ids', 'xbin', 'xbin_centers', 'ybin', 'ybin_centers']:
                    self.assertIn(name, group)
                np.testing.assert_allclose(group['tuning_curves'][...], rm.tuning_curves)
                np.testing.assert_array_equal(group['neuron_ids'][...], rm.neuron_ids)


class TestRatemapPrivateStorageCompatibility(unittest.TestCase):
    """Pin current private attribute names so an attrs conversion keeps pickle-safe names."""

    def test_private_neuron_id_storage_names(self):
        rm = build_synthetic_ratemap_1d()
        self.assertTrue(hasattr(rm, '_neuron_ids'))
        self.assertTrue(hasattr(rm, '_neuron_extended_ids'))
        np.testing.assert_array_equal(rm._neuron_ids, rm.neuron_ids)

    def test_metadata_backed_by_datawriter_storage(self):
        rm = build_synthetic_ratemap_1d(metadata={'m': 1})
        # DataWriter stores via _metadata after property assignment in __init__
        self.assertTrue(hasattr(rm, '_metadata') or rm.metadata is not None)
        self.assertEqual(rm.metadata.get('m'), 1)


if __name__ == '__main__':
    unittest.main()
