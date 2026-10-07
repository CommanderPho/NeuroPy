import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
import numpy as np
import pandas as pd

tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
sys.path.insert(0, str(root_project_folder))

from neuropy.utils.position_util import ShapelyMaze, ShapelyMazeCollection, CircularRingLinearizationParams, build_shapely_maze_collection_for_session


def resolve_shapely_valid_epochs(curr_active_pipeline, pos_df: pd.DataFrame, shapely_maze_collection: ShapelyMazeCollection, maze_epoch_keys: List[str], epochs_df: Optional[pd.DataFrame] = None, valid_epochs_override: Optional[Dict[str, Tuple[float, float]]] = None, min_position_samples: int = 100,
        min_epoch_duration_sec: float = 60.0, min_on_track_fraction: float = 0.3, max_track_distance_cm: float = 25.0, enable_position_occupancy_refinement: bool = True, debug_print: bool = True) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, str]]:
    """Resolve per-maze time bounds for shapely linearization with tiered fallbacks.

    Priority per key: override -> session epochs -> occupancy refinement -> template fallback -> omit.
    Returns (valid_epochs, provenance) where provenance values are one of:
    'override', 'epochs', 'epochs_refined', 'occupancy', 'template_fallback', 'missing'.
    """

    def _subfn_extract_epoch_bounds_from_epochs_df(epochs_df: Optional[pd.DataFrame], label: str, start_col: str = 'start', stop_col: str = 'stop') -> Optional[Tuple[float, float]]:
        """Return (start, stop) for a single epoch label, or None if missing."""
        if epochs_df is None or len(epochs_df) == 0 or 'label' not in epochs_df.columns:
            return None
        label_rows = epochs_df[epochs_df['label'] == label]
        if len(label_rows) == 0:
            return None
        if len(label_rows) > 1:
            duration_col = 'duration' if 'duration' in label_rows.columns else None
            if duration_col is not None:
                label_rows = label_rows.sort_values(by=duration_col, ascending=False)
            else:
                label_rows = label_rows.copy()
                label_rows['_duration'] = label_rows[stop_col].astype(float) - label_rows[start_col].astype(float)
                label_rows = label_rows.sort_values(by='_duration', ascending=False)
            if debug_print:
                print(f"resolve_shapely_valid_epochs: label {label!r} has {len(label_rows)} rows; using longest duration.")
        row = label_rows.iloc[0]
        return (float(row[start_col]), float(row[stop_col]))

    def _subfn_compute_on_track_mask(pos_df: pd.DataFrame, shapely_maze: ShapelyMaze) -> np.ndarray:
        """Boolean mask: True where sample is within max_track_distance_cm of the maze skeleton."""
        return shapely_maze.compute_on_track_mask(pos_df['x'].to_numpy(), pos_df['y'].to_numpy(), max_track_distance_cm)

    def _subfn_validate_shapely_epoch_bounds(pos_df: pd.DataFrame, shapely_maze: ShapelyMaze, t0: float, t1: float) -> bool:
        """Return True if epoch bounds contain enough on-track position samples."""
        if t1 <= t0:
            return False
        if (t1 - t0) < min_epoch_duration_sec:
            return False
        window_df = pos_df[(pos_df['t'] >= t0) & (pos_df['t'] <= t1)].dropna(subset=['x', 'y'], how='any')
        if len(window_df) < min_position_samples:
            return False
        on_track_mask = _subfn_compute_on_track_mask(window_df, shapely_maze)
        on_track_fraction = float(np.mean(on_track_mask)) if len(on_track_mask) > 0 else 0.0
        return on_track_fraction >= min_on_track_fraction

    def _subfn_infer_epoch_bounds_from_track_occupancy(pos_df: pd.DataFrame, shapely_maze: ShapelyMaze, search_t0: float, search_t1: float) -> Optional[Tuple[float, float]]:
        """Infer epoch bounds from the largest contiguous on-track occupancy segment within the search window."""
        min_segment_samples = min(min_position_samples, 50)
        min_segment_duration_sec = min(min_epoch_duration_sec, 30.0)
        if search_t1 <= search_t0:
            return None
        window_df = pos_df[(pos_df['t'] >= search_t0) & (pos_df['t'] <= search_t1)].dropna(subset=['x', 'y', 't'], how='any')
        if len(window_df) < min_segment_samples:
            return None
        window_df = window_df.sort_values('t').reset_index(drop=True)
        on_track_mask = _subfn_compute_on_track_mask(window_df, shapely_maze)
        if not np.any(on_track_mask):
            return None
        regions = contiguous_regions(on_track_mask)
        if len(regions) == 0:
            return None
        best_region = None
        best_n_samples = -1
        t_values = window_df['t'].to_numpy()
        for region in regions:
            start_idx, end_idx = int(region[0]), int(region[1])
            n_samples = end_idx - start_idx
            if n_samples < min_segment_samples:
                continue
            t_start, t_end = float(t_values[start_idx]), float(t_values[end_idx - 1])
            if (t_end - t_start) < min_segment_duration_sec:
                continue
            if n_samples > best_n_samples:
                best_n_samples = n_samples
                best_region = (t_start, t_end)
        return best_region

    # ==================================================================================================================================================================================================================================================================================== #
    # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #
    valid_epochs_override = valid_epochs_override or {}
    template_valid_epochs = shapely_maze_collection.valid_epochs or {}
    resolved_valid_epochs: Dict[str, Tuple[float, float]] = {}
    provenance: Dict[str, str] = {}


    # ==================================================================================================================================================================================================================================================================================== #
    # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #
    epochs_determined_by_sess_pos_df_dict = {k:(curr_active_pipeline.filtered_sessions[k].position.to_dataframe().dropna(subset=['x', 'y', 't'], how='any', inplace=False)['t'].min(), curr_active_pipeline.filtered_sessions[k].position.to_dataframe().dropna(subset=['x', 'y', 't'], how='any', inplace=False)['t'].max()) for k in maze_epoch_keys}
    resolved_valid_epochs: Dict[str, Tuple[float, float]] = epochs_determined_by_sess_pos_df_dict

    # pos_df = pos_df.dropna(subset=['x', 'y', 't'], how='any')
    # if len(pos_df) == 0:
    #     if debug_print:
    #         print("resolve_shapely_valid_epochs: empty position dataframe; no bounds resolved.")
    #     return resolved_valid_epochs, provenance
    # session_t_min, session_t_max = float(pos_df['t'].min()), float(pos_df['t'].max())
    # maze_keys = [k for k in maze_epoch_keys if k in shapely_maze_collection.shapelyMazes]
    # missing_geometry_keys = set(maze_epoch_keys) - set(maze_keys)
    # if missing_geometry_keys and debug_print:
    #     print(f"resolve_shapely_valid_epochs: no shapely geometry for keys {sorted(missing_geometry_keys)}; skipping.")
    # prior_maze_stop: Optional[float] = None

    # for maze_key in maze_keys:
    #     shapely_maze = shapely_maze_collection.shapelyMazes[maze_key]
    #     bounds: Optional[Tuple[float, float]] = None
    #     source: str = 'missing'
    #     if maze_key in valid_epochs_override:
    #         bounds = (float(valid_epochs_override[maze_key][0]), float(valid_epochs_override[maze_key][1]))
    #         source = 'override'
    #     epoch_bounds = _subfn_extract_epoch_bounds_from_epochs_df(epochs_df=epochs_df, label=maze_key)
    #     search_t0 = epoch_bounds[0] if epoch_bounds is not None else session_t_min
    #     search_t1 = epoch_bounds[1] if epoch_bounds is not None else session_t_max
    #     if prior_maze_stop is not None:
    #         search_t0 = max(search_t0, prior_maze_stop)
    #     if bounds is None and epoch_bounds is not None:
    #         t0, t1 = epoch_bounds
    #         if prior_maze_stop is not None:
    #             t0 = max(t0, prior_maze_stop)
    #         if _subfn_validate_shapely_epoch_bounds(pos_df, shapely_maze, t0, t1):
    #             bounds = (t0, t1)
    #             source = 'epochs'
    #         elif enable_position_occupancy_refinement:
    #             occupancy_bounds = _subfn_infer_epoch_bounds_from_track_occupancy(pos_df, shapely_maze, search_t0=max(t0, search_t0), search_t1=t1)
    #             if occupancy_bounds is not None and _subfn_validate_shapely_epoch_bounds(pos_df, shapely_maze, occupancy_bounds[0], occupancy_bounds[1]):
    #                 bounds = occupancy_bounds
    #                 source = 'epochs_refined'
    #     if bounds is None and enable_position_occupancy_refinement:
    #         occupancy_bounds = _subfn_infer_epoch_bounds_from_track_occupancy(pos_df, shapely_maze, search_t0=search_t0, search_t1=search_t1)
    #         if occupancy_bounds is not None and _subfn_validate_shapely_epoch_bounds(pos_df, shapely_maze, occupancy_bounds[0], occupancy_bounds[1]):
    #             bounds = occupancy_bounds
    #             source = 'occupancy'
    #     if bounds is None and epoch_bounds is not None:
    #         # A session's own epoch-label bounds are more trustworthy than another session's hardcoded template times, even when on-track geometry validation is weak (e.g. RatU/RatJ reusing RatK/RatS maze geometry whose LineString does not match this session's track). Use them (unvalidated) before falling back to the cross-session template times.
    #         t0, t1 = epoch_bounds
    #         if prior_maze_stop is not None:
    #             t0 = max(t0, prior_maze_stop)
    #         if t1 > t0:
    #             bounds = (t0, t1)
    #             source = 'epochs_unvalidated'
    #             if debug_print:
    #                 print(f"resolve_shapely_valid_epochs: {maze_key} using session epoch bounds {bounds} (unvalidated; geometry validation failed but preferred over cross-session template fallback).")
    #     if bounds is None and maze_key in template_valid_epochs:
    #         template_bounds = (float(template_valid_epochs[maze_key][0]), float(template_valid_epochs[maze_key][1]))
    #         t0, t1 = template_bounds
    #         if prior_maze_stop is not None:
    #             t0 = max(t0, prior_maze_stop)
    #         if _subfn_validate_shapely_epoch_bounds(pos_df, shapely_maze, t0, t1):
    #             bounds = (t0, t1)
    #             source = 'template_fallback'
    #     if bounds is not None:
    #         resolved_valid_epochs[maze_key] = bounds
    #         provenance[maze_key] = source
    #         prior_maze_stop = bounds[1]
    #         if debug_print:
    #             print(f"resolve_shapely_valid_epochs: {maze_key} -> {bounds} (source={source})")
    #     else:
    #         provenance[maze_key] = 'missing'
    #         if debug_print:
    #             print(f"resolve_shapely_valid_epochs: {maze_key} -> MISSING (all tiers failed)")
    return resolved_valid_epochs, provenance

def _make_horizontal_track_maze():
    return ShapelyMaze(nodes=[(-50.0, 0.0), (50.0, 0.0)])


def _make_pos_on_track(t_start: float, t_end: float, n_samples: int = 200, x: float = 0.0, y: float = 0.0) -> pd.DataFrame:
    t = np.linspace(t_start, t_end, n_samples)
    return pd.DataFrame({'t': t, 'x': np.full(n_samples, x), 'y': np.full(n_samples, y)})


class TestShapelyValidEpochs(unittest.TestCase):

    def setUp(self):
        self.maze1 = _make_horizontal_track_maze()
        self.maze2 = ShapelyMaze(nodes=[(-50.0, 50.0), (50.0, 50.0)])
        self.template = ShapelyMazeCollection(shapelyMazes={'maze1': self.maze1, 'maze2': self.maze2}, valid_epochs={'maze1': (100.0, 200.0), 'maze2': (300.0, 400.0)})

    def test_resolves_from_epochs_when_labels_match(self):
        pos_df = pd.concat([_make_pos_on_track(110.0, 190.0, y=0.0), _make_pos_on_track(310.0, 390.0, y=50.0)], ignore_index=True)
        epochs_df = pd.DataFrame({'start': [110.0, 310.0], 'stop': [190.0, 390.0], 'label': ['maze1', 'maze2']})
        valid_epochs, provenance = resolve_shapely_valid_epochs(pos_df=pos_df, shapely_maze_collection=self.template, maze_epoch_keys=['maze1', 'maze2'], epochs_df=epochs_df, debug_print=False)
        self.assertEqual(provenance['maze1'], 'epochs')
        self.assertEqual(provenance['maze2'], 'epochs')
        self.assertAlmostEqual(valid_epochs['maze1'][0], 110.0)
        self.assertAlmostEqual(valid_epochs['maze2'][0], 310.0)

    def test_overlapping_two_novel_epochs_still_extracts_maze_labels(self):
        pos_df = pd.concat([_make_pos_on_track(11070.0, 13970.0, n_samples=250), _make_pos_on_track(20756.0, 24004.0, n_samples=250, y=50.0)], ignore_index=True)
        epochs_df = pd.DataFrame({'start': [0.0, 11070.0, 21176.0, 13972.0, 24006.0], 'stop': [11066.0, 13970.0, 24004.0, 20754.0, 42305.0], 'label': ['pre', 'maze1', 'maze2', 'post1', 'post2']})
        valid_epochs, provenance = resolve_shapely_valid_epochs(pos_df=pos_df, shapely_maze_collection=self.template, maze_epoch_keys=['maze1', 'maze2'], epochs_df=epochs_df, min_position_samples=50, min_epoch_duration_sec=10.0, debug_print=False)
        self.assertIn('maze1', valid_epochs)
        self.assertIn('maze2', valid_epochs)
        self.assertEqual(provenance['maze1'], 'epochs')

    def test_occupancy_recovers_when_epoch_bounds_are_wrong(self):
        t = np.linspace(300.0, 400.0, 200)
        y = np.where((t >= 350.0) & (t <= 390.0), 50.0, 999.0)
        pos_df = pd.DataFrame({'t': t, 'x': np.zeros(200), 'y': y})
        epochs_df = pd.DataFrame({'start': [300.0], 'stop': [400.0], 'label': ['maze2']})
        valid_epochs, provenance = resolve_shapely_valid_epochs(pos_df=pos_df, shapely_maze_collection=self.template, maze_epoch_keys=['maze2'], epochs_df=epochs_df, min_position_samples=50, min_epoch_duration_sec=10.0, min_on_track_fraction=0.45, debug_print=False)
        self.assertIn(provenance['maze2'], ('epochs_refined', 'occupancy'))
        self.assertGreater(valid_epochs['maze2'][1] - valid_epochs['maze2'][0], 25.0)
        self.assertLess(valid_epochs['maze2'][1] - valid_epochs['maze2'][0], 45.0)

    def test_missing_maze2_label_uses_fallback_tier(self):
        pos_df = pd.concat([_make_pos_on_track(110.0, 190.0), _make_pos_on_track(300.0, 380.0, y=50.0)], ignore_index=True)
        epochs_df = pd.DataFrame({'start': [110.0], 'stop': [190.0], 'label': ['maze1']})
        valid_epochs, provenance = resolve_shapely_valid_epochs(pos_df=pos_df, shapely_maze_collection=self.template, maze_epoch_keys=['maze1', 'maze2'], epochs_df=epochs_df, min_position_samples=50, min_epoch_duration_sec=10.0, debug_print=False)
        self.assertEqual(provenance['maze1'], 'epochs')
        self.assertIn(provenance['maze2'], ('occupancy', 'template_fallback'))
        self.assertIn('maze2', valid_epochs)

    def test_valid_epochs_override_wins(self):
        pos_df = _make_pos_on_track(500.0, 600.0, y=50.0)
        epochs_df = pd.DataFrame({'start': [300.0], 'stop': [400.0], 'label': ['maze2']})
        override = {'maze2': (500.0, 600.0)}
        valid_epochs, provenance = resolve_shapely_valid_epochs(pos_df=pos_df, shapely_maze_collection=self.template, maze_epoch_keys=['maze2'], epochs_df=epochs_df, valid_epochs_override=override, debug_print=False)
        self.assertEqual(provenance['maze2'], 'override')
        self.assertEqual(valid_epochs['maze2'], (500.0, 600.0))

    def test_template_fallback_when_occupancy_disabled(self):
        pos_df = pd.concat([_make_pos_on_track(110.0, 190.0), _make_pos_on_track(300.0, 380.0, y=50.0)], ignore_index=True)
        epochs_df = pd.DataFrame({'start': [110.0], 'stop': [190.0], 'label': ['maze1']})
        valid_epochs, provenance = resolve_shapely_valid_epochs(pos_df=pos_df, shapely_maze_collection=self.template, maze_epoch_keys=['maze1', 'maze2'], epochs_df=epochs_df, min_position_samples=50, min_epoch_duration_sec=10.0, enable_position_occupancy_refinement=False, debug_print=False)
        self.assertEqual(provenance['maze2'], 'template_fallback')
        self.assertAlmostEqual(valid_epochs['maze2'][0], 300.0)

    def test_all_tiers_fail_omits_key_without_exception(self):
        pos_df = _make_pos_on_track(1000.0, 1100.0, n_samples=10, y=999.0)
        valid_epochs, provenance = resolve_shapely_valid_epochs(pos_df=pos_df, shapely_maze_collection=self.template, maze_epoch_keys=['maze1'], epochs_df=None, min_position_samples=100, debug_print=False)
        self.assertNotIn('maze1', valid_epochs)
        self.assertEqual(provenance['maze1'], 'missing')

    def test_build_shapely_maze_collection_for_session(self):
        pos_df = _make_pos_on_track(110.0, 190.0)
        epochs_df = pd.DataFrame({'start': [110.0], 'stop': [190.0], 'label': ['maze1']})
        collection = build_shapely_maze_collection_for_session(pos_df=pos_df, geometry_template=self.template, maze_epoch_keys=['maze1'], epochs_df=epochs_df, debug_print=False)
        self.assertIn('maze1', collection.valid_epochs)
        self.assertIn('maze1', collection.shapelyMazes)


class TestShapelyMazeLinearization(unittest.TestCase):

    def test_linestring_mode_backwards_compatible(self):
        maze = _make_horizontal_track_maze()
        df = pd.DataFrame({'x': np.linspace(-40.0, 40.0, 9), 'y': np.zeros(9)})
        linestring_lin = maze.shapely_linearize_trajectory(df)
        dispatch_lin = maze.linearize_trajectory(df)
        pd.testing.assert_series_equal(linestring_lin, dispatch_lin)
        self.assertAlmostEqual(float(dispatch_lin.iloc[0]), 10.0, places=5)
        self.assertAlmostEqual(float(dispatch_lin.iloc[-1]), 90.0, places=5)

    def test_angular_ring_center_only_maps_full_circle(self):
        maze = ShapelyMaze(
            nodes=[(0.0, 0.0), (1.0, 0.0)],
            linearization_mode='angular_ring',
            ring_params=CircularRingLinearizationParams(center_x=0.0, center_y=0.0),
        )
        df = pd.DataFrame({'x': [100.0, 0.0, -100.0, 0.0, np.nan], 'y': [0.0, 100.0, 0.0, -100.0, 0.0]})
        lin = maze.linearize_trajectory(df)
        np.testing.assert_allclose(lin.iloc[:4].to_numpy(), np.array([0.0, 0.25, 0.5, 0.75]), atol=1e-12)
        self.assertTrue(np.isnan(lin.iloc[4]))

    def test_angular_ring_linearizes_valid_arc_to_unit_interval(self):
        center_x, center_y, radius = 0.0, 0.0, 100.0
        gap_start, gap_end = np.deg2rad(-25.0), np.deg2rad(25.0)
        maze = ShapelyMaze(
            nodes=[(radius, 0.0), (0.0, radius), (-radius, 0.0), (0.0, -radius)],
            linearization_mode='angular_ring',
            ring_params=CircularRingLinearizationParams(
                center_x=center_x, center_y=center_y, radius_cm=radius,
                gap_angle_start_rad=gap_start, gap_angle_end_rad=gap_end,
                arc_direction='ccw', max_radius_deviation_cm=5.0, output_range=(0.0, 1.0),
            ),
        )
        angles_deg = np.array([90.0, 180.0, -90.0, 0.0, 10.0, -10.0])
        angles_rad = np.deg2rad(angles_deg)
        df = pd.DataFrame({'x': center_x + radius * np.cos(angles_rad), 'y': center_y + radius * np.sin(angles_rad)})
        lin = maze.linearize_trajectory(df)
        self.assertTrue(np.all(np.isfinite(lin.iloc[:3])))
        self.assertTrue(np.all((lin.iloc[:3] >= 0.0) & (lin.iloc[:3] <= 1.0)))
        self.assertTrue(lin.iloc[1] > lin.iloc[0])
        self.assertTrue(lin.iloc[2] > lin.iloc[1])
        self.assertTrue(np.all(np.isnan(lin.iloc[3:])))

    def test_angular_ring_on_track_mask(self):
        center_x, center_y, radius = 0.0, 0.0, 100.0
        maze = ShapelyMaze(
            nodes=[(radius, 0.0), (0.0, radius), (-radius, 0.0)],
            linearization_mode='angular_ring',
            ring_params=CircularRingLinearizationParams(
                center_x=center_x, center_y=center_y, radius_cm=radius,
                gap_angle_start_rad=np.deg2rad(-25.0), gap_angle_end_rad=np.deg2rad(25.0),
                arc_direction='ccw', max_radius_deviation_cm=5.0,
            ),
        )
        x = np.array([0.0, -100.0, 0.0, 100.0])
        y = np.array([100.0, 0.0, 130.0, 0.0])
        mask = maze.compute_on_track_mask(x, y, max_track_distance_cm=15.0)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])

    def test_angular_ring_center_only_on_track_mask_all_finite_points(self):
        maze = ShapelyMaze(
            nodes=[(0.0, 0.0), (1.0, 0.0)],
            linearization_mode='angular_ring',
            ring_params=CircularRingLinearizationParams(center_x=0.0, center_y=0.0),
        )
        x = np.array([0.0, -100.0, 0.0, np.nan])
        y = np.array([100.0, 0.0, 130.0, 0.0])
        mask = maze.compute_on_track_mask(x, y, max_track_distance_cm=15.0)
        np.testing.assert_array_equal(mask, np.array([True, True, True, False]))


if __name__ == '__main__':
    unittest.main()
