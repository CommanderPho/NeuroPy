import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
sys.path.insert(0, str(root_project_folder))

from neuropy.utils.position_util import ShapelyMaze, ShapelyMazeCollection, resolve_shapely_valid_epochs, build_shapely_maze_collection_for_session


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


if __name__ == '__main__':
    unittest.main()
