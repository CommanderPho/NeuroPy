from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.ndimage import gaussian_filter1d
from rdp import rdp ## used by `simplify_position_trajectory`

from .. import core
from neuropy.utils.mathutil import contiguous_regions, threshPeriods, compute_grid_bin_bounds, map_value
from neuropy.utils.mixins.binning_helpers import compute_spanning_bins
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
from enum import Enum
from attrs import define, field, Factory
import numpy as np
from neuropy.core.epoch import NamedTimerange, EpochsAccessor, Epoch
from shapely.geometry import LineString, Point # for ShapelyMaze


class RegularizationApproach(Enum):
    """Docstring for RegularizationApproach."""
    RAW_VALUES = "raw_values"
    SUBTRACT_MIN = "subtract_min"
    RESTORE_X_RANGE = "restore_x_range" # restores the original range of the x values after performing the linearization.
    


@define(slots=False)
class CircularRingLinearizationParams:
    center_x: float
    center_y: float
    radius_cm: Optional[float] = None
    gap_angle_start_rad: Optional[float] = None
    gap_angle_end_rad: Optional[float] = None
    arc_direction: str = 'ccw'
    max_radius_deviation_cm: Optional[float] = None
    output_range: Tuple[float, float] = (0.0, 1.0)
    angle_origin_rad: float = 0.0


    @classmethod
    def _normalize_angles(cls, theta: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(theta), np.cos(theta))


    @classmethod
    def _angles_in_excluded_gap(cls, theta: np.ndarray, gap_start: float, gap_end: float) -> np.ndarray:
        if gap_start <= gap_end:
            return (theta >= gap_start) & (theta <= gap_end)
        return (theta >= gap_start) | (theta <= gap_end)


    @classmethod
    def _ccw_angle_delta(cls, from_angle: float, to_angle: np.ndarray) -> np.ndarray:
        return (to_angle - from_angle) % (2.0 * np.pi)


    @classmethod
    def _excluded_gap_span_rad(cls, gap_start: float, gap_end: float) -> float:
        if gap_start <= gap_end:
            return float(gap_end - gap_start)
        return float((gap_end + 2.0 * np.pi) - gap_start)


@define(slots=False)
class ShapelyMaze:
    nodes: List[Tuple[float, float]] = field(default=Factory(list))
    linearization_mode: str = 'linestring'
    ring_params: Optional[CircularRingLinearizationParams] = None
    maze_track_line: LineString = field(default=None, init=False)

    def __attrs_post_init__(self):
        self.maze_track_line = LineString(self.nodes)

    def shapely_linearize_trajectory(self, df: pd.DataFrame):
        """
        Linearize trajectory points by projecting them onto a Shapely LineString.

        Args:
            df (pd.DataFrame): DataFrame with 'x' and 'y' columns.
            track_line (LineString): Shapely LineString representing the maze path.

        Returns:
            pd.Series: Linearized positions (distance along the track_line).
        """
        points = df.apply(lambda row: Point(row['x'], row['y']), axis=1)
        linear_positions = [self.maze_track_line.project(p) for p in points]
        return pd.Series(linear_positions, index=df.index)


    def linearize_trajectory(self, df: pd.DataFrame) -> pd.Series:
        if self.linearization_mode == 'angular_ring':
            return self.angular_ring_linearize_trajectory(df)
        return self.shapely_linearize_trajectory(df)


    def angular_ring_linearize_trajectory(self, df: pd.DataFrame) -> pd.Series:
        if self.ring_params is None:
            raise ValueError("angular_ring linearization requires ring_params on ShapelyMaze.")
        p = self.ring_params
        x = df['x'].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
        finite_xy = np.isfinite(x) & np.isfinite(y)
        theta = p._normalize_angles(np.arctan2(y - p.center_y, x - p.center_x))
        has_gap = (p.gap_angle_start_rad is not None) or (p.gap_angle_end_rad is not None)
        if has_gap:
            if (p.gap_angle_start_rad is None) or (p.gap_angle_end_rad is None):
                raise ValueError("angular_ring gap exclusion requires both gap_angle_start_rad and gap_angle_end_rad.")
            in_gap = p._angles_in_excluded_gap(theta, p.gap_angle_start_rad, p.gap_angle_end_rad)
            excluded_span = p._excluded_gap_span_rad(p.gap_angle_start_rad, p.gap_angle_end_rad)
            valid_span = (2.0 * np.pi) - excluded_span
            arc_origin = p.gap_angle_end_rad
        else:
            in_gap = np.zeros(len(df), dtype=bool)
            valid_span = 2.0 * np.pi
            arc_origin = p.angle_origin_rad
        delta = p._ccw_angle_delta(arc_origin, theta)
        if p.arc_direction.lower() != 'ccw':
            delta = (valid_span - delta) % valid_span
        lin = np.full(len(df), np.nan, dtype=float)
        off_ring = np.zeros(len(df), dtype=bool)
        if (p.radius_cm is not None) and (p.max_radius_deviation_cm is not None):
            r = np.hypot(x - p.center_x, y - p.center_y)
            off_ring = np.abs(r - p.radius_cm) > p.max_radius_deviation_cm
        valid = finite_xy & (~in_gap) & (~off_ring) & (delta <= valid_span)
        lo, hi = p.output_range
        lin[valid] = lo + (delta[valid] / valid_span) * (hi - lo)
        return pd.Series(lin, index=df.index)


    def compute_on_track_mask(self, x: np.ndarray, y: np.ndarray, max_track_distance_cm: float) -> np.ndarray:
        if self.linearization_mode == 'angular_ring' and self.ring_params is not None:
            p = self.ring_params
            finite_xy = np.isfinite(x) & np.isfinite(y)
            theta = p._normalize_angles(np.arctan2(y - p.center_y, x - p.center_x))
            on_track = finite_xy.copy()
            has_gap = (p.gap_angle_start_rad is not None) or (p.gap_angle_end_rad is not None)
            if has_gap:
                if (p.gap_angle_start_rad is None) or (p.gap_angle_end_rad is None):
                    raise ValueError("angular_ring gap exclusion requires both gap_angle_start_rad and gap_angle_end_rad.")
                on_track &= ~p._angles_in_excluded_gap(theta, p.gap_angle_start_rad, p.gap_angle_end_rad)
            if p.radius_cm is not None:
                r = np.hypot(x - p.center_x, y - p.center_y)
                on_track &= np.abs(r - p.radius_cm) <= max_track_distance_cm
            return on_track
        distances = np.array([self.maze_track_line.distance(Point(xi, yi)) for xi, yi in zip(x, y)])
        return distances <= max_track_distance_cm


@define(slots=False)
class ShapelyMazeCollection:
    shapelyMazes: Dict[str, ShapelyMaze] = field(default=Factory(dict))
    valid_epochs: Dict[str, Tuple[float, float]] = field(default=Factory(dict))    


# def resolve_shapely_valid_epochs(curr_active_pipeline, pos_df: pd.DataFrame, shapely_maze_collection: ShapelyMazeCollection, maze_epoch_keys: List[str], epochs_df: Optional[pd.DataFrame] = None, valid_epochs_override: Optional[Dict[str, Tuple[float, float]]] = None, min_position_samples: int = 100,
#         min_epoch_duration_sec: float = 60.0, min_on_track_fraction: float = 0.3, max_track_distance_cm: float = 25.0, enable_position_occupancy_refinement: bool = True, debug_print: bool = True) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, str]]:
#     """Resolve per-maze time bounds for shapely linearization with tiered fallbacks.

#     Priority per key: override -> session epochs -> occupancy refinement -> template fallback -> omit.
#     Returns (valid_epochs, provenance) where provenance values are one of:
#     'override', 'epochs', 'epochs_refined', 'occupancy', 'template_fallback', 'missing'.
#     """

#     def _subfn_extract_epoch_bounds_from_epochs_df(epochs_df: Optional[pd.DataFrame], label: str, start_col: str = 'start', stop_col: str = 'stop') -> Optional[Tuple[float, float]]:
#         """Return (start, stop) for a single epoch label, or None if missing."""
#         if epochs_df is None or len(epochs_df) == 0 or 'label' not in epochs_df.columns:
#             return None
#         label_rows = epochs_df[epochs_df['label'] == label]
#         if len(label_rows) == 0:
#             return None
#         if len(label_rows) > 1:
#             duration_col = 'duration' if 'duration' in label_rows.columns else None
#             if duration_col is not None:
#                 label_rows = label_rows.sort_values(by=duration_col, ascending=False)
#             else:
#                 label_rows = label_rows.copy()
#                 label_rows['_duration'] = label_rows[stop_col].astype(float) - label_rows[start_col].astype(float)
#                 label_rows = label_rows.sort_values(by='_duration', ascending=False)
#             if debug_print:
#                 print(f"resolve_shapely_valid_epochs: label {label!r} has {len(label_rows)} rows; using longest duration.")
#         row = label_rows.iloc[0]
#         return (float(row[start_col]), float(row[stop_col]))

#     def _subfn_compute_on_track_mask(pos_df: pd.DataFrame, shapely_maze: ShapelyMaze) -> np.ndarray:
#         """Boolean mask: True where sample is within max_track_distance_cm of the maze skeleton."""
#         return shapely_maze.compute_on_track_mask(pos_df['x'].to_numpy(), pos_df['y'].to_numpy(), max_track_distance_cm)

#     def _subfn_validate_shapely_epoch_bounds(pos_df: pd.DataFrame, shapely_maze: ShapelyMaze, t0: float, t1: float) -> bool:
#         """Return True if epoch bounds contain enough on-track position samples."""
#         if t1 <= t0:
#             return False
#         if (t1 - t0) < min_epoch_duration_sec:
#             return False
#         window_df = pos_df[(pos_df['t'] >= t0) & (pos_df['t'] <= t1)].dropna(subset=['x', 'y'], how='any')
#         if len(window_df) < min_position_samples:
#             return False
#         on_track_mask = _subfn_compute_on_track_mask(window_df, shapely_maze)
#         on_track_fraction = float(np.mean(on_track_mask)) if len(on_track_mask) > 0 else 0.0
#         return on_track_fraction >= min_on_track_fraction

#     def _subfn_infer_epoch_bounds_from_track_occupancy(pos_df: pd.DataFrame, shapely_maze: ShapelyMaze, search_t0: float, search_t1: float) -> Optional[Tuple[float, float]]:
#         """Infer epoch bounds from the largest contiguous on-track occupancy segment within the search window."""
#         min_segment_samples = min(min_position_samples, 50)
#         min_segment_duration_sec = min(min_epoch_duration_sec, 30.0)
#         if search_t1 <= search_t0:
#             return None
#         window_df = pos_df[(pos_df['t'] >= search_t0) & (pos_df['t'] <= search_t1)].dropna(subset=['x', 'y', 't'], how='any')
#         if len(window_df) < min_segment_samples:
#             return None
#         window_df = window_df.sort_values('t').reset_index(drop=True)
#         on_track_mask = _subfn_compute_on_track_mask(window_df, shapely_maze)
#         if not np.any(on_track_mask):
#             return None
#         regions = contiguous_regions(on_track_mask)
#         if len(regions) == 0:
#             return None
#         best_region = None
#         best_n_samples = -1
#         t_values = window_df['t'].to_numpy()
#         for region in regions:
#             start_idx, end_idx = int(region[0]), int(region[1])
#             n_samples = end_idx - start_idx
#             if n_samples < min_segment_samples:
#                 continue
#             t_start, t_end = float(t_values[start_idx]), float(t_values[end_idx - 1])
#             if (t_end - t_start) < min_segment_duration_sec:
#                 continue
#             if n_samples > best_n_samples:
#                 best_n_samples = n_samples
#                 best_region = (t_start, t_end)
#         return best_region

#     # ==================================================================================================================================================================================================================================================================================== #
#     # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
#     # ==================================================================================================================================================================================================================================================================================== #
#     valid_epochs_override = valid_epochs_override or {}
#     template_valid_epochs = shapely_maze_collection.valid_epochs or {}
#     resolved_valid_epochs: Dict[str, Tuple[float, float]] = {}
#     provenance: Dict[str, str] = {}


#     # ==================================================================================================================================================================================================================================================================================== #
#     # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
#     # ==================================================================================================================================================================================================================================================================================== #
#     epochs_determined_by_sess_pos_df_dict = {k:(curr_active_pipeline.filtered_sessions[k].position.to_dataframe().dropna(subset=['x', 'y', 't'], how='any', inplace=False)['t'].min(), curr_active_pipeline.filtered_sessions[k].position.to_dataframe().dropna(subset=['x', 'y', 't'], how='any', inplace=False)['t'].max()) for k in maze_epoch_keys}
#     resolved_valid_epochs: Dict[str, Tuple[float, float]] = epochs_determined_by_sess_pos_df_dict

#     # pos_df = pos_df.dropna(subset=['x', 'y', 't'], how='any')
#     # if len(pos_df) == 0:
#     #     if debug_print:
#     #         print("resolve_shapely_valid_epochs: empty position dataframe; no bounds resolved.")
#     #     return resolved_valid_epochs, provenance
#     # session_t_min, session_t_max = float(pos_df['t'].min()), float(pos_df['t'].max())
#     # maze_keys = [k for k in maze_epoch_keys if k in shapely_maze_collection.shapelyMazes]
#     # missing_geometry_keys = set(maze_epoch_keys) - set(maze_keys)
#     # if missing_geometry_keys and debug_print:
#     #     print(f"resolve_shapely_valid_epochs: no shapely geometry for keys {sorted(missing_geometry_keys)}; skipping.")
#     # prior_maze_stop: Optional[float] = None

#     # for maze_key in maze_keys:
#     #     shapely_maze = shapely_maze_collection.shapelyMazes[maze_key]
#     #     bounds: Optional[Tuple[float, float]] = None
#     #     source: str = 'missing'
#     #     if maze_key in valid_epochs_override:
#     #         bounds = (float(valid_epochs_override[maze_key][0]), float(valid_epochs_override[maze_key][1]))
#     #         source = 'override'
#     #     epoch_bounds = _subfn_extract_epoch_bounds_from_epochs_df(epochs_df=epochs_df, label=maze_key)
#     #     search_t0 = epoch_bounds[0] if epoch_bounds is not None else session_t_min
#     #     search_t1 = epoch_bounds[1] if epoch_bounds is not None else session_t_max
#     #     if prior_maze_stop is not None:
#     #         search_t0 = max(search_t0, prior_maze_stop)
#     #     if bounds is None and epoch_bounds is not None:
#     #         t0, t1 = epoch_bounds
#     #         if prior_maze_stop is not None:
#     #             t0 = max(t0, prior_maze_stop)
#     #         if _subfn_validate_shapely_epoch_bounds(pos_df, shapely_maze, t0, t1):
#     #             bounds = (t0, t1)
#     #             source = 'epochs'
#     #         elif enable_position_occupancy_refinement:
#     #             occupancy_bounds = _subfn_infer_epoch_bounds_from_track_occupancy(pos_df, shapely_maze, search_t0=max(t0, search_t0), search_t1=t1)
#     #             if occupancy_bounds is not None and _subfn_validate_shapely_epoch_bounds(pos_df, shapely_maze, occupancy_bounds[0], occupancy_bounds[1]):
#     #                 bounds = occupancy_bounds
#     #                 source = 'epochs_refined'
#     #     if bounds is None and enable_position_occupancy_refinement:
#     #         occupancy_bounds = _subfn_infer_epoch_bounds_from_track_occupancy(pos_df, shapely_maze, search_t0=search_t0, search_t1=search_t1)
#     #         if occupancy_bounds is not None and _subfn_validate_shapely_epoch_bounds(pos_df, shapely_maze, occupancy_bounds[0], occupancy_bounds[1]):
#     #             bounds = occupancy_bounds
#     #             source = 'occupancy'
#     #     if bounds is None and epoch_bounds is not None:
#     #         # A session's own epoch-label bounds are more trustworthy than another session's hardcoded template times, even when on-track geometry validation is weak (e.g. RatU/RatJ reusing RatK/RatS maze geometry whose LineString does not match this session's track). Use them (unvalidated) before falling back to the cross-session template times.
#     #         t0, t1 = epoch_bounds
#     #         if prior_maze_stop is not None:
#     #             t0 = max(t0, prior_maze_stop)
#     #         if t1 > t0:
#     #             bounds = (t0, t1)
#     #             source = 'epochs_unvalidated'
#     #             if debug_print:
#     #                 print(f"resolve_shapely_valid_epochs: {maze_key} using session epoch bounds {bounds} (unvalidated; geometry validation failed but preferred over cross-session template fallback).")
#     #     if bounds is None and maze_key in template_valid_epochs:
#     #         template_bounds = (float(template_valid_epochs[maze_key][0]), float(template_valid_epochs[maze_key][1]))
#     #         t0, t1 = template_bounds
#     #         if prior_maze_stop is not None:
#     #             t0 = max(t0, prior_maze_stop)
#     #         if _subfn_validate_shapely_epoch_bounds(pos_df, shapely_maze, t0, t1):
#     #             bounds = (t0, t1)
#     #             source = 'template_fallback'
#     #     if bounds is not None:
#     #         resolved_valid_epochs[maze_key] = bounds
#     #         provenance[maze_key] = source
#     #         prior_maze_stop = bounds[1]
#     #         if debug_print:
#     #             print(f"resolve_shapely_valid_epochs: {maze_key} -> {bounds} (source={source})")
#     #     else:
#     #         provenance[maze_key] = 'missing'
#     #         if debug_print:
#     #             print(f"resolve_shapely_valid_epochs: {maze_key} -> MISSING (all tiers failed)")
#     return resolved_valid_epochs, provenance


def build_shapely_maze_collection_for_session(curr_active_pipeline, pos_df: pd.DataFrame, geometry_template: ShapelyMazeCollection, maze_epoch_keys: List[str], epochs_df: Optional[pd.DataFrame] = None, valid_epochs_override: Optional[Dict[str, Tuple[float, float]]] = None, min_position_samples: int = 100, min_epoch_duration_sec: float = 60.0, min_on_track_fraction: float = 0.3, max_track_distance_cm: float = 25.0, enable_position_occupancy_refinement: bool = True, debug_print: bool = True) -> ShapelyMazeCollection:
    """Build a session-specific ShapelyMazeCollection: geometry from template, valid_epochs resolved dynamically."""
    #TODO 2026-06-23 07:47: - [ ] Made much simplier
    assert curr_active_pipeline is not None
    # epochs_df[epochs_df['label'] == k]
    epochs_determined_by_sess_pos_df_dict = {k:(epochs_df[epochs_df['label'] == k]['start'].min(), epochs_df[epochs_df['label'] == k]['stop'].max()) for k in maze_epoch_keys}
    # epochs_determined_by_sess_pos_df_dict = {k:(curr_active_pipeline.filtered_sessions[k].position.to_dataframe().dropna(subset=['x', 'y', 't'], how='any', inplace=False)['t'].min(), curr_active_pipeline.filtered_sessions[k].position.to_dataframe().dropna(subset=['x', 'y', 't'], how='any', inplace=False)['t'].max()) for k in maze_epoch_keys}
    resolved_valid_epochs: Dict[str, Tuple[float, float]] = epochs_determined_by_sess_pos_df_dict

    # resolved_valid_epochs, provenance = resolve_shapely_valid_epochs(curr_active_pipeline, pos_df=pos_df, shapely_maze_collection=geometry_template, maze_epoch_keys=maze_epoch_keys, epochs_df=epochs_df, valid_epochs_override=valid_epochs_override, min_position_samples=min_position_samples, min_epoch_duration_sec=min_epoch_duration_sec, min_on_track_fraction=min_on_track_fraction, max_track_distance_cm=max_track_distance_cm, enable_position_occupancy_refinement=enable_position_occupancy_refinement, debug_print=debug_print)
    return ShapelyMazeCollection(shapelyMazes=deepcopy(geometry_template.shapelyMazes), valid_epochs=resolved_valid_epochs)


def linearize_position_df(pos_df: pd.DataFrame, sample_sec=3, method="isomap", sigma=2, override_position_sampling_rate_Hz=None, regularization_approach:RegularizationApproach=RegularizationApproach.RAW_VALUES,
                          all_session_mazes: Optional[ShapelyMazeCollection]=None, track_definition=None):
    """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.
    ISOMAP is more versatile but also more computationally expensive.

    Parameters
    ----------
    track_names: list of track names, each must match an epoch in epochs class.
    sample_sec : int, optional
        sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results
    method : str, optional
        by default 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020) or
        'PCA' (for straight tracks)
    override_position_sampling_rate_Hz: float, optional - ignored except for method="isomap". If provided, used for downsampling dataframe prior to computation. Otherwise sampling rate is approximated from pos_df['t'] column.
    track_definition: TrackDefinition instance or registry key (e.g. 'w_maze') when method='track_graph'.
    
    Modifies:
        Adds the 'lin_pos' column to the provided position dataframe.
    """
    if method.lower() == "track_graph":
        assert track_definition is not None, "track_definition must be provided when using method == 'track_graph'."
        if isinstance(track_definition, str):
            if track_definition == 'w_maze':
                from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import w_maze
                track_definition = w_maze
            else:
                raise ValueError(f"Unsupported track_definition key: {track_definition!r}")
        work_df = deepcopy(pos_df)
        valid_xy_mask = work_df[['x', 'y']].notna().all(axis='columns')
        work_df['lin_pos'] = np.nan
        if valid_xy_mask.any():
            linearized_subset = track_definition.get_linearized_position(position=work_df.loc[valid_xy_mask, ['x', 'y']].copy())
            work_df.loc[linearized_subset.index, 'lin_pos'] = linearized_subset['lin_pos'].to_numpy()
            for col_name in ['track_segment_id', 'track_projected_x_position', 'track_projected_y_position']:
                if col_name in linearized_subset.columns:
                    work_df.loc[linearized_subset.index, col_name] = linearized_subset[col_name].to_numpy()
            work_df.loc[linearized_subset.index, 'linearization_method'] = 'track_graph'
        return work_df

    pos_df = deepcopy(pos_df).dropna(subset=['x','y'], how='any')
    
    xy_pos = pos_df[['x','y']].to_numpy()
    
    xlinear = None
    if method.lower() == "pca":
        pca = PCA(n_components=1)
        xlinear = pca.fit_transform(xy_pos).squeeze()


    elif method.lower() == "isomap":
        imap = Isomap(n_neighbors=5, n_components=2)
        # downsample points to reduce memory load and time
        if override_position_sampling_rate_Hz is not None:
            position_sampling_rate_Hz = override_position_sampling_rate_Hz
        else:
            # compute sampling rate from the 't' column:
            assert 't' in pos_df.columns
            position_sampling_rate_Hz = 1.0 / np.nanmean(np.diff(pos_df['t'].to_numpy())) # In Hz, returns 29.969777
        num_end_samples = np.round(int(position_sampling_rate_Hz) * sample_sec)
        pos_ds = xy_pos[0:-1:num_end_samples]
        imap.fit(pos_ds)
        iso_pos = imap.transform(xy_pos)
        # Keep iso_pos here in case we want to use 2nd dimension (transverse to track) in future...
        if iso_pos.std(axis=0)[0] < iso_pos.std(axis=0)[1]:
            iso_pos[:, [0, 1]] = iso_pos[:, [1, 0]]
        xlinear = iso_pos[:, 0]
        

    elif method.lower() == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("UMAP method requires the 'umap-learn' library. Please install it via 'pip install umap-learn'.") from e

        # Downsample points for fitting, as in ISOMAP
        if override_position_sampling_rate_Hz is not None:
            position_sampling_rate_Hz = override_position_sampling_rate_Hz
        else:
            assert 't' in pos_df.columns
            position_sampling_rate_Hz = 1.0 / np.nanmean(np.diff(pos_df['t'].to_numpy()))
        num_end_samples = int(np.round(position_sampling_rate_Hz * sample_sec))
        pos_ds = xy_pos[::num_end_samples]
        t_ds = pos_df['t'].to_numpy()[::num_end_samples]
        t_all = pos_df['t'].to_numpy()

        reducer = umap.UMAP(
            n_neighbors=10,       # or tune as desired
            n_components=2,       # retain 2D manifold for possible future use; will use first dimension for linearization
            metric='euclidean',   # or tune if necessary
            random_state=1337,      # for reproducibility
            verbose=False
        )
        embedding_ds = reducer.fit_transform(pos_ds)

        # For continuity, you may want to flip axes based on variance as with ISOMAP
        if embedding_ds.std(axis=0)[0] < embedding_ds.std(axis=0)[1]:
            embedding_ds[:, [0, 1]] = embedding_ds[:, [1, 0]]

        # Use the first UMAP dimension as the linearized projection for now
        xlinear_ds = embedding_ds[:, 0]

        # Interpolate for all timepoints
        from scipy.interpolate import interp1d
        interp_func = interp1d(t_ds, xlinear_ds, kind='linear', fill_value="extrapolate", assume_sorted=True)
        xlinear = interp_func(t_all)

    elif method.lower() == "shapely":
        ## Added 2025-11-26 - Shapely uses user-defined track geometry (shapes) to properly linearize the 2D position in a manner way more efficient than UMAP (which exceeds memory bounds)
        assert all_session_mazes is not None, f"all_session_mazes must be provided (defining the maze/track geometry) when using method == 'shapely'."
        ## INPUTS: pos_df, all_session_mazes (with shapelyMazes and valid_epochs)
        piecewise_lin_pos = pd.Series(np.nan, index=pos_df.index, dtype=float)
        t_col = pos_df['t'].to_numpy()

        for track_maze_key, shapely_maze in all_session_mazes.shapelyMazes.items():
            # Get the valid epoch time bounds for this maze
            epoch_bounds = all_session_mazes.valid_epochs.get(track_maze_key, None)
            if epoch_bounds is None:
                continue
            maze_start_t, maze_end_t = epoch_bounds

            # Find rows where 't' falls within the maze's valid epoch
            time_mask = (t_col >= maze_start_t) & (t_col <= maze_end_t)
            if not np.any(time_mask):
                continue

            # Check if pre-computed linearized position column exists
            source_col_name = f'shapely_linearized_position_{track_maze_key}'
            if source_col_name in pos_df.columns:
                # Use pre-computed values
                piecewise_lin_pos.loc[time_mask] = pos_df.loc[time_mask, source_col_name]
            else:
                # Compute linearization on-the-fly using the ShapelyMaze
                track_pos_df = pos_df.loc[time_mask, ['x', 'y']]
                linearized_values = shapely_maze.linearize_trajectory(track_pos_df)
                piecewise_lin_pos.loc[time_mask] = linearized_values.values

        xlinear = piecewise_lin_pos.to_numpy()
        # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import bapun_proper_linearize_tracks
        # pos_df_dict, maze_track_line_dict = bapun_proper_linearize_tracks(curr_active_pipeline)

    else:
        print('ERROR: invalid method name: {}'.format(method))
        
    if (sigma is not None) and (sigma > 0.0):
        xlinear = gaussian_filter1d(xlinear, sigma=sigma) # smooth
        
    if regularization_approach.name == RegularizationApproach.SUBTRACT_MIN.name:
        xlinear -= np.min(xlinear) # required to prevent mapping to negative values
    elif regularization_approach.name == RegularizationApproach.RESTORE_X_RANGE.name:
        xlinear = -1.0 * xlinear # flip over the y-axis first
        lin_pos_bounds = compute_grid_bin_bounds(xlinear)[0]
        x_bounds = compute_grid_bin_bounds(pos_df['x'].to_numpy())[0]
        # print(f'lin_pos_bounds: {lin_pos_bounds}, x_bounds: {x_bounds}')
        xlinear = map_value(xlinear, lin_pos_bounds, x_bounds) # map xlinear from its current bounds range to the xbounds range
    else:
        assert regularization_approach.name == RegularizationApproach.RAW_VALUES.name, f"Invalid regularization approach!"
    pos_df['lin_pos'] = xlinear # add the linearized position to the dataframe as the 'lin_pos' column
    return pos_df


def linearize_position(position: core.Position, sample_sec=3, method="isomap", sigma=2, **kwargs) -> core.Position:
    """linearize trajectory. Use method='PCA' for off-angle linear track, method='ISOMAP' for any non-linear track.
    ISOMAP is more versatile but also more computationally expensive.

    Parameters
    ----------
    track_names: list of track names, each must match an epoch in epochs class.
    sample_sec : int, optional
        sample a point every sample_sec seconds for training ISOMAP, by default 3. Lower it if inaccurate results
    method : str, optional
        by default 'ISOMAP' (for any continuous track, untested on t-maze as of 12/22/2020) or
        'PCA' (for straight tracks)

    """
    if isinstance(position, pd.DataFrame):
        pos_df = linearize_position_df(position, sample_sec=sample_sec, method=method, sigma=sigma, override_position_sampling_rate_Hz=None, **kwargs)
        xlinear = pos_df['lin_pos'].to_numpy()
        return core.Position.from_separate_arrays(pos_df['t'].to_numpy(), xlinear, lin_pos=xlinear, metadata=None)
    else:
        pos_df = position.to_dataframe() # convert from a Position object
        pos_df = linearize_position_df(pos_df, sample_sec=sample_sec, method=method, sigma=sigma, override_position_sampling_rate_Hz=position.sampling_rate, **kwargs)
        xlinear = pos_df['lin_pos'].to_numpy()
        return core.Position.from_separate_arrays(position.time, xlinear, lin_pos=xlinear, metadata=position.metadata)

    
    


def calculate_run_direction(
    position: core.Position,
    speedthresh=(10, 20),
    merge_dur=2,
    min_dur=2,
    smooth_speed=50,
    min_dist=50,
):
    """Divide running epochs into forward and backward.
    Currently only works for one dimensional position data

    Parameters
    ----------
    speedthresh : tuple, optional
        low and high speed threshold for speed, by default (10, 20)
    merge_dur : int, optional
        two epochs if less than merge_dur (seconds) apart they will be merged , by default 2 seconds
    min_dur : int, optional
        minimum duration of a run epoch, by default 2 seconds
    smooth_speed : int, optional
        speed is smoothed, increase if epochs are fragmented, by default 50
    min_dist : int, optional
        the animal should cover this much distance in one direction within the lap to be included, by default 50
    plot : bool, optional
        plots the epochs with position and speed data, by default True
    """

    assert position.ndim == 1, "Run direction only supports one dimensional position"

    trackingsampling_rate = position.time
    posdata = position.to_dataframe()

    posdata = posdata[(posdata.time > period[0]) & (posdata.time < period[1])]
    x = posdata.linear
    time = posdata.time
    speed = posdata.speed
    speed = gaussian_filter1d(posdata.speed, sigma=smooth_speed)

    high_speed = threshPeriods(
        speed,
        lowthresh=speedthresh[0],
        highthresh=speedthresh[1],
        minDistance=merge_dur * trackingsampling_rate,
        minDuration=min_dur * trackingsampling_rate,
    )
    val = []
    for epoch in high_speed:
        displacement = x[epoch[1]] - x[epoch[0]]
        # distance = np.abs(np.diff(x[epoch[0] : epoch[1]])).sum()

        if np.abs(displacement) > min_dist:
            if displacement < 0:
                val.append(-1)
            elif displacement > 0:
                val.append(1)
        else:
            val.append(0)
    val = np.asarray(val)

    # ---- deleting epochs where animal ran a little distance------
    high_speed = np.delete(high_speed, np.where(val == 0)[0], axis=0)
    val = np.delete(val, np.where(val == 0)[0])

    high_speed = np.around(high_speed / trackingsampling_rate + period[0], 2)
    data = pd.DataFrame(high_speed, columns=["start", "stop"])
    # data["duration"] = np.diff(high_speed, axis=1)
    data["direction"] = np.where(val > 0, "forward", "backward")

    self.epochs = run_epochs

    return run_epochs


def calculate_run_epochs(
    position: core.Position,
    speedthresh=(10, 20),
    merge_dur=2,
    min_dur=2,
    smooth_speed=50,
):
    """Divide running epochs into forward and backward.
    Currently only works for one dimensional position data

    Parameters
    ----------
    speedthresh : tuple, optional
        low and high speed threshold for speed, by default (10, 20)
    merge_dur : int, optional
        two epochs if less than merge_dur (seconds) apart they will be merged , by default 2 seconds
    min_dur : int, optional
        minimum duration of a run epoch, by default 2 seconds
    smooth_speed : int, optional
        speed is smoothed, increase if epochs are fragmented, by default 50
    min_dist : int, optional
        the animal should cover this much distance in one direction within the lap to be included, by default 50
    plot : bool, optional
        plots the epochs with position and speed data, by default True
    """

    sampling_rate = position.sampling_rate

    x = position.x
    time = position.time
    speed = position.speed
    speed = gaussian_filter1d(position.speed, sigma=smooth_speed)

    high_speed = threshPeriods(
        speed,
        lowthresh=speedthresh[0],
        highthresh=speedthresh[1],
        minDistance=merge_dur * sampling_rate,
        minDuration=min_dur * sampling_rate,
    )
    val = []
    for epoch in high_speed:
        displacement = x[epoch[1]] - x[epoch[0]]
        # distance = np.abs(np.diff(x[epoch[0] : epoch[1]])).sum()

        if np.abs(displacement) > min_dist:
            if displacement < 0:
                val.append(-1)
            elif displacement > 0:
                val.append(1)
        else:
            val.append(0)
    val = np.asarray(val)

    # ---- deleting epochs where animal ran a little distance------
    high_speed = np.delete(high_speed, np.where(val == 0)[0], axis=0)
    val = np.delete(val, np.where(val == 0)[0])

    high_speed = np.around(high_speed / sampling_rate + period[0], 2)
    data = pd.DataFrame(high_speed, columns=["start", "stop"])
    # data["duration"] = np.diff(high_speed, axis=1)
    data["direction"] = np.where(val > 0, "forward", "backward")

    return run_epochs


def compute_position_grid_size(*any_1d_series, num_bins:tuple):
    """  Computes the required bin_sizes from the required num_bins (for each dimension independently)
    Usage:
    out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64))
    active_grid_bin = tuple(out_grid_bin_size)
    print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
    
    History:
        Extracted from pyphocorehelpers.indexing_helpers import compute_position_grid_size for use in BaseDataSessionFormats
    
    """
    assert (len(any_1d_series)) == len(num_bins), f'(len(other_1d_series)) must be the same length as the num_bins tuple! But (len(other_1d_series)): {(len(any_1d_series))} and len(num_bins): {len(num_bins)}!'
    num_series = len(num_bins)
    out_bins = []
    out_bins_info = []
    out_bin_grid_step_size = np.zeros((num_series,))

    for i in np.arange(num_series):
        xbins, xbin_info = compute_spanning_bins(any_1d_series[i], num_bins=num_bins[i])
        out_bins.append(xbins)
        out_bins_info.append(xbin_info)
        out_bin_grid_step_size[i] = xbin_info.step

    return out_bin_grid_step_size, out_bins, out_bins_info


# @function_attributes(short_name=None, tags=['downsample', 'trajectory', 'path', 'subsample', 'efficiency', 'position'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-21 06:48', related_items=[])
def simplify_position_trajectory(position_df: pd.DataFrame, epsilon: float = 0.5, algorithm: str = "rdp", algo: str = "iter") -> pd.Series:
    """Simplify position trajectories using line simplification algorithms to reduce plotted points while preserving path features.
    
    Uses the Ramer-Douglas-Peucker (RDP) algorithm by default to downsample position trajectories
    while maintaining the essential shape of the path. Returns a boolean mask that can be used to
    filter the original dataframe to the simplified trajectory. Supports both 2D (x, y) and 3D (x, y, z) positions.
    
    Parameters
    ----------
    position_df : pd.DataFrame
        DataFrame with ['x', 'y'] columns (required) and optionally ['z'] column for 3D positions.
        The function automatically detects if 'z' is present and processes 3D coordinates accordingly.
    epsilon : float, optional
        Simplification tolerance. Higher values result in more aggressive simplification (fewer points retained).
        Default is 0.5.
    algorithm : str, optional
        Algorithm choice. Currently only "rdp" is supported. Default is "rdp".
    algo : str, optional
        For RDP algorithm, use "iter" (iterative) to enable return_mask support. Default is "iter".
    
    Returns
    -------
    pd.Series
        Boolean mask of the same length as position_df. True indicates the point is kept in the
        simplified trajectory, False indicates it should be filtered out.
        The mask preserves the original dataframe index for proper filtering with `.loc[mask]`.
    
    Examples
    --------
    >>> position_df = ...  # DataFrame with ['x', 'y'] columns
    >>> mask = simplify_position_trajectory(position_df, epsilon=0.5)
    >>> simplified_df = position_df.loc[mask]
    
    >>> position_df_3d = ...  # DataFrame with ['x', 'y', 'z'] columns
    >>> mask = simplify_position_trajectory(position_df_3d, epsilon=0.5)
    >>> simplified_df = position_df_3d.loc[mask]
    """
    # Validate input
    if not isinstance(position_df, pd.DataFrame):
        raise TypeError(f"position_df must be a pandas DataFrame, got {type(position_df)}")
    
    required_columns = ['x', 'y']
    missing_columns = [col for col in required_columns if col not in position_df.columns]
    if missing_columns:
        raise ValueError(f"position_df must contain columns {required_columns}, missing: {missing_columns}")
    
    # Determine if 3D coordinates are present
    has_z = 'z' in position_df.columns
    coord_columns = ['x', 'y', 'z'] if has_z else ['x', 'y']
    
    # Handle edge cases
    if len(position_df) == 0:
        return pd.Series([], dtype=bool, index=position_df.index)
    
    if len(position_df) < 3:
        # RDP requires at least 3 points. For < 3 points, return all True
        return pd.Series([True] * len(position_df), dtype=bool, index=position_df.index)
    
    # Extract coordinates and handle NaN values
    xyz_pos = position_df[coord_columns].to_numpy()
    
    # Check for NaN values
    nan_mask = np.isnan(xyz_pos).any(axis=1)
    if nan_mask.all():
        # All points are NaN
        return pd.Series([False] * len(position_df), dtype=bool, index=position_df.index)
    
    # Filter out NaN points for RDP processing
    valid_mask = ~nan_mask
    valid_xyz = xyz_pos[valid_mask]
    
    if len(valid_xyz) < 3:
        # After filtering NaNs, we have < 3 valid points
        result_mask = pd.Series([False] * len(position_df), dtype=bool, index=position_df.index)
        result_mask[valid_mask] = True  # Keep the valid points
        return result_mask
    
    # Apply RDP algorithm
    if algorithm.lower() == "rdp":
        if algo != "iter":
            raise ValueError(f"algo must be 'iter' to enable return_mask support, got '{algo}'")
        
        # RDP with return_mask=True returns a boolean mask
        # RDP automatically handles 2D (n, 2) or 3D (n, 3) coordinate arrays
        rdp_mask = rdp(valid_xyz, epsilon=epsilon, algo=algo, return_mask=True)
        
        # Create full mask with same length as original dataframe
        result_mask = pd.Series([False] * len(position_df), dtype=bool, index=position_df.index)
        # Map the RDP mask back to the original positions (only for valid points)
        valid_indices = position_df.index[valid_mask]
        result_mask.loc[valid_indices] = rdp_mask
        
        return result_mask
    else:
        raise ValueError(f"Unsupported algorithm: '{algorithm}'. Currently only 'rdp' is supported.")


