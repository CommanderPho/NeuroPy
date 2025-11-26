from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.ndimage import gaussian_filter1d

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
class ShapelyMaze:
    nodes: List[Tuple[float, float]] = field(default=Factory(list))
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
        # Create Point objects from 'x' and 'y' columns
        # Using .apply() with a lambda function for potentially better performance than a list comprehension on large DFs
        points = df.apply(lambda row: Point(row['x'], row['y']), axis=1)

        # Project each point onto the LineString and get the distance along it
        linear_positions = [self.maze_track_line.project(p) for p in points]
        return pd.Series(linear_positions, index=df.index)
    

@define(slots=False)
class ShapelyMazeCollection:
    shapelyMazes: Dict[str, ShapelyMaze] = field(default=Factory(dict))
    valid_epochs: Dict[str, Tuple[float, float]] = field(default=Factory(dict))    



def linearize_position_df(pos_df: pd.DataFrame, sample_sec=3, method="isomap", sigma=2, override_position_sampling_rate_Hz=None, regularization_approach:RegularizationApproach=RegularizationApproach.RAW_VALUES,
                          all_session_mazes: Optional[ShapelyMazeCollection]=None):
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
    
    Modifies:
        Adds the 'lin_pos' column to the provided position dataframe.
    """
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
                linearized_values = shapely_maze.shapely_linearize_trajectory(track_pos_df)
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

