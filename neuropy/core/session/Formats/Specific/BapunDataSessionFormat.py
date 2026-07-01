 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## typehinting only imports here
    from neuropy.core.position import Position

from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Callable, Union, Any
from pathlib import Path
from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatBaseRegisteredClass
from neuropy.core.session.dataSession import DataSession
from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec

# For specific load functions:
from neuropy.core import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains, ClusterlessSpikeEvents, default_clusterless_spike_events_path, load_clusterless_spike_events
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters # used particularly to build Bapun-style filters
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import HardcodedProcessingParameters
from neuropy.utils.position_util import ShapelyMaze, ShapelyMazeCollection, CircularRingLinearizationParams, build_shapely_maze_collection_for_session
from shapely import box ## used by `build_Bapun_Day4OpenField_laps_from_reward_zones`
# from shapely.geometry import LineString, Point 


# ## compute valid_epochs via ```
# valid_epochs = {}
# for a_maze_name, a_plotter in sync_plotters.items():
#     pos_df = curr_active_pipeline.filtered_sessions[a_maze_name].position.to_dataframe()
#     valid_epochs[a_maze_name] = (pos_df['t'].min(), pos_df['t'].max())

# valid_epochs
# ```

# linearization_method: str = 'umap'

# linearization_method: str = 'shapely'
# Geometry templates for TwoNovel shapely linearization. valid_epochs are RatS/RatK reference bounds used only as fallback when dynamic resolution fails (see build_shapely_maze_collection_for_session).
Day5TwoNovel_all_session_mazes: ShapelyMazeCollection = ShapelyMazeCollection(shapelyMazes = {
    # Define the skeletons (re-using the coordinates identified earlier)
    # "N"-shaped maze
    'maze1': ShapelyMaze(nodes = [
        (-66.31, 88.82),  # TL
        (-64.57, -54.62), # BL
        (25.0, 65.0),  # TR
        (73.88, -59.85)   # BR
    ]),
    # "U"-shaped maze
    'maze2': ShapelyMaze(nodes = [
        (-48.62, 63.79),  # Top-Left
        (-33.99, -41.77), # Bot-Left
        (3.32, -49.23),   # Bot-Mid
        (37.16, -34.00),  # Bot-Right
        (52.74, 76.34)    # Top-Right
    ]),
},
    valid_epochs =  {'maze1': (11070.0, 13970.0), 'maze2': (20756.0, 24004.0)}, # fallback only (RatS Day5TwoNovel reference); resolved per-session at linearization time - 'maze_GLOBAL': (0.0, 42305.0), 
)


# RatK/U-style TwoNovel geometry template; valid_epochs are fallback bounds only (RatK Day3TwoNovel reference).
RatK_Day3TwoNovel_all_session_mazes: ShapelyMazeCollection = ShapelyMazeCollection(shapelyMazes = {
    # Define the skeletons (re-using the coordinates identified earlier)
    'maze1': ShapelyMaze(nodes=[
        (-88.0,  58.0),   # TL — top-left hub
        (-66.0, -54.0),   # BL — bottom-left hub
        ( 26.0,  86.0),   # TR — top-center/right hub
        (108.0, -18.0),   # BR — bottom-right hub
    ]),
    # Blue = maze2 ("U"-shaped): TL -> BL -> BM -> BR -> TR
    'maze2': ShapelyMaze(nodes=[
        (-135.35897298, 58.64455010),   # Top-Left  (left arm top; gap in ring is between TL and TR)
        (-29.46830373, -62.98060411),   # Bot-Left
        (-17.93761552, 101.89932855),   # Top-Mid   (tp[most point)
        (128.88129774, -43.31796770),   # Bot-Right
    ]),
},
    valid_epochs =  {'maze1': (8950.11800747698, 12441.993530592174), 'maze2': (23253.12015019876, 26010.99556335396)}, # fallback only (RatK Day3TwoNovel reference); resolved per-session at linearization time - 'maze_GLOBAL': (0.0, 42305.0), 
)


# 
RatJ_Day3TwoNovel_all_session_mazes: ShapelyMazeCollection = ShapelyMazeCollection(shapelyMazes = {
    # Zig-zag / "N" maze — path order follows the track skeleton
    'maze1': ShapelyMaze(nodes=[
        (-117.68,  333.81),  # Top hub
        ( 237.35,  -90.93),  # Right-middle hub
        ( 105.74, -412.40),  # Bottom-right hub
        (-368.46,  142.26),  # Left-middle hub
    ]),
    # Ring maze — LineString runs along the ring; gap is between last node and first
    'maze2': ShapelyMaze(
        nodes=[(149.84, 64.67), (187.69, -154.38), (111.17, 102.19), (-205.84, 116.18)],
        linearization_mode='angular_ring',
        ring_params=CircularRingLinearizationParams(center_x=-30.0, center_y=-155.0, 
                        gap_angle_start_rad=np.deg2rad(-6), gap_angle_end_rad=np.deg2rad(4), arc_direction='ccw', ## PERFECT nearly perfect (gap_angle_start_rad: -6, gap_angle_start_rad: 4)
                        output_range=(-435.0, 320.0)),
    ),
},
    valid_epochs = {'maze1': (8950.11800747698, 12441.993530592174), 'maze2': (23253.12015019876, 26010.99556335396)}, # fallback only (RatK Day3TwoNovel reference); resolved per-session at linearization time - 'maze_GLOBAL': (0.0, 42305.0), 
)



def plot_shapely_maze(grid_bin_bounds: Tuple[Tuple[float, float], Tuple[float, float]], reward_zones_dict: Dict[str, Polygon], position_df: Optional[pd.DataFrame]=None, ax=None):
    """ plots the generic shapely maze extracted from the hardcoded parameters for this session

    Usage:
    
        from neuropy.core.session.Formats.BaseDataSessionFormats import HardcodedProcessingParameters
        from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass, plot_shapely_maze

        hardcoded_params: HardcodedProcessingParameters = BapunDataSessionFormatRegisteredClass._get_session_specific_parameters(session_context=curr_active_pipeline.get_session_context())
        active_reward_zones_dict = hardcoded_params.lap_estimation_parameters['reward_zones'](curr_active_pipeline.filtered_sessions['roam'])
        pos_df: pd.DataFrame = curr_active_pipeline.filtered_sessions['roam'].position.to_dataframe()[['t', 'x', 'y']] ## only care about x, y columns really

        _out = plot_shapely_maze(grid_bin_bounds=hardcoded_params.grid_bin_bounds, 
                                reward_zones_dict=active_reward_zones_dict,
                                position_df=pos_df,
                                ax=None)
                                
    """
    import matplotlib.pyplot as plt
    from shapely import box, Polygon
    from shapely.geometry import LineString, Point
    from shapely.plotting import plot_polygon, patch_from_polygon

    grid_bin_bounds_box = None
    if not isinstance(grid_bin_bounds, Polygon):
        ## convert to a polygon
        if len(grid_bin_bounds) == 2:
            # min_tuple, max_tuple = grid_bin_bounds ## unpack
            # xmin, ymin = min_tuple
            # xmax, ymax = max_tuple
            x_tuple, y_tuple = grid_bin_bounds ## unpack
            xmin, xmax = x_tuple
            ymin, ymax = y_tuple
            grid_bin_bounds_box = box(xmin, ymin, xmax, ymax)
        elif len(grid_bin_bounds) == 4:
            xmin, ymin, xmax, ymax = grid_bin_bounds ## unpack directly
            grid_bin_bounds_box = box(xmin, ymin, xmax, ymax)
            
        else:
            raise ValueError(f'grid_bin_bounds is unexpected format: {grid_bin_bounds}')
    else:
        grid_bin_bounds_box = grid_bin_bounds ## already a box or Polygon object
                
            
    _out = {'maze': None, 'reward_zones': {}, 'pos_line': None}

    ## Plot maze sections:
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        _out['fig'] = fig
        _out['ax'] = ax
    else:
        _out['ax'] = ax
            
    _out['maze'] = plot_polygon(grid_bin_bounds_box, ax=ax, color='darkgrey', add_points=False)
    # perform_update_title_subtitle(

    if reward_zones_dict is not None:
        for k, a_zone in reward_zones_dict.items():
            _out['reward_zones'][k] = plot_polygon(a_zone, ax=ax, color='orange', add_points=False)


    if position_df is not None:
        assert len(position_df) > 0
        assert 'x' in position_df.columns
        assert 'y' in position_df.columns
        _out['pos_line'] = _out['ax'].plot(position_df['x'], position_df['y'], alpha=0.2)

    return _out



class BapunDataSessionFormatRegisteredClass(DataSessionFormatBaseRegisteredClass):
    """

    # Example Filesystem Hierarchy:
    📦Day5TwoNovel
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.eeg
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.flattened.spikes.npy <-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.maze1.linear.npy     <-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.maze2.linear.npy     <-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.mua.npy              <-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.neurons.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.nrs
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.paradigm.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.pbe.npy              <-OPT-GEN
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.position.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.probegroup.npy
     ┣ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.ripple.npy           <-OPT-GEN
     ┗ 📜RatS-Day5TwoNovel-2020-12-04_07-55-09.xml
    
    
    By default it attempts to find the single *.xml file in the root of this basedir, from which it determines the `session_name` as the stem (the part before the extension) of this file:
        basedir: Path('R:\data\Bapun\Day5TwoNovel')
        session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'
    
    From here, a list of known files to load from is determined:
        
    Usage:
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
        from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass

        _test_session = BapunDataSessionFormatRegisteredClass.build_session(Path('R:\data\Bapun\Day5TwoNovel'))
        _test_session, loaded_file_record_list = BapunDataSessionFormatRegisteredClass.load_session(_test_session)
        _test_session
        
    """
    _session_class_name = 'bapun'
    _session_default_relative_basedir = r'data/Bapun/Day5TwoNovel'
    _session_default_basedir = r'W:\data\Bapun\Day5TwoNovel' # WINDOWS
    # _session_default_basedir = r'/run/media/halechr/MoverNew/data/Bapun/Day5TwoNovel'
    _session_basepath_to_context_parsing_keys = ['format_name', 'animal', 'session_name']

    _time_variable_name = 't_seconds' # It's 't_rel_seconds' for kdiba-format data for example or 't_seconds' for Bapun-format data

    
    @classmethod
    def _get_session_specific_parameters(cls, session_context: IdentifyingContext) -> HardcodedProcessingParameters:
        """ session-specific type parameters 
         
        #TODO 2025-09-20 19:26: - [ ] Is this redudndant with preprocessing parameters?
        """
        
        bapun_open_field_grid_bin_bounds = (((-120.0, 120.0), (-120.0, 120.0)))



        # Custom Reward Zone Functions _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        def _subfn_rat_N_Day4Openfield_reward_zones(session) -> Dict[str, Polygon]:
            """ captures: None """
            xmin: float = -85.75619321393464
            xmax: float = 112.57838773103435
            ymin: float = -96.44772761274268
            ymax: float = 98.6220528078153
            # bapun_Day4OpenField_grid_bin_bounds = box(xmin, ymin, xmax, ymax)
            bapun_Day4OpenField_reward_zones = dict(    ## Define the two reward zones
                zone1 = box(xmin, 0.0, -60.0, 40.0),  # box(minx, miny, maxx, maxy, ccw=True)
                zone2 = box(80.0, 0.0, xmax, 40.0), # box(minx, miny, maxx, maxy, ccw=True)
            )
            return bapun_Day4OpenField_reward_zones



        def _subfn_rat_K_Day4Openfield_reward_zones(session) -> Dict[str, Polygon]:
            """ captures: None """
            xmin: float = -109.52
            xmax: float = 92.963
            ymin: float = -76.865
            ymax: float = 124.45
            bapun_Day4OpenField_reward_zones = dict(    ## Define the two reward zones
                zone1 = box(xmin, 0.0, -80.0, 50.0),  # box(minx, miny, maxx, maxy, ccw=True)
                zone2 = box(65.0, 0.0, xmax, 45.0), # box(minx, miny, maxx, maxy, ccw=True)
            )
            return bapun_Day4OpenField_reward_zones


        def _subfn_rat_S_Day1Openfield_reward_zones(session) -> Dict[str, Polygon]:
            """ captures: None """
            xmin: float = -109.52
            xmax: float = 92.963
            ymin: float = -76.865
            ymax: float = 124.45
            bapun_OpenField_reward_zones = dict(    ## Define the two reward zones
                zone1 = box(xmin, 0.0, -80.0, 50.0),  # box(minx, miny, maxx, maxy, ccw=True)
                zone2 = box(65.0, 0.0, xmax, 45.0), # box(minx, miny, maxx, maxy, ccw=True)
            )
            return bapun_OpenField_reward_zones


        def _subfn_rat_S_Day4Openfield_reward_zones(session) -> Dict[str, Polygon]:
            """ captures: None """
            return _subfn_rat_S_Day1Openfield_reward_zones(session=session) ## keep the same for RatS between Day1 and Day4 open field
            # xmin: float = -109.52
            # xmax: float = 92.963
            # ymin: float = -76.865
            # ymax: float = 124.45
            # bapun_OpenField_reward_zones = dict(    ## Define the two reward zones
            #     zone1 = box(xmin, 0.0, -80.0, 50.0),  # box(minx, miny, maxx, maxy, ccw=True)
            #     zone2 = box(65.0, 0.0, xmax, 45.0), # box(minx, miny, maxx, maxy, ccw=True)
            # )
            # return bapun_OpenField_reward_zones


        # bapun_open_field_grid_bin_bounds_rat_U = (((0.0, 142.0), (0.0, 30.0)))
        bapun_open_field_grid_bin_bounds_rat_U = (((0.0, 142.0), (-35.0, 175.0)))
        
        def _subfn_rat_U_Day5OpenfieldSD_reward_zones(session) -> Dict[str, Polygon]:
            """ captures: None """
            # xmin: float = -109.52
            # xmax: float = 92.963
            ymin: float = -35.0
            # ymax: float = 30.0 ## 30 is liberal, but 20.0 is more than enough
            ymax: float = 175.0

            # bapun_OpenField_reward_zones = dict(    ## Define the two reward zones
            #     zone1 = box(-np.inf, ymin, 30.0, ymax), # box(minx, miny, maxx, maxy, ccw=True) - Left Extrema
            #     zone2 = box(60.0, ymin, 85.0, ymax), # box(minx, miny, maxx, maxy, ccw=True)- Mid Extrema
            #     zone3 = box(120.0, ymin, np.inf, ymax), # box(minx, miny, maxx, maxy, ccw=True)- Right Extrema
            # )

            bapun_OpenField_reward_zones = dict(    ## Define the two reward zones
                zone1 = box(64.0, ymin, 84.0, -6.0), # box(minx, miny, maxx, maxy, ccw=True) - Left Extrema
                zone2 = box(60.0, 148.0, 75.0, ymax), # box(minx, miny, maxx, maxy, ccw=True)- Mid Extrema
                zone3 = box(118.0, 146.0, 140.0, ymax), # box(minx, miny, maxx, maxy, ccw=True)- Right Extrema
            )
            
            return bapun_OpenField_reward_zones



        bapun_grid_bin_bounds_rat_U_Day3TwoNovel = (((-120.0, 120.0), (-100.0, 120.0)))
        
        def _subfn_rat_U_Day3TwoNovel_reward_zones(session) -> Dict[str, Polygon]:
            """ captures: None """
            ## maze1
            bapun_OpenField_reward_zones_maze1 = dict(    ## Define the two reward zones
                zone1 = box(10.82278481, 22.03218646, 34.74683544, 62.94006659), # box(minx, miny, maxx, maxy, ccw=True) - Left Extrema
                zone2 = box(-19.55696203, 21.49944506, 3.6919831199999997, 62.94006659), # box(minx, miny, maxx, maxy, ccw=True)- Mid Extrema
            )

            bapun_OpenField_reward_zones_maze2 = dict(    ## Define the two reward zones
                zone1 = box(-113.97079277, 36.68257492, -66.61335188000001, 75.99223086), # box(minx, miny, maxx, maxy, ccw=True) - Left Extrema
                zone2 = box(66.94714882, -56.54716981, 145.34770515, 13.927857939999996), # box(minx, miny, maxx, maxy, ccw=True)- Mid Extrema
            )

            bapun_OpenField_reward_zones = {
                'maze1': bapun_OpenField_reward_zones_maze1,
                'maze2': bapun_OpenField_reward_zones_maze2,
            }

            # a_pos = deepcopy(session.position)
            sess_epoch_labels = session.epochs.labels
            if len(sess_epoch_labels) == 1:
                active_label: str = sess_epoch_labels[0]
                return bapun_OpenField_reward_zones[active_label]
            else:        
                return bapun_OpenField_reward_zones ## return the dict of dicts

        
        # 2026-06-23 04:03: - [X] Create for RatJ Day3TwoNovel
        bapun_open_field_RatJ_Day3TwoNovel_grid_bin_bounds = (((-435.0, 320.0), (-508.0, 430.0))) # #TODO 2026-06-23 04:02: - [ ] Created from image

        def _subfn_rat_J_Day3TwoNovel_reward_zones(session) -> Dict[str, Polygon]:
            """ captures: None """
            # maze1 ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # zone1 = box(-99.12051042, 103.68067387, 56.47279662, 401.69644642),  # box(minx, miny, maxx, maxy, ccw=True) [LineROI bbox]
            # zone2 = box(-358.89498119, -92.16315536, -74.92780048, 4.15648888),  # box(minx, miny, maxx, maxy, ccw=True) [LineROI bbox]
            # zone3 = box(-357.55593752, -228.61655016, -99.21250738, -9.02163126),  # box(minx, miny, maxx, maxy, ccw=True) [LineROI bbox]
            # zone4 = box(-149.74757351, 242.94310917, 23.56244256, 427.20760057),  # box(minx, miny, maxx, maxy, ccw=True)
            # zone5 = box(107.53999920, -283.17517004, 305.57308823, -79.92225482),  # box(minx, miny, maxx, maxy, ccw=True)
            # zone6 = box(-433.01532458, -128.95211812, -240.64225088, 76.83618038),  # box(minx, miny, maxx, maxy, ccw=True)
            # zone7 = box(-2.66624123, -505.94180058, 188.66228564, -299.91009164),  # box(minx, miny, maxx, maxy, ccw=True)

            bapun_reward_zones_maze1 = dict(    ## Define the two reward zones
                zone1 = box(-149.74757351, 242.94310917, 23.56244256, 427.20760057), # box(minx, miny, maxx, maxy, ccw=True) - Left Extrema
                zone2 = box(-2.66624123, -505.94180058, 188.66228564, -299.91009164), # box(minx, miny, maxx, maxy, ccw=True)- Mid Extrema
            )

            # maze2 ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # zone1 = box(-302.83370969, -251.47244444, 166.03490015, 199.82232874),  # box(minx, miny, maxx, maxy, ccw=True) [EllipseROI bbox]
            # zone2 = box(120.03109112, -115.45836275, 255.52874219, -49.62497088),  # box(minx, miny, maxx, maxy, ccw=True)
            # zone3 = box(64.99205260, -251.85335703, 242.99603212, -178.66040707),  # box(minx, miny, maxx, maxy, ccw=True)

            bapun_reward_zones_maze2 = dict(    ## Define the two reward zones
                zone1 = box(120.03109112, -115.45836275, 255.52874219, -49.62497088), # box(minx, miny, maxx, maxy, ccw=True) - Left Extrema
                zone2 = box(64.99205260, -251.85335703, 242.99603212, -178.66040707), # box(minx, miny, maxx, maxy, ccw=True)- Mid Extrema
            )

            bapun_OpenField_reward_zones = {
                'maze1': bapun_reward_zones_maze1,
                'maze2': bapun_reward_zones_maze2,
            }

            # a_pos = deepcopy(session.position)
            sess_epoch_labels = session.epochs.labels
            if len(sess_epoch_labels) == 1:
                active_label: str = sess_epoch_labels[0]
                return bapun_OpenField_reward_zones[active_label]
            else:        
                return bapun_OpenField_reward_zones ## return the dict of dicts



        # Custom Lap Building Functions ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        def _subfn_rat_N_Day4Openfield_build_Bapun_Day4OpenField_laps_from_reward_zones(session):
            """ captures: cls, _subfn_rat_N_Day4Openfield_reward_zones """
            bapun_Day4OpenField_reward_zones = _subfn_rat_N_Day4Openfield_reward_zones(session=session)
            return cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=bapun_Day4OpenField_reward_zones)

        def _subfn_rat_K_Day4Openfield_build_Bapun_Day4OpenField_laps_from_reward_zones(session):
            """ captures: cls, _subfn_rat_K_Day4Openfield_reward_zones """
            bapun_Day4OpenField_reward_zones = _subfn_rat_K_Day4Openfield_reward_zones(session=session)
            return cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=bapun_Day4OpenField_reward_zones)

        def _subfn_rat_S_Day1Openfield_build_Bapun_laps_from_reward_zones(session):
            """ captures: cls, _subfn_rat_K_Day4Openfield_reward_zones """
            bapun_Day4OpenField_reward_zones = _subfn_rat_S_Day1Openfield_reward_zones(session=session)
            return cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=bapun_Day4OpenField_reward_zones)

        def _subfn_rat_S_Day4Openfield_build_Bapun_laps_from_reward_zones(session):
            """ captures: cls, _subfn_rat_K_Day4Openfield_reward_zones """
            bapun_Day4OpenField_reward_zones = _subfn_rat_S_Day4Openfield_reward_zones(session=session)
            return cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=bapun_Day4OpenField_reward_zones)

        def _subfn_rat_U_Day4Openfield_build_Bapun_Day5OpenfieldSD_laps_from_reward_zones(session):
            """ captures: cls, _subfn_rat_U_Day5OpenfieldSD_reward_zones """
            bapun_OpenField_reward_zones = _subfn_rat_U_Day5OpenfieldSD_reward_zones(session=session)
            return cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=bapun_OpenField_reward_zones)

        # lambda session: cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=_subfn_rat_U_Day5OpenfieldSD_reward_zones(session=session))


        the_dict: Dict[IdentifyingContext, HardcodedProcessingParameters]  = { #  
            # Open Field Section _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # VALIDATE RatJ Day4Openfield: clone RatK Day4Openfield (single-maze open field, pre/maze/post); reuses _subfn_rat_K_Day4Openfield_reward_zones + _subfn_rat_K_Day4Openfield_build_Bapun_Day4OpenField_laps_from_reward_zones; batch context ProcessBatchOutputs_Bapun_Batch.ipy OpenField list
            IdentifyingContext(format_name= 'bapun', animal= 'RatJ', session_name= 'Day4Openfield'): HardcodedProcessingParameters(
                decoder_building_session_names=['maze'],
                global_session_name='maze',
                non_global_activity_session_names=['maze'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_K_Day4Openfield_reward_zones, custom_lap_estimation_fn=_subfn_rat_K_Day4Openfield_build_Bapun_Day4OpenField_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatK', session_name= 'Day4Openfield'): HardcodedProcessingParameters(
                # decoder_building_session_names=['maze', 'sprinkle', 'maze_GLOBAL'],
                decoder_building_session_names=['maze'],
                # global_session_name='maze_GLOBAL',
                global_session_name='maze',
                # non_global_activity_session_names=['maze', 'sprinkle'],
                non_global_activity_session_names=['maze'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_K_Day4Openfield_reward_zones, custom_lap_estimation_fn=_subfn_rat_K_Day4Openfield_build_Bapun_Day4OpenField_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatN', session_name= 'Day4OpenField'): HardcodedProcessingParameters(decoder_building_session_names=['roam', 'sprinkle', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['roam', 'sprinkle'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_N_Day4Openfield_reward_zones, custom_lap_estimation_fn=_subfn_rat_N_Day4Openfield_build_Bapun_Day4OpenField_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=20.0, merging_adjacent_max_separation_sec=6.0,),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatS', session_name='Day1Openfield'): HardcodedProcessingParameters(decoder_building_session_names=['maze'],
                global_session_name='maze',
                non_global_activity_session_names=['maze'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_S_Day1Openfield_reward_zones, custom_lap_estimation_fn=_subfn_rat_S_Day1Openfield_build_Bapun_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatS', session_name='Day4Openfield'): HardcodedProcessingParameters(decoder_building_session_names=['maze'],
                global_session_name='maze',
                non_global_activity_session_names=['maze'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_S_Day4Openfield_reward_zones, custom_lap_estimation_fn=_subfn_rat_S_Day4Openfield_build_Bapun_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatU', session_name= 'RatUDay5OpenfieldSD'): HardcodedProcessingParameters(
                # decoder_building_session_names=['maze', 'sprinkle', 'maze_GLOBAL'],
                decoder_building_session_names=['roam', 'sprinkle', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['roam', 'sprinkle'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds_rat_U,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_U_Day5OpenfieldSD_reward_zones, custom_lap_estimation_fn=_subfn_rat_U_Day4Openfield_build_Bapun_Day5OpenfieldSD_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatU', session_name= 'Day5OpenfieldSD'): HardcodedProcessingParameters( ## #TODO 2026-05-06 06:11: - [ ] Note this is a purposeful repeat of the above entry, but for the renamed session_name (without "RatU" part. 
                decoder_building_session_names=['roam', 'sprinkle', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['roam', 'sprinkle'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds_rat_U,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_U_Day5OpenfieldSD_reward_zones, custom_lap_estimation_fn=_subfn_rat_U_Day4Openfield_build_Bapun_Day5OpenfieldSD_laps_from_reward_zones, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),


            # Two Novel Section __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #



            # VALIDATE RatJ Day3TwoNovel: clone RatS Day5TwoNovel (N/U shapely mazes); reuses RatJ_Day3TwoNovel_all_session_mazes (valid_epochs are RatS-specific — may need tuning); maze1/maze2/maze_GLOBAL epochs; notebook InteractivePipelineLoadFromPickle_Bapun_RatJ_D3TwoNovel
            IdentifyingContext(format_name= 'bapun', animal= 'RatJ', session_name= 'Day3TwoNovel'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=bapun_open_field_RatJ_Day3TwoNovel_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_J_Day3TwoNovel_reward_zones,
                                                custom_lap_estimation_fn=None, 
                                                # custom_lap_estimation_fn=lambda session: cls.build_Bapun_OpenField_laps_from_reward_zones(session=session, bapun_OpenField_reward_zones=_subfn_rat_U_Day5OpenfieldSD_reward_zones(session=session)), 
                                                use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='shapely', all_session_mazes=RatJ_Day3TwoNovel_all_session_mazes),
                # linearization_parameters=dict(method='umap', all_session_mazes=RatJ_Day3TwoNovel_all_session_mazes),
            ),

            IdentifyingContext(format_name= 'bapun', animal= 'RatK', session_name= 'Day3TwoNovel'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=bapun_grid_bin_bounds_rat_U_Day3TwoNovel,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_U_Day3TwoNovel_reward_zones, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='shapely', all_session_mazes=RatK_Day3TwoNovel_all_session_mazes), ## TODO: Annotate maze using shapely
                # linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),
            IdentifyingContext(format_name= 'bapun', animal= 'RatS', session_name= 'Day5TwoNovel'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds, #TODO 2026-06-21 08:13: - [ ] why using the openfield bounds instead of the TwoNovel bounds?
                lap_estimation_parameters=dict(reward_zones=None, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='shapely', all_session_mazes=Day5TwoNovel_all_session_mazes),
            ),
            # VALIDATE RatU Day3TwoNovel: clone RatK Day3TwoNovel; reuses _subfn_rat_U_Day3TwoNovel_reward_zones + RatK_Day3TwoNovel_all_session_mazes + bapun_grid_bin_bounds_rat_U_Day3TwoNovel (shapely valid_epochs are RatK-specific — may need tuning)
            IdentifyingContext(format_name= 'bapun', animal= 'RatU', session_name= 'Day3TwoNovel'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=bapun_grid_bin_bounds_rat_U_Day3TwoNovel,
                lap_estimation_parameters=dict(reward_zones=_subfn_rat_U_Day3TwoNovel_reward_zones, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0),
                linearization_parameters=dict(method='shapely', all_session_mazes=RatK_Day3TwoNovel_all_session_mazes),
            ),

            ## Fallback defaults:
            IdentifyingContext(format_name= 'bapun'): HardcodedProcessingParameters(decoder_building_session_names=['maze1', 'maze2', 'maze_GLOBAL'],
                global_session_name='maze_GLOBAL',
                non_global_activity_session_names=['maze1', 'maze2'],
                grid_bin_bounds=bapun_open_field_grid_bin_bounds,
                lap_estimation_parameters=dict(reward_zones=None, custom_lap_estimation_fn=None, use_full_2D_lap_estimation=True, minimum_epoch_duration = 2.5, minimum_run_speed=10.0, merging_adjacent_max_separation_sec=6.0,),
                linearization_parameters=dict(method='umap', all_session_mazes=None),
            ),									
        }

        best_match = IdentifyingContext.matching(the_dict, criteria=session_context.get_subset(subset_includelist=cls._session_basepath_to_context_parsing_keys).to_dict())
        return list(best_match.values())[0] ## return the first match


    @classmethod
    def get_session_name(cls, basedir):
        """ returns the session_name for this basedir, which determines the files to load. """
        # Find the only .xml file to obtain the session name
        return DataSessionFormatBaseRegisteredClass.find_session_name_from_sole_xml_file(basedir) # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'


    @classmethod
    def build_session_basedirs_dict(cls, global_data_root_parent_path, debug_print=False) -> Dict[IdentifyingContext, Path]:
        """Session ctx → folder under ``<global_data_root_parent_path>/Bapun``. Tries ``Animal/<on_disk_folder>`` then flat ``<on_disk_folder>`` (some drives use ``R:\\data\\Bapun\\Day5TwoNovel`` style layouts)."""
        if not isinstance(global_data_root_parent_path, Path):
            global_data_root_parent_path = Path(global_data_root_parent_path).resolve()
        bapun_root = global_data_root_parent_path.joinpath('Bapun')
        fmt = cls._session_class_name
        # (animal, logical session_name as in IdentifyingContext, optional on-disk folder name if it differs)
        session_specs: List[Tuple[str, str, Optional[str]]] = [
            ('RatJ', 'Day4Openfield', None),
            ('RatK', 'Day4Openfield', None),
            ('RatN', 'Day4OpenField', None),
            ('RatS', 'Day1OpenField', ('Day1OpenField', 'Day1Openfield')),
            ('RatS', 'Day1Openfield', ('Day1OpenField', 'Day1Openfield')),
            ('RatS', 'Day4OpenField', ('Day4OpenField','Day4Openfield')),
            # ('RatS', 'Day4Openfield', None),
            ('RatU', 'Day5Openfield', ('Day5Openfield', 'Day5OpenfieldSD', 'RatUDay5OpenfieldSD')),
            ('RatU', 'Day5OpenfieldSD', ('Day5Openfield', 'Day5OpenfieldSD', 'RatUDay5OpenfieldSD')),
            ('RatU', 'RatUDay5OpenfieldSD', ('Day5Openfield', 'Day5OpenfieldSD', 'RatUDay5OpenfieldSD')), 
            # VALIDATE path: on-disk folder is RatJ/Day3TwoNovel (not RatJDay3TwoNovel); session_name in IdentifyingContext stays Day3TwoNovel
            ('RatJ', 'Day3TwoNovel', ('Day3TwoNovel', 'RatJDay3TwoNovel')), # ('RatJ', 'Day3TwoNovel', 'RatJDay3TwoNovel'),
            ('RatK', 'Day3TwoNovel', None),
            ('RatS', 'Day5TwoNovel', None),
            ('RatU', 'Day3TwoNovel', ('Day3TwoNovel', 'RatUDay3TwoNovel')),
            # ('RatU', 'Day3TwoNovel', None),
        ]
        out: Dict[IdentifyingContext, Path] = {}
        for animal, session_name, disk_folder in session_specs:
            ctx = IdentifyingContext(format_name=fmt, animal=animal, session_name=session_name)
            disk_folder = disk_folder or session_name
            

            def _subfn_try_find_path(a_disk_folder): 
                """ captures: bapun_root, animal """
                nested = bapun_root.joinpath(animal, a_disk_folder)
                flat = bapun_root.joinpath(a_disk_folder)
                is_valid: bool = False
                if nested.is_dir():
                    chosen = nested
                    is_valid = True
                elif flat.is_dir():
                    chosen = flat
                    is_valid = True
                else:
                    chosen = nested

                return (chosen, is_valid)
            
                
                                
            is_valid: bool = False
            if isinstance(disk_folder, (List, Tuple, Set)):
                
                for a_folder in disk_folder:
                    if not is_valid:
                        chosen, is_valid = _subfn_try_find_path(a_disk_folder=a_folder)
                        if is_valid:
                            out[ctx] = chosen.resolve() ## add the context
                            if debug_print:
                                print(f'Bapun build_session_basedirs_dict: found valid folder for {ctx}; using default path {chosen}')
                        else:
                            if debug_print:
                                print(f'Bapun build_session_basedirs_dict: no extant folder for {ctx}; using default path {chosen}')
                            continue
                        
            else:
                chosen, is_valid = _subfn_try_find_path(a_disk_folder=disk_folder)
                if is_valid:
                    out[ctx] = chosen.resolve()
                else:
                    if debug_print:
                        print(f'Bapun build_session_basedirs_dict: no extant folder for {ctx}; using default path {chosen}')
                        
        ## END for animal, session_name, disk_folder in session_specs....
        
        return out


    @classmethod
    def get_session_spec(cls, session_name):
        return SessionFolderSpec(required_files=[SessionFileSpec('{}.xml', session_name, 'The primary .xml configuration file', cls._load_xml_file),
                                           SessionFileSpec('{}.probegroup.npy', session_name, 'The numpy data file containing information about the spatial layout of recording probes', cls._load_probegroup_file),
                                           SessionFileSpec('{}.position.npy', session_name, 'The numpy data file containing the recorded animal positions (as generated by optitrack) over time.', cls._load_position_file),
                                           SessionFileSpec('{}.paradigm.npy', session_name, 'The numpy data file containing the recording epochs. Each epoch is defined as a: (label:str, t_start: float (in seconds), t_end: float (in seconds))', cls._load_paradigm_file)],
                    optional_files=[SessionFileSpec('{}.neurons.npy', session_name, 'The numpy data file containing information about neural activity.', cls._load_neurons_file),
                                    SessionFileSpec('{}.clusterless_spikes.npz', session_name, 'The NPZ data file containing clusterless spike events.', cls._load_clusterless_spike_events_file)],
                    additional_validation_requirements=[lambda basedir: cls._has_required_spike_source(Path(basedir), session_name)]
                    )


    @classmethod
    def _has_required_spike_source(cls, session_basedir: Path, session_name: str) -> bool:
        session_basedir = Path(session_basedir)
        return (session_basedir / f'{session_name}.neurons.npy').exists() or default_clusterless_spike_events_path(session_basedir, session_name).exists()


    ### Specific Load Functions used in the session_spec
    @classmethod
    def _load_neurons_file(cls, filepath, session): # .neurons
        ## here we need to handle the "1" type cells
        """
            neuron_type
                array(['pyr', 'pyr', 'pyr', 'mua', '1', 'inter', 'pyr', '1', 'pyr', 'pyr', '1', '1', '1', 'pyr', 'mua', 'mua', 'pyr', '1', 'pyr', 'mua', 'mua', 'pyr', '1', 'mua', '1', 'pyr', 'pyr', 'pyr', 'mua', 'pyr', 'pyr', 'pyr', 'pyr', '1', '1', 'pyr', 'pyr', '1', 'pyr', 'mua', '1', '1', '1', '1', '1', 'inter', 'pyr', 'mua', '1', 'pyr', 'pyr', 'pyr', 'mua', 'pyr', 'pyr', 'inter', 'pyr', 'pyr', 'pyr', '1', 'pyr', 'pyr', 'pyr', 'pyr', 'pyr', '1', 'pyr', '1', 'pyr', 'pyr', '1', 'pyr', 'mua', 'pyr', 'mua', '1', 'pyr'], dtype='<U5')
            neuron_type[neuron_type == '1']
                array(['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], dtype='<U5')
            neuron_type[neuron_type == '1'] = 'mua'

        """
        session.neurons = Neurons.from_file(filepath)

        return session


    @classmethod
    def _load_clusterless_spike_events_file(cls, filepath, session):
        events: ClusterlessSpikeEvents = load_clusterless_spike_events(filepath)
        events.filename = Path(filepath)
        print(f'Loading success: {Path(filepath).name}.')
        session.clusterless_spike_events = events
        return session

    @classmethod
    def _load_probegroup_file(cls, filepath, session): # .probegroup
        session.probegroup = ProbeGroup.from_file(filepath)
        return session

    @classmethod
    def _load_position_file(cls, filepath, session): # .position
        session.position = Position.from_file(filepath)

        ## the traces ['x', 'y'] should have the highest range in them, with 'z' being minimal. If 'z' has higheer variance than 'y' and the session identifier is matching, swap them before continuing
        """
        # Can check with:
            pos_df = curr_active_pipeline.filtered_sessions['maze_GLOBAL'].position.df.copy()
            pos_df.plot(x='t', y=['x', 'y', 'z'])
        """
        pos_df: pd.DataFrame = session.position.df.copy()

        if 'z' in pos_df.columns:
            y_ptp: float = np.ptp(pos_df['y'].to_numpy())
            z_ptp: float = np.ptp(pos_df['z'].to_numpy())
            needs_variable_swap: bool = (z_ptp > y_ptp)

            if needs_variable_swap:
                ## swap 'y' and 'z' columns
                pos_df = pos_df.rename(columns={'y': 'z', 'z':'y'}, inplace=False)
                session.position.df = pos_df ## assignment okay or need private member?
                # session.position._df = pos_df ## assignment okay or need private member?

                ## perform save differences to file if possible
                session.position.filename = session.filePrefix.with_suffix('.position.npy')
                session.position.save()

        return session


    @classmethod
    def _load_paradigm_file(cls, filepath, session): # .paradigm
        session.paradigm = Epoch.from_file(filepath)  # "epoch" field of file
        return session
    

    @classmethod
    def _subfn_FIXUP_Bapun_RatU_paradigm_epoch_times(cls, curr_paradigm_df: pd.DataFrame, start_col: str = 'start', stop_col: str = 'stop') -> pd.DataFrame:
        """ 2025-05-06 - fixes epoch times given manually observed corrections 
        """

        roam_row_idx: int = np.where(curr_paradigm_df['label'] == 'roam')[0][0]
        sprinkle_row_idx: int = np.where(curr_paradigm_df['label'] == 'sprinkle')[0][0]

        ## 2026-05-06 - Corrected roam/sprinkle start and end times
        corrected_paradigm_epoch_records = [
            ['roam', 8031, 10421],
            ['sprinkle', 10477, 11745],
        ]
        corrected_paradigm_epoch_df: pd.DataFrame = pd.DataFrame.from_records(corrected_paradigm_epoch_records, columns=['label', start_col, stop_col])
        # corrected_paradigm_epoch_df

        curr_paradigm_df.iat[roam_row_idx, 0] = corrected_paradigm_epoch_df[corrected_paradigm_epoch_df['label'] == 'roam'][start_col].iloc[0]
        curr_paradigm_df.iat[roam_row_idx, 1] = corrected_paradigm_epoch_df[corrected_paradigm_epoch_df['label'] == 'roam'][stop_col].iloc[0]

        curr_paradigm_df.iat[sprinkle_row_idx, 0] = corrected_paradigm_epoch_df[corrected_paradigm_epoch_df['label'] == 'sprinkle'][start_col].iloc[0]
        curr_paradigm_df.iat[sprinkle_row_idx, 1] = corrected_paradigm_epoch_df[corrected_paradigm_epoch_df['label'] == 'sprinkle'][stop_col].iloc[0]

        duration_col_name = 't_duration' if 't_duration' in curr_paradigm_df.columns else 'duration'
        curr_paradigm_df[duration_col_name] = curr_paradigm_df[stop_col] - curr_paradigm_df[start_col]

        curr_paradigm_df = curr_paradigm_df.reset_index(drop=True, inplace=False)
        # _time_column_name_synonyms = {"start":{'begin','start','start_t'},
        #     'stop':['end','stop','stop_t'],
        #     "t_duration":['duration'],
        # }
        # curr_paradigm_df = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(df=curr_paradigm_df, required_columns_synonym_dict=_time_column_name_synonyms)

        return curr_paradigm_df


    @classmethod
    def _bapun_session_fixup_epochs_to_be_non_overlapping(cls, curr_sess_context: IdentifyingContext, bapun_epochs: Epoch, enable_global_epoch: bool=True) -> Epoch:
        """ fixes up the loaded epochs
        has two conflicting (overlapping) epochs:
        "maze"
        "sprinkle"
        
        
                start   stop     label  duration
        0      0   7407       pre      7407
        1   7423  11483      maze      4060
        3  10186  11483  sprinkle      1297
        2  11497  25987      post     14490

        [4 rows x 4 columns]

        --- we want to compute the disjoint set 'roam', 'sprinkle'
        
        
        Usage:
        
        
        bapun_epochs: Epoch = deepcopy(curr_active_pipeline.sess.epochs)
        bapun_epochs = BapunDataSessionFormatRegisteredClass._bapun_session_fixup_epochs_to_be_non_overlapping(bapun_epochs=bapun_epochs)
        
        """
        from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch
        
        bapun_epochs_df: pd.DataFrame = ensure_dataframe(bapun_epochs) #.to_dataframe()
        bapun_params = cls._get_session_specific_parameters(session_context=curr_sess_context)

        is_bapun_Day4OpenField_sess: bool = False
        needs_update: bool = True
        if curr_sess_context is not None:
            is_bapun_Day1OpenField_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'session_name': 'Day1OpenField'}) ## all must match, 'animal': 'RatN'
            is_bapun_Day4OpenField_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'session_name': 'Day4OpenField'}) ## all must match, 'animal': 'RatN'
            # is_bapun_ratK_Day4OpenField_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'animal': 'RatK', 'session_name': 'Day4Openfield'}) ## all must match, 'animal': 'RatK'
            # VALIDATE epoch fixup: RatK + RatJ Day4Openfield share single-maze pre/maze/post (3 epochs); disable maze_GLOBAL when already fixed
            is_bapun_ratK_or_ratJ_Day4Openfield_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'animal': ['RatK', 'RatJ'], 'session_name': 'Day4Openfield'})
            # is_bapun_RatUDay5OpenfieldSD_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'session_name': 'RatUDay5OpenfieldSD'}) ## all must match, 'animal': 'RatN'
            # VALIDATE epoch fixup: RatU Day5 on-disk session; batch uses Day5Openfield context name alongside legacy Day5OpenfieldSD / RatUDay5OpenfieldSD aliases
            is_bapun_RatU_Day5OpenfieldSD_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'animal': 'RatU', 'session_name': ['RatUDay5OpenfieldSD', 'Day5OpenfieldSD', 'Day5Openfield']})
            is_bapun_RatS_Openfield_sess = curr_sess_context.query(criteria={'format_name':'bapun', 'animal': 'RatS', 'session_name': ['Day1OpenField', 'Day4Openfield', 'Day1OpenField', 'Day4OpenField']}) ## all must match, 'animal': 'RatS'

            if is_bapun_ratK_or_ratJ_Day4Openfield_sess:
                assert (len(bapun_epochs_df) == 3), f"{len(bapun_epochs_df)}"
                needs_update = not ((len(bapun_epochs_df) == 3) and (['pre', 'maze', 'post'] == bapun_epochs_df['label'].to_list()))
                if not needs_update:
                    print(f'overrinding enable_global_epoch == False since this session only has one maze epoch')
                    enable_global_epoch = False
                # assert (len(bapun_epochs_df) == 4), f"{len(bapun_epochs_df)}"
                # needs_update = (len(bapun_epochs_df) == 4) and ('roam' not in bapun_epochs_df['label'].to_list())
            elif (is_bapun_Day1OpenField_sess or is_bapun_RatS_Openfield_sess): ## is_bapun_RatSOpenfield_sess is sufficient
                #TODO 2026-06-03 03:10: - [ ] Fix the epochs
                assert (len(bapun_epochs_df) == 1), f"{len(bapun_epochs_df)}"
                print(f'overrinding enable_global_epoch == False since this session only has one maze epoch')
                enable_global_epoch = False

            elif is_bapun_Day4OpenField_sess:
                assert (len(bapun_epochs_df) == 4), f"{len(bapun_epochs_df)}"
                needs_update = (len(bapun_epochs_df) == 4) and ('roam' not in bapun_epochs_df['label'].to_list())
            elif is_bapun_RatU_Day5OpenfieldSD_sess:
                assert (len(bapun_epochs_df) >= 6), f"{len(bapun_epochs_df)}"
                needs_update = (len(bapun_epochs_df) >= 6) and ('roam' not in bapun_epochs_df['label'].to_list())


        # if ((is_bapun_Day4OpenField_sess or is_bapun_RatUDay5OpenfieldSD_sess) and needs_update):
        if (is_bapun_Day4OpenField_sess and needs_update and (not is_bapun_RatS_Openfield_sess)): ## Not RatS
            ## Applicable to Day4OpenField only: add the 'roam' row if it doesn't already exist
            # bapun_epochs_arr = bapun_epochs_df.to_numpy()
            # new_roam_row = [bapun_epochs_arr[1, 0], (bapun_epochs_arr[2, 0]-1), 'roam', 0.0] # ['start', 'stop', 'label', 'duration']
            # # bapun_epochs_arr[1, 1] = bapun_epochs_arr[2, 0] - 1 # overwrite the "maze" portion's end-time
            # fixed_bapun_epochs_arr = deepcopy(bapun_epochs_arr)
            # fixed_bapun_epochs_arr[1, :] = new_roam_row ## replace the entire 2nd row (the 'maze' row) with the new 'roam' row.
            # # build final epochs_df from fixed_bapun_epochs_arr
            # bapun_epochs_df: pd.DataFrame = pd.DataFrame(fixed_bapun_epochs_arr, columns=['start', 'stop', 'label', 'duration'])
            # bapun_epochs_df['label'] = bapun_epochs_df['label'].str.replace('maze', 'roam', n=1) ## rename 'maze' to 'roam'
            bapun_epochs_df['label'] = bapun_epochs_df['label'].map(lambda k: {'maze':'roam'}.get(k, k))  ## rename 'maze' to 'roam'
            bapun_epochs_df.loc[(bapun_epochs_df['label'] == 'roam'), 'stop'] = (bapun_epochs_df[bapun_epochs_df['label'] == 'sprinkle']['start'].item() - 1e-6)  # make sure 'roam' row doesn't overlap the 'sprinkle' row
            bapun_epochs_df[['start', 'stop', 'duration']] = bapun_epochs_df[['start', 'stop', 'duration']].astype(float)
            bapun_epochs_df['duration'] = bapun_epochs_df['stop'] - bapun_epochs_df['start'] ## recompute duration
            
        elif (is_bapun_RatU_Day5OpenfieldSD_sess and needs_update):
            
            # if not hasattr(curr_active_pipeline.sess, '_BAK_paradigm'):
            #     curr_active_pipeline.sess._BAK_paradigm = deepcopy(curr_active_pipeline.sess.paradigm) ## make backup of existing epochs/paradigm
                
            ## INPUTS: bapun_epochs_df
            # curr_paradigm_df: pd.DataFrame = bapun_epochs_df # ensure_dataframe(curr_active_pipeline.sess.paradigm)
            bapun_epochs_df = cls._subfn_FIXUP_Bapun_RatU_paradigm_epoch_times(curr_paradigm_df=bapun_epochs_df)
            bapun_epochs_df

            
            # Save/Set/Etc copied from notebook
            # did_change: bool = np.all(curr_active_pipeline.sess._BAK_paradigm != curr_paradigm_df)
            # # did_change: bool = np.all(curr_active_pipeline.sess.paradigm != curr_paradigm_df)

            # ## did_change
            # print(f'did_change: {did_change}')
            # if did_change:
                
            #     def get_resolved_paradigm_path(sess) -> Path:
            #         return sess.basepath.joinpath(f'{sess.session_name}.paradigm.npy').resolve()

            #     # paradigm_path = getattr(curr_active_pipeline.sess.paradigm, "filename", None)
            #     fn = getattr(curr_active_pipeline.sess.paradigm, "filename", None)
            #     paradigm_path = Path(fn).resolve() if fn is not None else get_resolved_paradigm_path(curr_active_pipeline.sess)
            #     ## just replace path
            #     print(f'paradigm_path: "{paradigm_path.as_posix()}"')

            #     backup_path = paradigm_path.with_suffix(paradigm_path.suffix + ".pre_edit.bak")  # e.g. *.npy.pre_edit.bak
            #     if paradigm_path.is_file() and not backup_path.is_file():
            #         print(f'creating backup of original paradigm file: "{paradigm_path}" -> "{backup_path}"...')
            #         shutil.copy2(paradigm_path, backup_path)
            #         print(f'\tdone.')

            #     # ==================================================================================================================================================================================================================================================================================== #
            #     # overwrite the old value:                                                                                                                                                                                                                                                             #
            #     # ==================================================================================================================================================================================================================================================================================== #
            #     curr_active_pipeline.sess.paradigm = curr_paradigm_df
            #     # curr_active_pipeline.sess.epochs = curr_paradigm_df

            #     ## INPUTS: paradigm_path
            #     # paradigm_path = Path(r"...") / f"{curr_active_pipeline.sess.session_name}.paradigm.npy"  # or sess.config.resolved path to that file
            #     modified = Epoch(curr_paradigm_df.copy())  # validates/normalizes via Epoch ctor
            #     modified.metadata = getattr(curr_active_pipeline.sess.paradigm, "metadata", None)
            #     modified.filename = paradigm_path.resolve()    
            #     modified.save(status_print=True)
                

        if enable_global_epoch:
            # maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            if 'maze_GLOBAL' not in bapun_epochs_df['label']:
                assert len(bapun_params.non_global_activity_session_names) >= 2, f"bapun_params.non_global_activity_session_names: {bapun_params.non_global_activity_session_names}"
                bapun_epochs_df = bapun_epochs_df.epochs.adding_global_epoch_row(global_epoch_name='maze_GLOBAL',
                                             first_included_epoch_name=bapun_params.non_global_activity_session_names[0], last_included_epoch_name=bapun_params.non_global_activity_session_names[-1], inplace=False)

        return Epoch(bapun_epochs_df, metadata=bapun_epochs.metadata)



    @classmethod
    def session_fixup_epochs(cls, sess, override_session_epochs: Optional[Epoch]=None, enable_global_epoch: bool=True, override_extant: bool=True, **kwargs) -> Epoch:
        """ fixes up the loaded epochs
        has two conflicting (overlapping) epochs:
        "maze"
        "sprinkle"
        
        
                start   stop     label  duration
        0      0   7407       pre      7407
        1   7423  11483      maze      4060
        3  10186  11483  sprinkle      1297
        2  11497  25987      post     14490

        [4 rows x 4 columns]

        --- we want to compute the disjoint set 'roam', 'sprinkle'
        
        
        Usage:

        session_epochs: Epoch = BapunDataSessionFormatRegisteredClass.session_fixup_epochs(sess=curr_active_pipeline.sess)
        """
        # curr_sess_context = sess.get_session_context()
        curr_sess_context = sess.get_context()
        
        # is_bapun_Day4OpenField_sess: bool = curr_sess_context.query(criteria={'format_name':'bapun', 'animal': 'RatN', 'session_name': 'Day4OpenField'}) ## all must match
        updated_epochs: Epoch = deepcopy(sess.epochs)
        needs_update: bool = False        

        if (not hasattr(sess, 'epochs_bak')):
            print(f'fixing up session computation epochs...')
            sess.epochs_bak = deepcopy(sess.epochs) ## backup the bad ones
            needs_update = True
        else:
            print(f'WARN: already fixedup session epochs.')
            if override_extant:
                print(f'\trestoring backed up epochs:')
                sess.epochs = deepcopy(sess.epochs_bak)
                needs_update = True


        if needs_update:
            if override_session_epochs is None:
                override_session_epochs = deepcopy(sess.epochs)
            
            updated_epochs = cls._bapun_session_fixup_epochs_to_be_non_overlapping(curr_sess_context=curr_sess_context, bapun_epochs=override_session_epochs, enable_global_epoch=enable_global_epoch, **kwargs)
            sess.epochs = updated_epochs
            print(f'\tdone. new epochs: \n{updated_epochs}\n')

            
        return updated_epochs


    @classmethod
    def build_shapely_maze_collection_for_session(cls, sess, hardcoded_params: HardcodedProcessingParameters, valid_epochs_override: Optional[Dict[str, Tuple[float, float]]] = None, debug_print: bool = True) -> ShapelyMazeCollection:
        """Build session-specific ShapelyMazeCollection: geometry from hardcoded template, valid_epochs resolved from sess.epochs with fallbacks."""
        linearization_parameters = hardcoded_params.linearization_parameters or {}
        geometry_template = linearization_parameters.get('all_session_mazes', None)
        assert geometry_template is not None, "hardcoded_params.linearization_parameters must include all_session_mazes for shapely linearization."
        if valid_epochs_override is None:
            valid_epochs_override = linearization_parameters.get('valid_epochs_override', None)
        return build_shapely_maze_collection_for_session(pos_df=sess.position.to_dataframe(), geometry_template=geometry_template, maze_epoch_keys=hardcoded_params.non_global_activity_session_names, epochs_df=sess.epochs.to_dataframe(), valid_epochs_override=valid_epochs_override, debug_print=debug_print)



    
    # Not limited:
    @classmethod
    def build_filters_any_epochs(cls, sess, filter_name_suffix=None):
        return build_custom_epochs_filters(sess, filter_name_suffix=filter_name_suffix)
    
    # Any epoch on the maze, not limited to pyramidal cells, etc
    @classmethod
    def build_filters_any_maze_epochs(cls, sess, filter_name_suffix=None, override_epoch_name_includelist: Optional[List[str]]=None):
        all_epochs_names: List[str] = sess.epochs.to_dataframe()['label'].to_list()
        if override_epoch_name_includelist is not None:
            ## use `override_epoch_name_includelist`
            maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem in override_epoch_name_includelist, names))
            assert (len(maze_only_name_filter_fn(all_epochs_names)) > 0), f"could not match any of the epochs in override_epoch_name_includelist: {override_epoch_name_includelist} session name formats. Need an override!\n\tall_epochs_names: {all_epochs_names}"

        else:
            maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.startswith('maze'), names))
            if len(maze_only_name_filter_fn(all_epochs_names)) == 0:
                ## no epochs starting with "maze", assume it's the other type of session
                maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem in ['roam', 'sprinkle'], names))
                assert (len(maze_only_name_filter_fn(all_epochs_names)) > 0), f"could not match either 'maze*' or ['roam', 'sprinkle'] session name formats. Need an override!\n\tall_epochs_names: {all_epochs_names}"


        maze_only_filters = build_custom_epochs_filters(sess, epoch_name_includelist=maze_only_name_filter_fn, filter_name_suffix=filter_name_suffix)
        return maze_only_filters


    @classmethod
    def build_default_filter_functions(cls, sess, epoch_name_includelist=None, filter_name_suffix=None, include_global_epoch=False):
        ## TODO: currently hard-coded
        # active_session_filter_configurations = cls.build_filters_any_epochs(sess)
        active_session_filter_configurations = cls.build_filters_any_maze_epochs(sess, filter_name_suffix=filter_name_suffix, override_epoch_name_includelist=epoch_name_includelist)
        return active_session_filter_configurations
    
        
    #######################################################
    ## Bapun Nupy Format Only Methods:
    # @classmethod
    # def _default_compute_flattened_spikes(cls, session, timestamp_scale_factor=(1/1E4), spike_timestamp_column_name='t_seconds', progress_tracing=True):
    #     spikes_df = FlattenedSpiketrains.build_spike_dataframe(session, timestamp_scale_factor=timestamp_scale_factor, spike_timestamp_column_name=spike_timestamp_column_name, progress_tracing=progress_tracing)
    #     print(f'spikes_df.columns: {spikes_df.columns}')
    #     session.flattened_spiketrains = FlattenedSpiketrains(spikes_df, time_variable_name=spike_timestamp_column_name, t_start=session.neurons.t_start) # FlattenedSpiketrains(spikes_df)
    #     print('\t Done!')
    #     return session
    
    # @classmethod
    # def _add_missing_spikes_df_columns(cls, spikes_df, neurons_obj):
    #     spikes_df, neurons_obj._reverse_cellID_index_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
    #     spikes_df['t'] = spikes_df[cls._time_variable_name] # add the 't' column required for visualization


    @classmethod
    def _get_clusterless_spike_events_path(cls, session) -> Path:
        session_name: str = cls._get_session_name(session)
        return default_clusterless_spike_events_path(Path(session.filePrefix).parent, session_name)


    @classmethod
    def _get_neurons_file_path(cls, session) -> Path:
        return Path(session.filePrefix).parent / f'{cls._get_session_name(session)}.neurons.npy'


    @classmethod
    def _get_session_name(cls, session) -> str:
        return getattr(getattr(session, 'config', None), 'session_name', None) or getattr(session, 'name', None) or Path(session.filePrefix).name


    @classmethod
    def _try_load_clusterless_spike_events_file(cls, session):
        clusterless_save_path: Path = cls._get_clusterless_spike_events_path(session)
        if clusterless_save_path.exists() and getattr(session, 'clusterless_spike_events', None) is None:
            session = cls._load_clusterless_spike_events_file(clusterless_save_path, session)
        return session


    @classmethod
    def _try_load_optional_spike_sources(cls, session, loaded_file_record_list=None):
        if loaded_file_record_list is None:
            loaded_file_record_list = []
        neurons_path: Path = cls._get_neurons_file_path(session)
        if neurons_path.exists() and (neurons_path not in loaded_file_record_list) and (getattr(session, 'neurons', None) is None):
            session = cls._load_neurons_file(neurons_path, session)
            loaded_file_record_list.append(neurons_path)
        clusterless_save_path: Path = cls._get_clusterless_spike_events_path(session)
        if clusterless_save_path.exists() and (clusterless_save_path not in loaded_file_record_list):
            session = cls._load_clusterless_spike_events_file(clusterless_save_path, session)
            loaded_file_record_list.append(clusterless_save_path)
        return session, loaded_file_record_list
        
    
    ## Main load function:
    @classmethod
    def load_session(cls, session, debug_print=False, enable_continue_on_required_path_failure: bool=False):
        """ allow to fail or return incomplete.
        """
        session, loaded_file_record_list = DataSessionFormatBaseRegisteredClass.load_session(session, debug_print=debug_print) # call the super class load_session(...) to load the common things (.recinfo, .filePrefix, .eegfile, .datfile)
        remaining_required_filespecs = {k: v for k, v in session.config.resolved_required_filespecs_dict.items() if k not in loaded_file_record_list}
        if debug_print:
            print(f'remaining_required_filespecs: {remaining_required_filespecs}')
        
        for file_path, file_spec in remaining_required_filespecs.items():
            print(f'loading file_path: "{file_path}"...')
            if not enable_continue_on_required_path_failure:
                session = file_spec.session_load_callback(file_path, session)
                loaded_file_record_list.append(file_path)
            else:
                try:
                    session = file_spec.session_load_callback(file_path, session)
                    loaded_file_record_list.append(file_path)
                except Exception as e:
                    print(f'e: {e} but enable_continue_on_required_path_failure == True so continuing...')


        session, loaded_file_record_list = cls._try_load_optional_spike_sources(session, loaded_file_record_list=loaded_file_record_list)


        # ==================================================================================================================================================================================================================================================================================== #
        # Define subfunctions that capture and modify most just for conveninece                                                                                                                                                                                                                #
        # ==================================================================================================================================================================================================================================================================================== #

        def _perform_position_comps(session):
            # ['.neurons.npy','.probegroup.npy','.position.npy','.paradigm.npy']
            # session = DataSessionLoader.__default_compute_bapun_flattened_spikes(session)
            # Load or compute linear positions if needed:        
            if (not session.position.has_linear_pos):
                # compute linear positions:
                print(f'computing linear positions for all active epochs ({session.epochs}) for session...')
                # end result will be session.computed_traces of the same length as session.traces in terms of frames, with all non-maze times holding NaN values
                session.position.linear_pos = np.full_like(session.position.time, np.nan)
                
                # ['pre', 'maze', 'sprinkle', 'post']

                # only_included_pos_computation_labels tries to work around a memory error            
                # NOTE: WARNING: DataSession.compute_linearized_position(session, an_epoch_label) causes MemoryErrors when called for an_epoch_label that doesn't have position data. For Bapun's r'W:\Data\Bapun\RatS\Day5TwoNovel' data, ['maze1', 'maze2'] were the acceptable epochs (and they were hardcoded).
                # I encountered the MemoryError when I ran the same load_session function for r'W:\Data\Bapun\RatN\Day4OpenField', which has the epochs ['pre', 'maze', 'sprinkle', 'post']. Restricting these computations to ['maze'] solves the problem for this session. It seems like I could just detect if there are any position samples left in the filtered position dataframe within the DataSession.compute_linearized_position(session, an_epoch_label), and handle the error there if there aren't (which I believe would prevent the MemoryError and excessive computation).
                
                #TODO 2025-12-17 12:16: - [ ] Needs generalization for the other session names (e.g. 'roam', )
                only_included_pos_computation_labels = ['maze']
                all_epoch_labels = list(session.epochs.labels)
                
                if 'sprinkle' in all_epoch_labels:
                    only_included_pos_computation_labels = ['maze', 'sprinkle']

                else:
                    only_included_pos_computation_labels = ['maze']

                # only_included_pos_computation_labels = None
                if only_included_pos_computation_labels is None:
                    only_included_pos_computation_labels = session.epochs.labels # all labels if no restrictions are specified
                    

                for an_epoch_label in session.epochs.labels:
                    if an_epoch_label in only_included_pos_computation_labels:
                        an_epoch_timeslice_indicies, active_positions_maze1, linearized_positions_curr_epoch = DataSession._perform_compute_session_linearized_position(session, an_epoch_label, method='umap')
                        session.position.linear_pos[an_epoch_timeslice_indicies] = linearized_positions_curr_epoch.traces
                    

                ## Previous 'manual' maze1 and maze2 way that fails for any sessions without these epochs:    
                # acitve_epoch_timeslice_indicies1, active_positions_maze1, linearized_positions_maze1 = DataSession.compute_linearized_position(session, 'maze1')
                # acitve_epoch_timeslice_indicies2, active_positions_maze2, linearized_positions_maze2 = DataSession.compute_linearized_position(session, 'maze2')
                # session.position.linear_pos[acitve_epoch_timeslice_indicies1] = linearized_positions_maze1.traces
                # session.position.linear_pos[acitve_epoch_timeslice_indicies2] = linearized_positions_maze2.traces
                
                session.position.filename = session.filePrefix.with_suffix(".position.npy")
                # print('Saving updated position results to {}...'.format(session.position.filename))
                with ProgressMessagePrinter(session.position.filename, action='Saving', contents_description='updated position results'):
                    session.position.save()
                # print('done.\n')
            else:
                print('linearized position loaded from file.')


        def _perform_spike_comps(session):
            ## Load or compute flattened spikes since this format of data has the spikes ordered only by cell_id:
            ## flattened.spikes:
            active_file_suffix = '.flattened.spikes.npy'
            # active_file_suffix = '.new.flattened.spikes.npy'
            found_datafile = FlattenedSpiketrains.from_file(session.filePrefix.with_suffix(active_file_suffix))
            if found_datafile is not None:
                print('Loading success: {}.'.format(active_file_suffix))
                session.flattened_spiketrains = found_datafile
            else:
                # Otherwise load failed, perform the fallback computation
                print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
                if getattr(session, 'neurons', None) is None:
                    session = cls._try_load_clusterless_spike_events_file(session)
                    if getattr(session, 'clusterless_spike_events', None) is not None:
                        print('No neurons loaded; skipping flattened spiketrains recompute because clusterless spike events are available.')
                        return
                    raise ValueError(f'Cannot compute {active_file_suffix}: missing session.neurons and no clusterless spike events loaded.')
                session = cls._default_compute_flattened_spikes(session, spike_timestamp_column_name=cls._time_variable_name) # sets session.flattened_spiketrains
            
                ## Testing: Fixing spike positions
                spikes_df = session.spikes_df
                session, spikes_df = cls._default_compute_spike_interpolated_positions_if_needed(session, spikes_df, time_variable_name=cls._time_variable_name)
                cls._add_missing_spikes_df_columns(spikes_df, session.neurons) # add the missing columns to the dataframe
                session.flattened_spiketrains.filename = session.filePrefix.with_suffix(active_file_suffix) # '.flattened.spikes.npy'
                print('\t Saving computed flattened spiketrains results to {}...'.format(session.flattened_spiketrains.filename), end='')
                session.flattened_spiketrains.save()
                print('\t done.\n')

            session = cls._try_load_clusterless_spike_events_file(session)


        def _perform_postload_comps(session):
            # Common Extended properties:
            session = cls._default_extended_postload(session.filePrefix, session)
        

        if not enable_continue_on_required_path_failure:
            ## perform all three without handling failures:
            _perform_position_comps(session)
            _perform_spike_comps(session)
            _perform_postload_comps(session)
        else:
            try:
                _perform_position_comps(session)
            except Exception as e:
                print(f'_perform_position_comps failed with err: {e} but enable_continue_on_required_path_failure == True so continuing...')

            try:
                _perform_spike_comps(session)
            except Exception as e:
                print(f'_perform_spike_comps failed with err: {e} but enable_continue_on_required_path_failure == True so continuing...')

            try:
                _perform_postload_comps(session)
            except Exception as e:
                print(f'_perform_postload_comps failed with err: {e} but enable_continue_on_required_path_failure == True so continuing...')


        return session, loaded_file_record_list
    
    

    # @function_attributes(short_name=None, tags=['laps', 'shapely', 'segmentation', 'trajectories', 'position', 'Day4OpenField'], input_requires=[], output_provides=[], uses=['shapely'], used_by=[], creation_date='2026-02-20 06:56', related_items=[])
    @classmethod
    def _perform_build_Bapun_Day4OpenField_laps_from_reward_zones(cls, pos: Position, bapun_Day4OpenField_reward_zones: Dict=None):
        """ builds correct laps (transitions between the two reward zones on the open field maze for the 'roam' experiment

        Usage:

            curr_session = curr_active_pipeline.filtered_sessions['roam']
            pos: Position = curr_session.position
            laps_obj, pos = build_Bapun_Day4OpenField_laps_from_reward_zones(pos=pos)
            ## Update the current session
            curr_session.position = pos
            curr_session.laps = laps_obj
            ## get the output dataframe:
            pos_df: pd.DataFrame = pos.to_dataframe()
            pos_df

        """
        from shapely import box
        from shapely.geometry import LineString, Point

        ## Define the two reward zones
        zone1 = bapun_Day4OpenField_reward_zones.get('zone1', box(-np.inf, 0.0, -60.0, 40.0)) # box(minx, miny, maxx, maxy, ccw=True)
        zone2 = bapun_Day4OpenField_reward_zones.get('zone2', box(80.0, 0.0, np.inf, 40.0)) # box(minx, miny, maxx, maxy, ccw=True)

        pos_df: pd.DataFrame = pos.to_dataframe()

        points = pos_df.apply(lambda row: Point(row['x'], row['y']), axis=1)
        pos_df['zone_id'] = -1 ## initialize column
        is_zone1 = [p.within(zone1) for p in points]
        is_zone2 = [p.within(zone2) for p in points]

        pos_df.loc[is_zone1, 'zone_id'] = 1
        pos_df.loc[is_zone2, 'zone_id'] = 2

        # changes = pos_df['zone_id'].diff()
        pos_df['zone_id_prev_next'] = list(
            zip(
                pos_df['zone_id'].shift(1),
                pos_df['zone_id'].shift(-1)
            )
        )
        # np.unique(pos_df['zone_id_prev_next']) # [(nan, -1.0), (-1.0, -1.0), (-1.0, 1.0), (-1.0, 2.0), (-1.0, nan), (-1.0, -1.0), (-1.0, 1.0), (-1.0, 2.0), (1.0, -1.0), (1.0, 1.0), (2.0, -1.0), (2.0, 2.0)]
        pos._df['zone_id'] = pos_df['zone_id']
        # pos._df['zone_id_prev_next'] = curr_position_df['zone_id_prev_next'] ## this one we don't need to add, it's just for building laps/transitions
        
        # ## Define Zone Enter/Exit times:
        # enter_zone1_times = (pos_df['zone_id_prev_next'] == (-1.0, 1.0))
        # exit_zone1_times = (pos_df['zone_id_prev_next'] == (1.0, -1.0))

        # enter_zone2_times = (pos_df['zone_id_prev_next'] == (-1.0, 2.0))
        # exit_zone2_times = (pos_df['zone_id_prev_next'] == (2.0, -1.0))

        new_lap_epochs_df = []
        last_zone1_exit = None
        last_zone2_exit = None
        last_successful_zone_id = None

        lap_dir_to_lap_dir_integer_mapping = {'L': 0.0, 'R': 1.0}
        for a_row in pos_df.itertuples():
            if a_row.zone_id_prev_next == (-1.0, 1.0):
                ## zone_1_enter
                if (last_successful_zone_id is not None) and (last_successful_zone_id != 1.0) and (last_zone2_exit is not None): ## last condition assumes only 2 zones
                    ## this ends a successful leftward lap
                    new_lap_epochs_df.append({'lap_dir': lap_dir_to_lap_dir_integer_mapping['L'], 'start': last_zone2_exit, 'stop': a_row.t})
                last_successful_zone_id = 1.0
            elif a_row.zone_id_prev_next == (1.0, -1.0):
                ## zone_1_exit
                last_successful_zone_id = 1.0
                last_zone1_exit = a_row.t
            elif a_row.zone_id_prev_next == (-1.0, 2.0):
                ## zone_2_enter
                if (last_successful_zone_id is not None) and (last_successful_zone_id != 2.0) and (last_zone1_exit is not None): ## last condition assumes only 2 zones
                    ## this ends a successful rightward lap
                    new_lap_epochs_df.append({'lap_dir': lap_dir_to_lap_dir_integer_mapping['R'], 'start': last_zone1_exit, 'stop': a_row.t})
                last_successful_zone_id = 2.0
            elif a_row.zone_id_prev_next == (2.0, -1.0):
                ## zone_2_exit
                last_successful_zone_id = 2.0
                last_zone2_exit = a_row.t
            else:
                ## catches all self-transitions
                pass

        ## Build the dataframe:
        new_lap_epochs_df = pd.DataFrame.from_records(new_lap_epochs_df)
        new_lap_epochs_df['duration'] = new_lap_epochs_df['stop'] - new_lap_epochs_df['start']
        new_lap_epochs_df['label'] = new_lap_epochs_df.index.astype(int)
        new_lap_epochs_df['lap_id'] = new_lap_epochs_df.index.astype(int)

        if 'is_LR_dir' not in new_lap_epochs_df.columns:
            new_lap_epochs_df['is_LR_dir'] = (new_lap_epochs_df['lap_dir'] == 0)

        if 'lap' not in new_lap_epochs_df.columns:
            new_lap_epochs_df['lap'] = new_lap_epochs_df['lap_id'].astype(int)

        # new_lap_epochs_df

        new_laps_obj: Laps = Laps(new_lap_epochs_df)
        new_lap_epochs_df = new_laps_obj.to_dataframe()
        # new_laps_obj

        pos_df = pos_df.position.adding_lap_info(laps_df=new_lap_epochs_df, inplace=False)
        ## OUTPUTS: new_laps_obj, pos_df 
        ## UPDATES: pos_df -- added lap, lap_dir

        # update:
        pos._df['lap'] = pos_df['lap']
        pos._df['lap_dir'] = pos_df['lap_dir']
        
        if 'lap_dir_1D' in pos._df:
            pos._df['lap_dir_1D'] = pos_df['lap_dir']
            
        if 'lap_dir_2D' in pos._df:
            pos._df['lap_dir_2D'] = pos_df['lap_dir']
        
        return new_laps_obj, pos


    @classmethod
    def build_Bapun_OpenField_laps_from_reward_zones(cls, session, bapun_OpenField_reward_zones: Dict=None):
        """ builds correct laps (transitions between the two reward zones on the open field maze for the 'roam' experiment
        
        session = curr_active_pipeline.filtered_sessions['roam']
                
        """            
        from neuropy.core import Laps

        if bapun_OpenField_reward_zones is None:
            ## define default reward zones:
            xmin: float = -85.75619321393464
            xmax: float = 112.57838773103435
            ymin: float = -96.44772761274268
            ymax: float = 98.6220528078153
            bapun_Day4OpenField_grid_bin_bounds = box(xmin, ymin, xmax, ymax)

            bapun_OpenField_reward_zones = dict(    ## Define the two reward zones
                zone1 = box(xmin, 0.0, -60.0, 40.0),  # box(minx, miny, maxx, maxy, ccw=True)
                zone2 = box(80.0, 0.0, xmax, 40.0), # box(minx, miny, maxx, maxy, ccw=True)
            )


        pos: Position = session.position
        laps_obj, pos = cls._perform_build_Bapun_Day4OpenField_laps_from_reward_zones(pos=pos, bapun_Day4OpenField_reward_zones=bapun_OpenField_reward_zones)
        ## Update the current session
        session.position = pos
        session.laps = laps_obj
        ## get the output dataframe:
        pos_df: pd.DataFrame = pos.to_dataframe()
        pos_df

        return session
