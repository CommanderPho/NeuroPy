from copy import deepcopy
import numpy as np
import pandas as pd

from neuropy.core.laps import Laps

from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs # for _build_new_lap_and_intra_lap_intervals




## Separate Runs on the Track

# Change of direction/inflection point by looking at the acceleration curve.

# velocity = np.insert(np.diff(pos), 0, 0)
# acceleration = np.insert(np.diff(velocity), 0, 0)


# Emphasize/hightlight poisition points within a specified time range


# Define Run:
	# Find all times the animal crosses the midline (the line bisecting the track through its midpoint) of the track.

# def compute_lap_estimation(pos_df):
#     # estimates the laps from the positions
# 	# pos_df at least has the columns 't', 'x'
# 	velocity = np.insert(np.diff(pos_df['x']), 0, 0)
# 	acceleration = np.insert(np.diff(velocity), 0, 0)

# def _build_laps_object(pos_t_rel_seconds, desc_crossing_beginings, desc_crossing_midpoints, desc_crossing_endings, asc_crossing_beginings, asc_crossing_midpoints, asc_crossing_endings):
#     ## Build a custom Laps dataframe from the found points:
#     ### Note that these crossing_* indicies are for the position dataframe, not the spikes_df (which is what the previous Laps object was computed from).
#         # This means we don't have 'start_spike_index' or 'end_spike_index', and we'd have to compute them if we want them.
#     custom_test_laps_df = pd.DataFrame({
#         'start_position_index': np.concatenate([desc_crossing_beginings, asc_crossing_beginings]),
#         'end_position_index': np.concatenate([desc_crossing_endings, asc_crossing_endings]),
#         'lap_dir': np.concatenate([np.zeros_like(desc_crossing_midpoints), np.ones_like(asc_crossing_midpoints)])
#     })
#     # Get start/end times from the indicies
#     custom_test_laps_df['start_t_rel_seconds'] = np.concatenate([pos_t_rel_seconds[desc_crossing_beginings], pos_t_rel_seconds[asc_crossing_beginings]])
#     custom_test_laps_df['end_t_rel_seconds'] = np.concatenate([pos_t_rel_seconds[desc_crossing_endings], pos_t_rel_seconds[asc_crossing_endings]])
#     custom_test_laps_df['start'] = custom_test_laps_df['start_t_rel_seconds']
#     custom_test_laps_df['stop'] = custom_test_laps_df['end_t_rel_seconds']
#     # Sort the laps based on the start time, reset the index, and finally assign lap_id's from the sorted laps
#     custom_test_laps_df = custom_test_laps_df.sort_values(by=['start']).reset_index(drop=True) # sorts all values in ascending order
#     custom_test_laps_df['lap_id'] = (custom_test_laps_df.index + 1) # set the lap_id column to the index starting at 1
#     return Laps(custom_test_laps_df)

# def estimate_laps(pos_df: pd.DataFrame, hardcoded_track_midpoint_x=150.0):
#     desc_crossing_beginings, desc_crossing_midpoints, desc_crossing_endings, asc_crossing_beginings, asc_crossing_midpoints, asc_crossing_endings = _perform_estimate_laps(pos_df, hardcoded_track_midpoint_x=hardcoded_track_midpoint_x)
#     custom_test_laps_obj = _build_laps_object(pos_df['t'].to_numpy(), desc_crossing_beginings, desc_crossing_midpoints, desc_crossing_endings, asc_crossing_beginings, asc_crossing_midpoints, asc_crossing_endings)



def _subfn_perform_compute_laps_spike_indicies(laps_df: pd.DataFrame, spikes_df: pd.DataFrame, time_variable_name='t_rel_seconds'):
    """ Adds the 'start_spike_index' and 'end_spike_index' columns to the laps_df
    laps_df has two columns added: 'start_spike_index' and 'end_spike_index'
    spikes_df is not modified
        
    Known Usages: Called only by `compute_laps_spike_indicies(...)`
    """
    n_laps = len(laps_df['start'])
    start_spike_index = np.zeros_like(laps_df['start'])
    end_spike_index = np.zeros_like(laps_df['start'])
    for i in np.arange(n_laps):
        included_df = spikes_df[((spikes_df[time_variable_name] >= laps_df.loc[i,'start']) & (spikes_df[time_variable_name] <= laps_df.loc[i,'stop']))]
        included_indicies = included_df.index
        if len(included_indicies) > 0:
            start_spike_index[i] = included_indicies[0]
            end_spike_index[i] = included_indicies[-1]
        else:
            # there were no spikes at all that fell within this lap. Is it a real lap?
            start_spike_index[i] = 0 # or np.nan?
            end_spike_index[i] = 0

    # Add the start and end spike indicies to the laps df:
    laps_df['start_spike_index'] = start_spike_index
    laps_df['end_spike_index'] = end_spike_index

    return laps_df


def _subfn_compute_laps_spike_indicies(laps_obj: Laps, spikes_df: pd.DataFrame, time_variable_name='t_rel_seconds'):
    """ Determine the spikes included with each computed lap 

    Called only by `estimation_session_laps(...)`    
    """
    laps_obj._data = _subfn_perform_compute_laps_spike_indicies(laps_obj._data, spikes_df, time_variable_name=time_variable_name) # adds the 'start_spike_index' and 'end_spike_index' columns to the dataframe
    laps_obj._data = Laps._update_dataframe_computed_vars(laps_obj._data) # call this to update the column types and any computed columns that depend on the added columns (such as num_spikes)
    return laps_obj

 
def _subfn_perform_estimate_lap_splits_1D(pos_df: pd.DataFrame, hardcoded_track_midpoint_x=150.0):
    """ Pho 2021-12-20 - Custom lap computation based on position/velocity thresholding to detect laps
    pos_df
    hardcoded_track_midpoint_x: Take 150.0 as the x midpoint line to be crossed for each trajectory

    Usage:
        desc_crossing_beginings, desc_crossing_midpoints, desc_crossing_endings, asc_crossing_beginings, asc_crossing_midpoints, asc_crossing_endings = estimate_laps(pos_df)

        # there are three variables for each of the two movement directions (ascending and descending)
        # for each movement direction:        
            crossing_midpoints: the indicies were the animal crosses the hardcoded_track_midpoint_x
            beginnings and endings: the nearest {preceding/following} points where the animal changes direction.

    Known Usages: Called only by `estimation_session_laps(...)`
    """
    assert set(['x','velocity_x_smooth']).issubset(pos_df.columns), 'pos_df requires the columns "x", and "velocity_x_smooth" at a minimum'
    zero_centered_x = pos_df['x'] - hardcoded_track_midpoint_x
    zero_crossings_x = np.diff(np.sign(zero_centered_x))
    # Find ascending crossings:
    asc_crossing_midpoint_idxs = np.where(zero_crossings_x > 0)[0] # (24,), corresponding to increasing positions
    # find descending crossings:
    desc_crossing_midpoint_idxs = np.where(zero_crossings_x < 0)[0] # (24,)
    print(f'desc_crossings_x: {np.shape(desc_crossing_midpoint_idxs)}, asc_crossings_x: {np.shape(asc_crossing_midpoint_idxs)}') # desc_crossings_x: (24,), asc_crossings_x: (24,)

    ## Build output arrays:
    desc_crossing_begining_idxs = np.zeros_like(desc_crossing_midpoint_idxs)
    desc_crossing_ending_idxs = np.zeros_like(desc_crossing_midpoint_idxs)

    asc_crossing_begining_idxs = np.zeros_like(asc_crossing_midpoint_idxs)
    asc_crossing_ending_idxs = np.zeros_like(asc_crossing_midpoint_idxs)

    ## Ensure that there are the same number of desc/asc crossings (meaning full laps). Drop the last one of the set that has the extra if they aren't equal.
    if len(desc_crossing_midpoint_idxs) > len(asc_crossing_midpoint_idxs):
        print(f'WARNING: must drop last desc_crossing_midpoint.')
        assert len(desc_crossing_midpoint_idxs) > 1
        desc_crossing_midpoint_idxs = desc_crossing_midpoint_idxs[:-1] # all but the very last which is dropped
        
    elif len(asc_crossing_midpoint_idxs) > len(desc_crossing_midpoint_idxs):
        print(f'WARNING: must drop last asc_crossing_midpoints.')
        assert len(asc_crossing_midpoint_idxs) > 1
        asc_crossing_midpoint_idxs = asc_crossing_midpoint_idxs[:-1] # all but the very last which is dropped
        
    assert len(asc_crossing_midpoint_idxs) == len(desc_crossing_midpoint_idxs), f"desc_crossings_x: {np.shape(desc_crossing_midpoint_idxs)}, asc_crossings_x: {np.shape(asc_crossing_midpoint_idxs)}"
    desc_crossing_midpoint_idxs, asc_crossing_midpoint_idxs

    # testing-only, work on a single crossing:

    # If it's a descending crossing, we'll look for the times where the velocity becomes positive (meaning the animal is moving in the ascending direction again)
    for a_desc_crossing_i in np.arange(len(desc_crossing_midpoint_idxs)):
        a_desc_crossing = desc_crossing_midpoint_idxs[a_desc_crossing_i]
        # print(f'a_desc_crossing: {a_desc_crossing}')
        
        curr_remainder_pos_df = pos_df.loc[a_desc_crossing:, :]
        curr_next_transition_points = curr_remainder_pos_df[curr_remainder_pos_df['velocity_x_smooth'] > 0.0].index # the first increasing
        curr_next_transition_point = curr_next_transition_points[0] # desc endings
        desc_crossing_ending_idxs[a_desc_crossing_i] = curr_next_transition_point

        # Preceeding points:
        curr_preceeding_pos_df = pos_df.loc[0:a_desc_crossing, :]
        curr_prev_transition_points = curr_preceeding_pos_df[curr_preceeding_pos_df['velocity_x_smooth'] > 0.0].index # the last increasing # TODO: this is not quite right.
        curr_prev_transition_point = curr_prev_transition_points[-1] # Get last (nearest to curr_preceeding_pos_df's end) point. desc beginings
        desc_crossing_begining_idxs[a_desc_crossing_i] = curr_prev_transition_point

    # If it's an ascending crossing, we'll look for the nearest times where the velocity is positive (meaning the animal is moving in the descending direction):
    for a_asc_crossing_i in np.arange(len(asc_crossing_midpoint_idxs)):
        an_asc_crossing = asc_crossing_midpoint_idxs[a_asc_crossing_i]
        # print(f'a_desc_crossing: {a_desc_crossing}')
        curr_remainder_pos_df = pos_df.loc[an_asc_crossing:, :] # get from the an_asc_crossing up to the end of the pos_df

        # find the first index where the velocity is decreasing:
        curr_next_transition_points = curr_remainder_pos_df[curr_remainder_pos_df['velocity_x_smooth'] < 0.0].index # the first decreasing
        curr_next_transition_point = curr_next_transition_points[0] # asc endings
        asc_crossing_ending_idxs[a_asc_crossing_i] = curr_next_transition_point
        
        # Preceeding points:
        curr_preceeding_pos_df = pos_df.loc[0:an_asc_crossing, :] # get all pos_df prior to an_asc_crossing
        curr_prev_transition_points = curr_preceeding_pos_df[curr_preceeding_pos_df['velocity_x_smooth'] < 0.0].index #
        curr_prev_transition_point = curr_prev_transition_points[-1] # Get last (nearest to curr_preceeding_pos_df's end) point. desc beginings
        asc_crossing_begining_idxs[a_asc_crossing_i] = curr_prev_transition_point

    # return desc_crossing_beginings, desc_crossing_midpoints, desc_crossing_endings, asc_crossing_beginings, asc_crossing_midpoints, asc_crossing_endings
    return (desc_crossing_begining_idxs, desc_crossing_midpoint_idxs, desc_crossing_ending_idxs), (asc_crossing_begining_idxs, asc_crossing_midpoint_idxs, asc_crossing_ending_idxs)




def estimation_session_laps(sess, N=20, should_backup_extant_laps_obj=False, should_plot_laps_2d=False, time_variable_name='t_rel_seconds'):
    """ 2021-12-21 - Pho's lap estimation from the position data (only)
    Replaces the sess.laps which is computed or loaded from the spikesII.mat spikes data (which isn't very good)

    CAVIAT: Only works for the 1D track (not the 2D track)
    USES: Used in KDibaOldDataSessionFormat as a post-processing step to replace the laps computed from the spikesII.mat data

    """
    if should_plot_laps_2d:
        from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import plot_laps_2d

    # backup the extant laps object to prepare for the new one:
    if should_backup_extant_laps_obj:
        sess.old_laps_obj = deepcopy(sess.laps)
        
    if should_plot_laps_2d:
        # plot originals:
        fig, out_axes_list = plot_laps_2d(sess, legacy_plotting_mode=True)
        out_axes_list[0].set_title('Old SpikeII computed Laps')
    
    position_obj = sess.position
    # position_obj.dt
    position_obj.compute_higher_order_derivatives()
    pos_df = position_obj.compute_smoothed_position_info(N=N) ## Smooth the velocity curve to apply meaningful logic to it
    pos_df = position_obj.to_dataframe()
    # custom_test_laps = deepcopy(sess.laps)
    spikes_df = deepcopy(sess.spikes_df)

    lap_change_indicies = _subfn_perform_estimate_lap_splits_1D(pos_df)
    (desc_crossing_begining_idxs, desc_crossing_midpoint_idxs, desc_crossing_ending_idxs), (asc_crossing_begining_idxs, asc_crossing_midpoint_idxs, asc_crossing_ending_idxs) = lap_change_indicies

    ## Get the timestamps corresponding to the indicies:
    # pos_times = pos_df['t'].to_numpy()
    # desc_crossing_beginings, desc_crossing_midpoints, desc_crossing_endings, asc_crossing_beginings, asc_crossing_midpoints, asc_crossing_endings = [pos_times[idxs] for idxs in (desc_crossing_begining_idxs, desc_crossing_midpoint_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_midpoint_idxs, asc_crossing_ending_idxs)]

    custom_test_laps_obj = Laps.from_estimated_laps(pos_df['t'].to_numpy(), desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs)
    ## Determine the spikes included with each computed lap:
    custom_test_laps_obj = _subfn_compute_laps_spike_indicies(custom_test_laps_obj, spikes_df, time_variable_name=time_variable_name)
    sess.laps = deepcopy(custom_test_laps_obj) # replace the laps obj

    if should_plot_laps_2d:
        # plot computed:
        fig, out_axes_list = plot_laps_2d(sess, legacy_plotting_mode=False)
        out_axes_list[0].set_title('New Pho Position Thresholding Estimated Laps')

    return sess






# Load from the 'traj' variable of an exported SpikeII.mat file:


## Direction-dependent tuning curves (find the direction of the animal at the time of each spike, bin them into 8 radial directions, and show the curves separately.


# I wonder if it follows a predictable cycle.

# def get_lap_position(curr_lap_id):
#     curr_position_df = sess.position.to_dataframe()
#     curr_lap_t_start, curr_lap_t_stop = get_lap_times(curr_lap_id)
#     print('lap[{}]: ({}, {}): '.format(curr_lap_id, curr_lap_t_start, curr_lap_t_stop))

#     curr_lap_position_df_is_included = curr_position_df['t'].between(curr_lap_t_start, curr_lap_t_stop, inclusive=True) # returns a boolean array indicating inclusion in teh current lap
#     curr_lap_position_df = curr_position_df[curr_lap_position_df_is_included] 
#     # curr_position_df.query('-0.5 <= t < 0.5')
#     curr_lap_position_traces = curr_lap_position_df[['x','y']].to_numpy().T
#     print('\t {} positions.'.format(np.shape(curr_lap_position_traces)))
#     # print('\t {} spikes.'.format(curr_lap_num_spikes))
#     return curr_lap_position_traces



# # Main Track Barrier Parts:
# [63.5, 138.6] # bottom-left edge of the left-most track/platform barrier
# [63.5, 144.2] # top-left edge of the left-most track/platform barrier
# [223.9, 137.4] # bottom-right edge of the right-most track/platform barrier
# [223.9, 150.0] # top-right edge of the right-most track/platform barrier


## Laps:


# ==================================================================================================================== #
# 2022-12-13 _build_new_lap_and_intra_lap_intervals                                                                    #
# ==================================================================================================================== #

## Newest way of dropping bad laps:
from neuropy.utils.mixins.indexing_helpers import interleave_elements # for _build_new_lap_and_intra_lap_intervals

def _build_new_lap_and_intra_lap_intervals(sess):
    """ 
    
    Factored out of Notebook on 2022-12-13
    
    Usage:
        from neuropy.analyses.laps import _build_new_lap_and_intra_lap_intervals
        sess = curr_active_pipeline.sess
        sess, combined_records_list = _build_new_lap_and_intra_lap_intervals(sess)

    """
    ## Backup original laps object if it hasn't already been done:
    if not hasattr(sess, 'old_laps_obj'):
        print(f'backing up laps object to sess.old_laps_obj.')
        sess.old_laps_obj = deepcopy(sess.laps)

    ## Get only the non-overlapping laps (dropping the overlapping ones) and replace the old laps object in sess.laps with this new good one:
    is_non_overlapping_lap = get_non_overlapping_epochs(sess.laps.to_dataframe()[['start','stop']].to_numpy())
    only_good_laps_df = sess.laps.to_dataframe()[is_non_overlapping_lap]
    sess.laps = Laps(only_good_laps_df) # replace the laps object with the filtered one
    ## Extract the epochs from the new good (non-overlapping) laps:
    lap_specific_epochs = sess.laps.as_epoch_obj()
    any_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(len(sess.laps.lap_id))])
    even_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(0, len(sess.laps.lap_id), 2)])
    odd_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(1, len(sess.laps.lap_id), 2)])

    only_good_laps_df['epoch_type'] = 'lap'
    laps_records_list = list(only_good_laps_df[['epoch_type','start','stop','lap_id']].itertuples(index=False, name='lap')) # len(laps_records_list) # 74

    ## Build the intra-lap periods and make a dataframe for them (`intra_lap_df`)
    starts = [sess.paradigm[0][0,0]]
    stops = []
    preceeding_lap_ids = [-1]
    for i, row in only_good_laps_df.iterrows():
        # print(f'row: {row}')
        stops.append(row.start)
        starts.append(row.stop)
        preceeding_lap_ids.append(row.lap_id)
    stops.append(sess.paradigm[1][0,1])
    intra_lap_df = pd.DataFrame(dict(start=starts, stop=stops, preceding_lap_id=preceeding_lap_ids))
    intra_lap_df['intra_lap_interval_id'] = intra_lap_df.index.astype(int)
    intra_lap_df['epoch_type'] = 'intra'

    intra_lap_interval_records = list(intra_lap_df[['epoch_type','start','stop','intra_lap_interval_id']].itertuples(index=False, name='intera_lap')) # len(intra_lap_interval_records) # 75
    # intra_lap_interval_records

    ## interleave the two lists to get alternting tuples: [(inter-lap), (lap 1), (inter-lap), (lap 2), ...]
    combined_records_list = interleave_elements(intra_lap_interval_records[:-1], laps_records_list).tolist()
    combined_records_list.append(list(intra_lap_interval_records[-1])) # add the last element which was omitted so they would be the same length.
    # combined_records_list: [['intra', '0.0', '8.806375011103228', '0'],
    #  ['lap', '8.806375011103228', '16.146792372805066', '1'],
    #  ['intra', '16.146792372805066', '33.77391934779007', '1'],
    #  ['lap', '33.77391934779007', '39.23866662976798', '2'],
    #  ['intra', '39.23866662976798', '136.07198921125382', '2'],
    #  ['lap', '136.07198921125382', '143.75791423593182', '3'],
    #  ...
    #  ['intra', '1975.4521520885173', '1978.5230138762854', '73'],
    #  ['lap', '1978.5230138762854', '1988.0340438865824', '74'],
    #  ['intra', 1988.0340438865824, 2093.8978568242164, 74]]
    # combined_records_list
    
    ## Can build a pd.DataFrame version:
#     combined_df = pd.DataFrame.from_records(combined_records_list, columns=['epoch_type','start','stop','interval_type_id'], coerce_float=True)
#     combined_df['label'] = combined_df.index.astype("str") # add the required 'label' column so it can be convereted into an Epoch object
#     combined_df[['start','stop']] = combined_df[['start','stop']].astype('float')
#     combined_df[['interval_type_id']] = combined_df[['interval_type_id']].astype('int')
#     combined_df

    ## Can build an Epoch object version:
#     combined_epoch_obj = Epoch(epochs=combined_df)
#     combined_epoch_obj
    return sess, combined_records_list


