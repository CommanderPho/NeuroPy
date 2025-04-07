# ==================================================================================================================== #
# @ 2025-01-01 - Better Aggregation of Probabilities across bins                                                       #
# ==================================================================================================================== #
from neuropy.utils.indexing_helpers import PandasHelpers
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import numpy as np
import pandas as pd

# @metadata_attributes(short_name=None, tags=['aggregation', 'integration', 'confidence'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 00:00', related_items=[])
class TimeBinAggregation:
    """ Methods of aggregating over many time bins

	from neuropy.analyses.time_bin_aggregation import TimeBinAggregation, ParticleFilter

    from neuropy.analyses.time_bin_aggregation import TimeBinAggregation, ParticleFilter

    ## INPUTS: a_decoded_time_bin_marginal_posterior_df

    n_rolling_avg_window_tbins: int = 3
    # Create a copy to avoid modifying the original
    result_df = a_decoded_time_bin_marginal_posterior_df.copy()
    epoch_partitioned_dfs_dict = a_decoded_time_bin_marginal_posterior_df.pho.partition_df_dict(partitionColumn='parent_epoch_label')

    # Process each partition
    for k, df in epoch_partitioned_dfs_dict.items():
        rolling_avg = TimeBinAggregation.ToPerEpoch.peak_rolling_avg(df=df, column='P_Short', window=n_rolling_avg_window_tbins)    
        # Calculate the mean of P_Short for this group
        mean_p_short = TimeBinAggregation.ToPerEpoch.mean(df=df, column='P_Short')

        # Get indices from this partition
        indices = df.index
        # Assign the result to the corresponding rows in the result dataframe
        result_df.loc[indices, 'rolling_avg_P_Short'] = rolling_avg
        result_df.loc[indices, 'mean_P_Short'] = mean_p_short  # Same mean value for all rows in group
        
        # result_df.loc[indices

    ## OUTPUTS: result_df

    a_decoded_time_bin_marginal_posterior_df = deepcopy(result_df)
    a_decoded_time_bin_marginal_posterior_df

    # Then keep only the first entry for each 'parent_epoch_label'
    a_decoded_per_epoch_marginals_df = a_decoded_time_bin_marginal_posterior_df.groupby('parent_epoch_label').first().reset_index()
    a_decoded_per_epoch_marginals_df

    ## OUTPUTS: a_decoded_time_bin_marginal_posterior_df, a_decoded_per_epoch_marginals_df

	History:
		pyphoplacecellanalysis.SpecificResults.PendingNotebookCode -> neuropy.analyses.time_bin_aggregation

    """
    
    # ==================================================================================================================================================================================================================================================================================== #
    # ToCoarserTimeBin: Aggregation across adjacent time bins, resulting in a coarser graining than the existing time bins                                                                                                                                                                 #
    # ==================================================================================================================================================================================================================================================================================== #
    class ToCoarserTimeBin:

        @classmethod
        def rolling_avg(cls, df: pd.DataFrame, column: str='P_Short', window: int=3, *args, **kwargs) -> float:
            """
            Computes the streak-weighted P_Long for a given DataFrame, giving higher weight to longer sequences of adjacent bins.

            Args:
                df (pd.DataFrame): Input DataFrame containing the P_Long column.
                column (str): The column name for P_Long.
                threshold (float): Minimum probability to consider a bin as part of a streak. Default is 0.5.

            Returns:
                float: The streak-weighted P_Long for the DataFrame.
            """
            return df[column].rolling(window, *args, **kwargs).max()

    # END ToCoarserTimeBin _______________________________________________________________________________________________ #


    # ==================================================================================================================== #
    # ToPerEpoch: Aggregation across entire epoch, resulting in a single result summarizing each epoch 
    # ==================================================================================================================== #
    class ToPerEpoch:
        """ returns a single result summarizing each epoch """

        # @function_attributes(short_name=None, tags=['NDArray', 'streak', 'sequence', 'probability', 'likelihood'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 14:00', related_items=[])
        @classmethod
        def compute_epoch_p_long_with_streaks(cls, p_long: List[float], min_probability_threshold: float = 0.5) -> float:
            """ Prior to `compute_streak_weighted_p_long` version, which operates on dataframes


            Computes the overall P_Long for an epoch, giving higher weight to uninterrupted streaks.

            Args:
                p_long (List[float]): Probabilities of being in the "Long" state for each time bin.
                threshold (float): Minimum probability to consider a bin as part of a streak. Default is 0.5.

            Returns:
                float: The overall P_Long for the epoch.

            Usage:
                # Example usage
                p_long = a_lap_df[a_var_name].to_numpy() # [0.8, 0.7, 0.2, 0.9, 0.95, 0.3, 0.85, 0.87, 0.86]
                min_probability_threshold = 0.5  # Minimum probability to count as "Long"
                overall_p_long = compute_epoch_p_long_with_streaks(p_long, min_probability_threshold)
                print(f"Overall P_Long with streak weighting: {overall_p_long}")

            """
            streaks = []
            current_streak = []

            # Identify streaks
            for i, prob in enumerate(p_long):
                if prob >= min_probability_threshold:
                    current_streak.append(i)
                else:
                    if current_streak:
                        streaks.append(current_streak)
                        current_streak = []
            if current_streak:  # Add the last streak if it ends at the last bin
                streaks.append(current_streak)

            # Assign weights based on streak length (linearly)
            weights = [0] * len(p_long)
            for streak in streaks:
                streak_length = len(streak)
                for idx in streak:
                    weights[idx] = streak_length

            # Compute weighted average
            weighted_sum = np.sum(w * p for w, p in zip(weights, p_long))
            total_weight = np.sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else 0.0


        @classmethod
        def mean(cls, df: pd.DataFrame, column: str='P_Short', *args, **kwargs) -> float:
            """
            Computes the streak-weighted P_Long for a given DataFrame, giving higher weight to longer sequences of adjacent bins.

            Args:
                df (pd.DataFrame): Input DataFrame containing the P_Long column.
                column (str): The column name for P_Long.
                threshold (float): Minimum probability to consider a bin as part of a streak. Default is 0.5.

            Returns:
                float: The streak-weighted P_Long for the DataFrame.
            """
            return df[column].mean(*args, skipna=True, **kwargs)


        # @function_attributes(short_name=None, tags=['streak', 'sequence', 'dataframe', 'probability', 'likelihood'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 14:11', related_items=[])
        @classmethod
        def compute_streak_weighted_p_long(cls, df: pd.DataFrame, column: str, threshold: float = 0.5) -> float:
            """
            Computes the streak-weighted P_Long for a given DataFrame, giving higher weight to longer sequences of adjacent bins.

            Args:
                df (pd.DataFrame): Input DataFrame containing the P_Long column.
                column (str): The column name for P_Long.
                threshold (float): Minimum probability to consider a bin as part of a streak. Default is 0.5.

            Returns:
                float: The streak-weighted P_Long for the DataFrame.
            """
            p_long = df[column].values
            streaks = []
            current_streak = []

            # Identify streaks
            for i, prob in enumerate(p_long):
                if prob >= threshold:
                    current_streak.append(i)
                else:
                    if current_streak:
                        streaks.append(current_streak)
                        current_streak = []
            if current_streak:  # Add the last streak if it ends at the last bin
                streaks.append(current_streak)

            # Assign weights based on streak length
            weights = [0] * len(p_long)
            for streak in streaks:
                streak_length = len(streak)
                for idx in streak:
                    weights[idx] = streak_length

            # Compute weighted average
            weighted_sum = sum(w * p for w, p in zip(weights, p_long))
            total_weight = sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else 0.0


        @classmethod
        def peak_rolling_avg(cls, df: pd.DataFrame, column: str='P_Short', window: int=3, *args, min_periods:int=0, center:bool=True, **kwargs) -> float:
            """
            Computes the streak-weighted P_Long for a given DataFrame, giving higher weight to longer sequences of adjacent bins.

            Args:
                df (pd.DataFrame): Input DataFrame containing the P_Long column.
                column (str): The column name for P_Long.
                threshold (float): Minimum probability to consider a bin as part of a streak. Default is 0.5.


            KWARGS:
                https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html


                min_periods (int), default (min_periods will default to the size of the window):
                    Minimum number of observations in window required to have a value; otherwise, result is np.nan.

                center (bool), default False:
                    If False, set the window labels as the right edge of the window index.
                    If True, set the window labels as the center of the window index.

                closed (str), default 'right'
                    If 'right', the first point in the window is excluded from calculations.
                    If 'left', the last point in the window is excluded from calculations.
                    If 'both', the no points in the window are excluded from calculations.
                    If 'neither', the first and last points in the window are excluded from calculations.


            Returns:
                float: The streak-weighted P_Long for the DataFrame.
            """
            from neuropy.utils.indexing_helpers import PandasHelpers

            # Define a function to get the most extreme value in each window                    
            def _subfn_most_extreme(x):
                """Return the value with the largest absolute magnitude, preserving its sign."""
                # Handle empty or all-NaN series
                if (x.empty or x.isna().all()):
                    return np.nan
                    # return None
                # Find index of maximum absolute value, ignoring NaNs
                idx = x.abs().idxmax(skipna=True)
                # Return the original value at that index
                return x.loc[idx]

            kwargs = dict(min_periods=min_periods, center=center) | kwargs # Minimum number of observations in window required to have a value; otherwise, result is np.nan.

            # BEGIN FUNCTION BODY ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # Apply the function to rolling windows
            rolling_extreme = PandasHelpers.remap_range(df[column], from_range=(0.0, 1.0), to_range=(-1.0, 1.0), safety_check=True).rolling(window, *args, **kwargs).apply(_subfn_most_extreme, raw=False) ## map original probability range to -1, +1 so the `most_extreme` function works correctly
            # Then get the most extreme value across all windows
            idx = rolling_extreme.abs().idxmax(skipna=True)
            if np.isnan(idx).all():
                return idx
            else:
                return PandasHelpers.remap_range(rolling_extreme.loc[idx], from_range=(-1.0, 1.0), to_range=(0.0, 1.0), safety_check=False) # map back to original probability range 
            


# @metadata_attributes(short_name=None, tags=['UNUSED', 'ChatGPT', 'aggregation'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 00:00', related_items=['TimeBinAggregation'])
class ParticleFilter:
    """ Example: Generated by ChatGPT, does something about aggregating time bins

        ## TEST

        # Example usage
        def state_transition_func(particles):
            # Example state transition function: simple linear motion.
            return particles + 1  # Each state increases by 1

        def measurement_func(state):
            # Example measurement function: identity mapping.
            return state



        num_particles = 1000
        state_dim = 1
        process_noise = 1.0
        measurement_noise = 2.0

        pf = ParticleFilter(num_particles, state_dim, process_noise, measurement_noise)

        # Simulate a series of observations
        # observations = np.array([5, 6, 7, 8, 9])
        observations = a_lap_df[a_var_name].to_numpy()

        for obs in observations:
            pf.predict(state_transition_func)
            pf.update(obs, measurement_func)
            pf.resample()

            estimated_state = pf.estimate()
            print(f"Estimated State: {estimated_state}")


    """
    def __init__(self, num_particles: int, state_dim: int, process_noise: float, measurement_noise: float):
        """
        Initialize the Particle Filter.

        :param num_particles: Number of particles to use.
        :param state_dim: Dimension of the state space.
        :param process_noise: Standard deviation of process noise.
        :param measurement_noise: Standard deviation of measurement noise.
        """
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Initialize particles randomly within the state space
        self.particles = np.random.rand(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, state_transition_func):
        """
        Predict the next state of each particle using the state transition function.

        :param state_transition_func: A function to apply the state transition.
        """
        noise = np.random.normal(0, self.process_noise, size=(self.num_particles, self.state_dim))
        self.particles = state_transition_func(self.particles) + noise

    def update(self, observation: np.ndarray, measurement_func):
        """
        Update the particle weights based on the observation.

        :param observation: The observed data.
        :param measurement_func: A function to calculate the expected observation from a particle state.
        """
        for i in range(self.num_particles):
            predicted_obs = measurement_func(self.particles[i])
            error = observation - predicted_obs
            likelihood = np.exp(-0.5 * np.sum(error**2) / self.measurement_noise**2)
            self.weights[i] = likelihood

        # Normalize weights to sum to 1
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Resample particles based on their weights.
        """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self) -> np.ndarray:
        """
        Estimate the state as the weighted mean of the particles.

        :return: The estimated state.
        """
        return np.average(self.particles, weights=self.weights, axis=0)

