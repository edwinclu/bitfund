"""
A running version of dynamic segmentation algorithm, where initially you have a batch, and when
new data points arrive, you'll re-run segmentation.
Constraints:
    the existing segmentation should not change?
    the mean of last segmentation should not change?

Note: This is a stable version of segmentation_running.
When a new point arrives, we make sure the start point is not changed.
However, the running part is purely based on mono_n and drastic_factor.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # For linear model if we switch back


class ConsistentRunningSegmenter:
    def __init__(self, initial_data_array, penalty_lambda, condition_penalty=0.0, mono_n=10, drastic_factor=1.3):
        """
        Initializes the segmenter with an initial batch of data.

        Args:
            initial_data_array (np.array): Initial batch of data points (N, 2).
            penalty_lambda (float): The penalty for each new segment created.
            condition_penalty(float): Give at least this penalty if there are more than mono_n points on one side
            mono_n: the threshold to amplify the penalty, when more than mono_n consecutive points are on one side
            drastic_factor: if there is a drastic change of more than this, double the penalty.
        """
        if len(initial_data_array) == 0:
            raise ValueError("Initial data cannot be empty.")

        self.data = initial_data_array.copy()
        self.penalty_lambda = penalty_lambda
        self.condition_penalty = condition_penalty
        self.mono_n = mono_n
        self.drastic_factor = drastic_factor

        self.n = len(self.data)
        self.dp = np.full(self.n + 1, np.inf)
        self.path = np.zeros(self.n + 1, dtype=int)

        self.dp[0] = 0.0

        # Run initial batch segmentation
        self._run_dp_for_range(1, self.n + 1)
        print(f"Initial batch segmentation completed with {len(self.get_segments())} segments.")

    def _calculate_segment_cost_constant(self, segment_data_array):
        """
        Calculates the cost (sum of squared differences from the mean) for a given segment.
        Assumes segment_data_array is a numpy array of shape (n_points, 2).
        """
        if len(segment_data_array) == 0:
            return 0.0
        y = segment_data_array[:, 1]
        mean_y = np.mean(y)
        cost = np.sum((y - mean_y) ** 2)
        return cost

    def _calculate_segment_cost_running(self, segment_data_array):
        """
        Calculates the cost (sum of squared differences from the mean) for a given segment.
        Assumes segment_data_array is a numpy array of shape (n_points, 2).
        Bump the cost to multiples or condition_penalty if
            mono_n consecutive points > (<) mean,
            or any point changes drastically by a factor of x
        Returns (x,y)
            x is the cost of adding to the same segment.
            y is the suggested length of the new segment if to break.
        """

        if len(segment_data_array) >= self.mono_n + 1:
            y = segment_data_array[:, 1]
            n = len(segment_data_array)
            prior_mean = np.mean(y[0:-1])
            prior_mono_n_mean = np.mean(y[0: n - self.mono_n])
            # check for drastic change.
            if y[-1] > prior_mean * self.drastic_factor or y[-1] * self.drastic_factor < prior_mean:
                # if there is a drastic change, double the cost
                # last 1 point is the outlier
                return 2 * max(self._calculate_segment_cost_constant(segment_data_array[0:-1]), self.condition_penalty), 1
            greater_than = sum(1 for e in y[n - self.mono_n:n] if e > prior_mono_n_mean)
            less_than = sum(1 for e in y[n - self.mono_n:n] if e < prior_mono_n_mean)
            if greater_than == self.mono_n or less_than == self.mono_n:
                # if there are n consecutive mono side points, multiple the cost
                # last mono_n points are the outliers
                return self.mono_n * max(self._calculate_segment_cost_constant(segment_data_array[0:-1]),
                                    self.condition_penalty), self.mono_n
        return self._calculate_segment_cost_constant(segment_data_array), 0

    def _run_dp_for_range(self, start_idx_dp, end_idx_dp):
        """Helper to run DP for a specified range of indices."""
        # start_idx_dp and end_idx_dp are for the dp array (i.e., exclusive end for data points)
        for i in range(start_idx_dp, end_idx_dp):
            for j in range(i):  # j is the start index of the current segment (0-based for data)
                current_segment_data = self.data[j:i]
                cost = self._calculate_segment_cost_constant(current_segment_data)
                total_cost_candidate = self.dp[j] + cost + self.penalty_lambda

                if total_cost_candidate < self.dp[i]:
                    self.dp[i] = total_cost_candidate
                    self.path[i] = j

    def print_segments(self):
        current_end = self.n  # Start from the end of the full data array
        segments = []
        while current_end > 0:
            current_start = self.path[current_end]
            segments.append((current_start, current_end))
            current_end = current_start
        segments = reversed(segments)
        for i, j in segments:
            print(f"[{i},{j})", end=" ")
        print()


    def process_new_point(self, new_point):
        #print("Segments before processing new point: ", end="")
        #self.print_segments()
        """
        Processes a new data point arriving in the stream, extending the segmentation.

        Args:
            new_point (np.array): A single data point [x, y].
        """
        # Append new point to the data array
        self.data = np.vstack((self.data, new_point))

        # Update n (total number of points)
        self.n = len(self.data)

        # Extend dp and path arrays to accommodate the new point
        # np.pad adds 0s, we will overwrite the last one with inf as default
        self.dp = np.pad(self.dp, (0, 1), 'constant', constant_values=np.inf)
        self.path = np.pad(self.path, (0, 1), 'constant', constant_values=0)

        # check cost of the last segment, as we don't want to change other segments on the fly
        old_start_idx = self.path[-2]
        current_segment_data = self.data[old_start_idx:]
        constant_cost = self._calculate_segment_cost_constant(current_segment_data)
        running_cost, outliers = self._calculate_segment_cost_running(current_segment_data)

        if constant_cost < running_cost and outliers > 0:
            print(f"Find {outliers} outliers")
            # put outliers into a new segment
            new_segment_start_idx = self.n-outliers
            for i in range(new_segment_start_idx, self.n):
                new_segment_data = self.data[new_segment_start_idx:i]
                new_segment_cost = self._calculate_segment_cost_constant(new_segment_data)
                # all points after this segment points to the start point
                self.path[i+1] = new_segment_start_idx
                self.dp[i+1] = new_segment_cost
        else:
            # put into the same segment
            self.dp[-1] = constant_cost
            self.path[-1] = self.path[-2]

        #print(f"Processed point at x={new_point[0]}. Current segments: {len(self.get_segments())}")
        #print("Segments after processing new point: ", end="")
        #self.print_segments()

    def get_segments(self):
        """
        Reconstructs and returns the current set of segments.
        """
        segments = []
        current_end = self.n  # Start from the end of the full data array
        while current_end > 0:
            current_start = self.path[current_end]
            segment = self.data[current_start:current_end]
            segments.append(segment)
            current_end = current_start
        return list(reversed(segments))


# --- Sample Usage ---
if __name__ == "__main__":
    np.random.seed(42)
    time = np.arange(150)
    price = np.zeros(150)

    # Create segments with different trends and levels
    price[0:30] = 5 + 0.1 * time[0:30] + np.random.normal(0, 0.5, 30)
    price[30:60] = 7 - 0.05 * (time[30:60] - 30) + np.random.normal(0, 0.5, 30)
    price[60:90] = 6 + 0.2 * (time[60:90] - 60) + np.random.normal(0, 0.5, 30)
    price[90:120] = 10 + np.random.normal(0, 0.8, 30)  # Flat segment with more noise
    price[120:150] = 8 - 0.1 * (time[120:150] - 120) + np.random.normal(0, 0.5, 30)

    full_data_array = np.column_stack((time, price))

    # --- Consistent Running Segmentation ---
    # Initial batch size
    initial_batch_size = 20
    initial_data = full_data_array[:initial_batch_size]

    # Choose a penalty lambda. (Adjust based on 'constant' or 'linear' cost function)
    penalty_value = 150.0  # Example for constant model, might need tuning
    # penalty_value = 100.0 # Example for linear model, might need tuning

    # Initialize the segmenter with the initial batch
    segmenter = ConsistentRunningSegmenter(initial_data_array=initial_data,
                                           penalty_lambda=penalty_value,
                                           condition_penalty=2*penalty_value)  # or 'linear'

    # Store segments at different time steps to visualize consistency
    snapshots = {}
    snapshots[initial_batch_size - 1] = segmenter.get_segments()

    # Simulate streaming by processing points one by one
    for i in range(initial_batch_size, len(full_data_array)):
        new_point = full_data_array[i:i + 1]  # Get as a (1,2) array for vstack
        segmenter.process_new_point(new_point)
        # Take snapshots at specific points in time to observe changes
        if i % 20 == 0 or i == len(full_data_array) - 1:  # Snapshot every 20 points or at the end
            snapshots[i] = segmenter.get_segments()

    # --- Plotting the snapshots ---
    fig, axes = plt.subplots(len(snapshots), 1, figsize=(12, 5 * len(snapshots)), sharex=True, sharey=True)
    if len(snapshots) == 1:  # Handle case of only one subplot
        axes = [axes]

    snapshot_keys = sorted(snapshots.keys())

    for idx, t_idx in enumerate(snapshot_keys):
        current_segments = snapshots[t_idx]
        current_data_up_to_t = full_data_array[:t_idx + 1]

        ax = axes[idx]
        ax.plot(current_data_up_to_t[:, 0], current_data_up_to_t[:, 1], 'o-', markersize=3, label='Data Up to t')

        colors = plt.cm.get_cmap('viridis', len(current_segments))
        for i, segment in enumerate(current_segments):
            seg_x = segment[:, 0]
            seg_y = segment[:, 1]
            ax.plot(seg_x, seg_y, '-', linewidth=3, color=colors(i), label=f'Segment {i + 1}')

            # Plot model line (mean for constant, regression for linear)
            if len(seg_y) > 0:
                model_val = np.mean(seg_y)
                ax.plot(seg_x, [model_val] * len(seg_x), '--', color='black', linewidth=1)

        ax.set_title(f'Segmentation at Time t={t_idx} (Total points: {len(current_data_up_to_t)})')
        ax.grid(True)
        # ax.legend(loc='upper left', bbox_to_anchor=(1,1)) # If you want individual legends

    plt.xlabel('Time (x)')
    plt.ylabel('Price (y)')
    plt.tight_layout()
    plt.show()

    # You can inspect specific segments to confirm consistency
    # For example, compare segments[0] from t=99 and t=119
    # They should be identical up to the last segment of the earlier snapshot.