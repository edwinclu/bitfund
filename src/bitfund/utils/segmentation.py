"""
Algorithm Description: Dynamic Programming with Penalty
The problem can be framed as finding an optimal partitioning of the time series into segments such that the sum
of within-segment variations is minimized, subject to the continuity constraint.
"""

# Dynamic programming.
"""
1. Define "Variation within a Cluster/Segment":
Constant Model: If you assume the price within a segment is relatively constant, the variation can be the sum of squared differences from the segment's mean price: 
sum_k=i^j(y_k−bary∗i,j)^2, where 
bary∗i,j is the mean of y_k for k in[i,j].
Linear Model: If you expect trends within segments, you could fit a linear regression line y=mx+b and use the sum of squared residuals: 
sum_k=i^j(y_k−(mx_k+b))^2. This is generally more robust for time series.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def calculate_segment_cost_linear(segment_data):
    """
    Calculates the cost (sum of squared residuals from a linear fit) for a given segment.
    Assuming that there is a linear trend within a segment.
    segment_data: A numpy array of shape (n_points, 2), where column 0 is x (time) and column 1 is y (price).
    """
    if len(segment_data) < 2:
        # A single point or empty segment has no variation, but can be penalized later if desired.
        # For a linear fit, we need at least 2 points.
        return 0.0

    x = segment_data[:, 0]
    y = segment_data[:, 1]

    # Handle the case of constant x values (vertical line) if it can occur
    if np.all(x == x[0]):
        # If all x values are the same, fit a constant (mean of y)
        return np.sum((y - np.mean(y)) ** 2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    predicted_y = slope * x + intercept
    cost = np.sum((y - predicted_y) ** 2)
    return cost

def calculate_segment_cost_constant(segment_data):
    """
    Calculates the cost (sum of squared differences from the mean) for a given segment.
    Assuming that there is a constant within a segment.
    Args:
        segment_data: A numpy array of shape (n_points, 2), where column 0 is x (time) and column 1 is y (price).

    Returns:
        The sum of squared differences from the mean of the y-values in the segment.
    """
    if len(segment_data) == 0:
        return 0.0

    y = segment_data[:, 1]
    mean_y = np.mean(y)
    cost = np.sum((y - mean_y)**2)
    return cost


def segment_time_series_constant_model_np(data_array, penalty_lambda):
    """
    Segments time series data points using dynamic programming with a 'Constant Model'
    for within-segment variation.

    Args:
        data_array: A numpy array of shape (N, 2), where column 0 is x (time) and column 1 is y (price).
        penalty_lambda: The penalty for each new segment created. Higher values
                        result in fewer segments.

    Returns:
        A list of numpy arrays, where each inner numpy array represents a segment of data points.
    """
    n = len(data_array)
    if n == 0:
        return []
    if n == 1:
        return [data_array]  # Return a list containing the single point as a segment

    # DP array: dp[i] stores the minimum cost to segment data_array up to index i-1
    dp = np.full(n + 1, np.inf)
    dp[0] = 0.0  # Base case: cost to segment 0 points is 0

    # Path array: path[i] stores the start index of the last segment ending at i-1
    path = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):  # i is the end index (exclusive for slicing) + 1
        for j in range(i):  # j is the start index (inclusive)
            # Segment is from index j to i-1
            current_segment_data = data_array[j:i]

            # Use the constant model cost function
            cost = calculate_segment_cost_constant(current_segment_data)

            # The total cost includes the DP value of the previous segment end,
            # the current segment's cost, and a penalty for starting a new segment.
            total_cost_candidate = dp[j] + cost + penalty_lambda

            if total_cost_candidate < dp[i]:
                dp[i] = total_cost_candidate
                path[i] = j

    # Reconstruct the segments
    segments = []
    current_end = n
    while current_end > 0:
        current_start = path[current_end]
        segment = data_array[current_start:current_end]  # Slice the original numpy array
        segments.append(segment)
        current_end = current_start

    return list(reversed(segments))


# --- Sample Usage ---
if __name__ == "__main__":
    # Generate some synthetic time series data
    np.random.seed(42)
    time = np.arange(100)
    price = np.zeros(100)

    # Create segments with different trends and levels
    price[0:20] = 5 + 0.1 * time[0:20] + np.random.normal(0, 0.5, 20)
    price[20:40] = 7 - 0.05 * (time[20:40] - 20) + np.random.normal(0, 0.5, 20)
    price[40:60] = 6 + 0.2 * (time[40:60] - 40) + np.random.normal(0, 0.5, 20)
    price[60:80] = 10 + np.random.normal(0, 0.8, 20)  # Flat segment with more noise
    price[80:100] = 8 - 0.1 * (time[80:100] - 80) + np.random.normal(0, 0.5, 20)

    # Create the data as a NumPy array directly
    data_array = np.column_stack((time, price))

    # Experiment with different penalty_lambda values
    # Higher penalty results in less segments
    penalty_lambda = 50.0  # Adjust this value to control the number of segments.

    print(f"Segmenting with Constant Model and penalty_lambda = {penalty_lambda} (NumPy array input)")
    segments = segment_time_series_constant_model_np(data_array, penalty_lambda)

    print(f"Found {len(segments)} segments.")
    for i, seg in enumerate(segments):
        # When printing, use seg[0, 0] for x_start and seg[-1, 0] for x_end
        print(f"Segment {i + 1}: from x={seg[0, 0]} to x={seg[-1, 0]} (length {len(seg)})")

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(time, price, 'o-', markersize=3, label='Original Data')

    colors = plt.cm.get_cmap('viridis', len(segments))
    for i, segment in enumerate(segments):
        seg_x = segment[:, 0]
        seg_y = segment[:, 1]
        plt.plot(seg_x, seg_y, '-', linewidth=3, color=colors(i), label=f'Segment {i + 1}')

        # Plot the mean value for the constant model
        if len(seg_y) > 0:
            mean_val = np.mean(seg_y)
            plt.plot(seg_x, [mean_val] * len(seg_x), '--', color='black', linewidth=1)

    plt.title(f'Time Series Segmentation (Constant Model, NumPy Input, Penalty: {penalty_lambda})')
    plt.xlabel('Time (x)')
    plt.ylabel('Price (y)')
    plt.legend()
    plt.grid(True)
    plt.show()

