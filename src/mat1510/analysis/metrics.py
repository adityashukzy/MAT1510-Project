"""
Shared analysis utilities for entropy metrics.
Extracted from notebooks to avoid duplication.
"""
import numpy as np


def compute_per_step_metrics(entropy, window=10):
    """Compute per-step metrics for entropy sequence.

    Args:
        entropy: Array of entropy values at each token position
        window: Rolling window size for volatility and slope calculations

    Returns:
        Dictionary containing:
        - cumsum: Cumulative sum of entropy
        - volatility: Rolling standard deviation (window size)
        - slope: Rolling linear regression slope (window size)
    """
    n = len(entropy)

    # Cumulative sum
    cumsum = np.cumsum(entropy)

    # Rolling volatility (std)
    volatility = np.full(n, np.nan)
    for i in range(window, n):
        volatility[i] = np.std(entropy[i-window:i])

    # Rolling slope
    slope = np.full(n, np.nan)
    for i in range(window, n):
        slope[i] = np.polyfit(range(window), entropy[i-window:i], 1)[0]

    return {
        'cumsum': cumsum,
        'volatility': volatility,
        'slope': slope
    }


def compute_global_metrics(entropy, window=10):
    """Compute global metrics for entropy sequence.

    Args:
        entropy: Array of entropy values at each token position
        window: Rolling window size for phase shift detection

    Returns:
        Dictionary containing:
        - mean: Mean entropy
        - median: Median entropy
        - std: Standard deviation of entropy
        - num_spikes: Number of entropy spikes (bidirectional jumps in top 5%)
        - phase_shifts: Number of significant changes in rolling statistics
    """
    # Basic statistics
    mean_val = np.mean(entropy)
    median_val = np.median(entropy)
    std_val = np.std(entropy)

    # Compute deltas
    deltas = np.diff(entropy)
    abs_deltas = np.abs(deltas)

    # Spikes: tokens where both incoming and outgoing deltas are in top 5%
    threshold_95 = np.percentile(abs_deltas, 95)
    num_spikes = 0
    for i in range(1, len(entropy) - 1):
        delta_prev = abs(entropy[i] - entropy[i-1])
        delta_next = abs(entropy[i+1] - entropy[i])
        if delta_prev >= threshold_95 and delta_next >= threshold_95:
            num_spikes += 1

    # Phase shifts: detect significant changes in rolling mean/std
    rolling_mean = np.full(len(entropy), np.nan)
    rolling_std = np.full(len(entropy), np.nan)
    for i in range(window, len(entropy)):
        rolling_mean[i] = np.mean(entropy[i-window:i])
        rolling_std[i] = np.std(entropy[i-window:i])

    # Detect phase shifts: large changes in rolling statistics
    mean_changes = np.abs(np.diff(rolling_mean[window:]))
    std_changes = np.abs(np.diff(rolling_std[window:]))

    mean_change_threshold = np.percentile(mean_changes[~np.isnan(mean_changes)], 90) if len(mean_changes) > 0 else 0
    std_change_threshold = np.percentile(std_changes[~np.isnan(std_changes)], 90) if len(std_changes) > 0 else 0

    phase_shifts = 0
    for i in range(len(mean_changes)):
        if (not np.isnan(mean_changes[i]) and mean_changes[i] >= mean_change_threshold) or \
           (not np.isnan(std_changes[i]) and std_changes[i] >= std_change_threshold):
            phase_shifts += 1

    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'num_spikes': num_spikes,
        'phase_shifts': phase_shifts
    }
