from scipy.signal import savgol_filter

def apply_savitzky_golay(time_series, window_length, polyorder):
    """
    Applies a Savitzky-Golay filter to a time series.

    Parameters:
    - time_series: numpy array, the time series data.
    - window_length: int, the length of the filter window (must be an odd number).
    - polyorder: int, the order of the polynomial used to fit the samples.

    Returns:
    - smoothed_time_series: numpy array, the smoothed time series.
    """
    smoothed_time_series = savgol_filter(time_series, window_length, polyorder)
    return smoothed_time_series