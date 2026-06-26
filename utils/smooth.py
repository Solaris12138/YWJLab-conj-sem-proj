import numpy as np

from scipy.ndimage import gaussian_filter1d


def mean_smooth(data, sfreq, window_lenth):
    """
    Smooth with sliding windows (mean filter)
    
    Parameters
    ----------
    data : 2D array, shape (n_signals, n_times)
        The data to be smoothed.
    sfreq : int or float
        Sampling rate of signals.
    window_lenth : int
        The length of sliding windows.
    
    Returns
    -------
    smoothed_data : 2D array shape (n_signals, n_times)
    
    """
    
    window_samples = int(sfreq * window_lenth / 1000)
    window = np.ones(window_samples) / window_samples
    
    smoothed_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        smoothed_data[i, :] = np.convolve(data[i, :], window, mode="same")
    
    return smoothed_data


def gaussian_smooth(data, sigma):
    """
    Smooth with sliding windows (gaussian filter)
    
    Parameters
    ----------
    data : 2D array, shape (n_signals, n_times)
        The data to be smoothed.
    sigma : scalar
        Standard deviation for Gaussian kernel
    
    Returns
    -------
    smoothed_data : 2D array shape (n_signals, n_times)
    
    """
    
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=-1)
    
    return smoothed_data