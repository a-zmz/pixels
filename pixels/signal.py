"""
This module provides functions that operate on signal data.
"""


import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from scipy.ndimage import gaussian_filter1d

from pixels import PixelsError


def resample(array, from_hz, to_hz, padtype=None):
    """
    Resample an array from one sampling rate to another.

    Parameters
    ----------
    array : ndarray, Series or similar
        The data to be resampled.

    from_hz : int or float
        The starting frequency of the data.

    sample_rate : int or float, optional
        The resulting sample rate.

    padtype : string, optional
        How to deal with end effects. Default: 'minimum', where the resampling filter
        pretends the ends are extended using the array's minimum.

    Returns
    -------
    np.ndarray : Array of the same width of the input array, but altered height.

    """
    from_hz = float(from_hz)
    to_hz = float(to_hz)

    if from_hz == to_hz:
        return array

    if from_hz > to_hz:
        up = 1
        factor = from_hz / to_hz
        down = factor
        while not down.is_integer():
            down += factor
            up += 1

    elif from_hz < to_hz:
        factor = to_hz / from_hz
        up = factor
        down = 1
        while not up.is_integer():
            up += factor
            down += 1

    new_data = []
    if array.ndim == 1:
        cols = 1
        array = array.reshape((-1, cols))
    else:
        cols = array.shape[1]

    # resample_poly preallocates an entire new array of float64 values, so to prevent
    # MemoryErrors we will run it with 5GB chunks
    size_bytes = array[0].dtype.itemsize * array.size
    chunks = int(np.ceil(size_bytes / 5368709120))
    chunk_size = int(np.ceil(cols / chunks))

    if chunks > 1:
        print(f"    0%", end="\r")
    current = 0
    for _ in range(chunks):
        chunk_data = array[:, current:min(current + chunk_size, cols)]
        result = scipy.signal.resample_poly(
            chunk_data, up, down, axis=0, padtype=padtype or 'minimum'
        )
        new_data.extend(result)
        current += chunk_size
        print(f"    {100 * current / cols:.1f}%", end="\r")

    return np.stack(new_data, axis=0) #.astype(np.int16)


def binarise(data):
    """
    This normalises an array to between 0 and 1 and then makes all values below 0.5
    equal to 0 and all values above 0.5 to 1. The array is returned as np.int8 to save
    some memory when using large datasets.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        If the data is a dataframe then each column will individually be binarised.

    """
    if isinstance(data, pd.DataFrame):
        for column in data.columns:
            data[column] = _binarise_real(data[column])
    else:
        data = _binarise_real(data)

    return data


def _binarise_real(data):
    data = (data - min(data)) / max(data)
    return (data > 0.5).astype(np.int8)


def find_sync_lag(array1, array2, plot=False):
    """
    Find the lag between two arrays where they have the greatest number of the same
    values. This functions assumes that the lag is less than 120,000 points.

    Parameters
    ----------
    array1 : array, Series or similar
        The first array. A positive result indicates that this array has leading data
        not present in the second array. e.g. if lag == 5 then array2 starts on the 5th
        index of array1.

    array2 : array, Series or similar
        The array to look for in the first.

    plot : string, optional
        False (default), or a path specifying where to save a png of the best match.  If
        it already exists, it will be suffixed with the time.

    Returns
    -------
    int : The lag between the starts of the two arrays. A positive number indicates that
        the first array begins earlier than the second.

    float : The percentage of values that were identical between the two arrays when
        aligned with the calculated lag, for the length compared.

    """
    length = min(len(array1), len(array2)) // 2
    length = min(length, 120000)

    array1 = array1.squeeze()
    array2 = array2.squeeze()

    sync_p = []
    for i in range(length):
        matches = np.count_nonzero(array1[i:i + length] == array2[:length])
        sync_p.append(100 * matches / length)
    match_p = max(sync_p)
    lag_p = sync_p.index(match_p)

    sync_n = []
    for i in range(length):
        matches = np.count_nonzero(array2[i:i + length] == array1[:length])
        sync_n.append(100 * matches / length)
    match_n = max(sync_n)
    lag_n = sync_n.index(match_n)

    if match_p > match_n:
        lag = lag_p
        match = match_p
    else:
        lag = - lag_n
        match = match_n

    if plot:
        plot = Path(plot)
        if plot.exists():
            plot = plot.with_name(plot.stem + '_' + time.strftime('%y%m%d-%H%M%S') + '.png')
        fig, axes = plt.subplots(nrows=2, ncols=1)
        plot_length = min(length, 5000)
        if lag >= 0:
            axes[0].plot(array1[lag:lag + plot_length])
            axes[1].plot(array2[:plot_length])
        else:
            axes[0].plot(array1[:plot_length])
            axes[1].plot(array2[-lag:-lag + plot_length])
        fig.savefig(plot)
        print(f"    Sync plot saved to:\n    {plot}")

    return lag, match


def median_subtraction(array, axis):
    """
    Subtract the medians of a given axis.

    Parameters
    ----------
    array : array, Series or similar
        The array to process.

    axis : int
        The axis in which to take the median and subtract.

    Returns
    -------
    array : The processed array with medians subtracted.

    """
    if not array.ndim > axis:
        raise PixelsError("Not enough dimensions to perform median subtraction.")

    #for i in range(array.shape[axis]):
    array = np.median(array, axis=axis, keepdims=True)
    #    array[:, i] = array[:, i] - np.median(array[:, i], axis=axis)

    return array


def convolve(times, duration, sigma=None):
    """
    Create a continuous signal from a set of spike times in milliseconds and convolve
    into a smooth firing rate signal.

    Parameters
    -------
    times : pandas.DataFrame
        Spike times in milliseconds to use to generate signal. Each column should
        correspond to one unit's spike times.

    duration : int
        Number of milliseconds of final signal.

    sigma : float/int, optional
        Time in milliseconds of sigma of gaussian kernel to use. Default is 50 ms.

    """
    if sigma is None:
        sigma = 50

    # turn into array of 0s and 1s
    times_arr = np.zeros((int(duration), len(times.columns)))
    for i, unit in enumerate(times):
        u_times = times[unit] + duration / 2
        u_times = u_times[~np.isnan(u_times)].astype(np.int)
        try:
            times_arr[u_times, i] = 1
        except IndexError:
            # sometimes the conversion to np.int can make spikes round to the index just
            # outside of the range
            u_times.values[-1] = u_times.values[-1] - 1

    # convolve and re-scale so units are per second
    convolved = gaussian_filter1d(times_arr, sigma, axis=0) * 1000
    df = pd.DataFrame(convolved, columns=times.columns)

    return df
